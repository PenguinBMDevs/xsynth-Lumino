//! B0 原型（仅测试构建）：跨 voice SoA 批处理内核 vs 真实逐声部链路。
//!
//! 目的：回答「lane = voice 的批处理能否显著快于现有时间维 SIMD 链路」。
//! 决策门：批内核 ≥1.3×（端到端预估 ≥15%）才进入 B1 集成。
//!
//! 基线 = 生产同款真实链路（`SIMDStereoVoiceSampler` 最近邻 + 常量增益 +
//! `SIMDVoiceEnvelope` + `SIMDStereoVoiceCutoff` biquad + `SIMDStereoVoice` 混音），
//! 所有 voice 混入同一通道缓冲（与 `channel/mod.rs::read_samples` 语义一致）。
//! 内核 = 手工 SoA 内核：lane = voice，逐时间步推进（biquad 依赖链跨 voice 并行），
//! 每步对 8 个 voice 做水平求和后标量累加进共享缓冲。
//!
//! 运行：`cargo test --release -p xsynth-core batch_proto -- --nocapture`
//!
//! 注：内核是计时模型，包络用线性衰减近似（真实包络为分段 lerp 状态机）；
//! 输出不做逐位对照，只比较 RMS 量级与耗时。

use std::sync::Arc;
use std::time::Instant;

use simdeez::prelude::*;
use simdeez::simd_runtime_generate;

use crate::effects::{BiQuadFilter, FilterType};
use crate::soundfont::EnvelopeOptions;
use crate::voice::{
    BufferSamplers, EnvelopeDescriptor, SIMDConstant, SIMDConstantStereo, SIMDNearestSampleGrabber,
    SIMDStereoVoice, SIMDStereoVoiceCutoff, SIMDStereoVoiceSampler, SIMDVoiceEnvelope,
    SampleReaderLoopSustain, VoiceCombineSIMD, VoiceSampleGenerator,
};

const FRAMES: usize = 480;
/// 每轮测量迭代次数（每迭代 = 480 帧 × 8 voice）。
const ITERS: usize = 50;
/// 交错测量轮数，取最小值以压制机器漂移。
const ROUNDS: usize = 5;
const SAMPLE_LEN: usize = 48_000;
const LOOP_START: usize = 1_000;
const LOOP_END: usize = 40_000;
const DECAY_SECS: f32 = 0.4;
const SUSTAIN: f32 = 0.55;
const SR: f32 = 48_000.0;
/// 运行时所有 SIMD 宽度（Scalar=1 … AVX512=16）的上界。
const MAX_LANES: usize = 16;

fn sample_buffer(seed: f32) -> Arc<[f32]> {
    (0..SAMPLE_LEN)
        .map(|i| {
            let t = i as f32 * 0.01 + seed;
            (t.sin() * 0.7 + (t * 2.13).sin() * 0.3) * 0.5
        })
        .collect::<Vec<_>>()
        .into()
}

fn rms(buf: &[f32]) -> f64 {
    let mut energy = 0.0f64;
    for &s in buf {
        energy += (s as f64) * (s as f64);
    }
    (energy / buf.len() as f64).sqrt()
}

/// SoA 批处理内核状态（lane = voice）。
struct KernelState<S: Simd> {
    time: [f64; MAX_LANES],
    speed: [f32; MAX_LANES],
    /// 包络进度（0→1 线性推进，近似真实 `SIMDLerper` 的 lerp + progress 推进）。
    env_progress: S::Vf32,
    env_step: S::Vf32,
    env_start: S::Vf32,
    env_span: S::Vf32,
    one: S::Vf32,
    gain_l: S::Vf32,
    gain_r: S::Vf32,
    x1l: S::Vf32,
    x2l: S::Vf32,
    y1l: S::Vf32,
    y2l: S::Vf32,
    x1r: S::Vf32,
    x2r: S::Vf32,
    y1r: S::Vf32,
    y2r: S::Vf32,
    b0: S::Vf32,
    b1: S::Vf32,
    b2: S::Vf32,
    a1: S::Vf32,
    a2: S::Vf32,
}

impl<S: Simd> KernelState<S> {
    /// 渲染 `frames` 帧进共享交错缓冲：外层时间步串行，内层跨 voice 并行。
    ///
    /// `MODE` 为阶段消融开关（const 折叠，无运行时开销）：
    /// - 0：仅采样标量循环（向量路径与混音均不执行）
    /// - 1：采样 + 向量装配 + 水平求和 + 混音
    /// - 2：+ 增益/包络
    /// - 3：+ biquad（完整内核）
    fn render<const MODE: u32>(&mut self, out: &mut [f32], l: &[f32], r: &[f32], frames: usize) {
        // 必须在 `simd_invoke!` 内执行：它是唯一建立 `#[target_feature(enable = "avx2,fma")]`
        // 的入口。泛型方法体本身没有 target_feature，直接调用 intrinsic 会退化成函数调用。
        simd_invoke!(S, {
            let lanes = S::Vf32::WIDTH;
            let mut sl_arr = [0.0f32; MAX_LANES];
            let mut sr_arr = [0.0f32; MAX_LANES];
            for step in 0..frames {
                // 1) 采样（逐 voice 标量，与真实抓取器同构：时间推进 + 回绕 + 最近邻加载）。
                unsafe {
                    for k in 0..lanes {
                        let t = *self.time.get_unchecked(k);
                        *self.time.get_unchecked_mut(k) = t + *self.speed.get_unchecked(k) as f64;
                        let mut idx = t as i32;
                        if idx > LOOP_END as i32 {
                            // 与 SampleReaderLoopSustain::get 相同的多圈回绕语义。
                            let span = (LOOP_END - LOOP_START) as i32;
                            let d = idx - LOOP_END as i32 - 1;
                            idx = LOOP_START as i32 + if d >= span { d % span } else { d };
                        }
                        let i = idx as usize;
                        // SAFETY: 回绕后 idx ∈ [LOOP_START, LOOP_END] ⊂ 缓冲范围。
                        *sl_arr.get_unchecked_mut(k) = *l.get_unchecked(i);
                        *sr_arr.get_unchecked_mut(k) = *r.get_unchecked(i);
                    }
                }
                if MODE == 0 {
                    let mut ls = 0.0f32;
                    let mut rs = 0.0f32;
                    for k in 0..lanes {
                        ls += unsafe { *sl_arr.get_unchecked(k) };
                        rs += unsafe { *sr_arr.get_unchecked(k) };
                    }
                    unsafe {
                        *out.get_unchecked_mut(2 * step) += ls;
                        *out.get_unchecked_mut(2 * step + 1) += rs;
                    }
                    continue;
                }
                let mut sl = S::Vf32::load_from_slice(&sl_arr);
                let mut sr = S::Vf32::load_from_slice(&sr_arr);
                if MODE >= 2 {
                    // 2) 增益 → 包络（与真实链顺序一致）。
                    // 包络用真实 `SIMDLerper` 同构的「progress 推进 + lerp」：
                    // progress = min(progress + step, 1)；env = start + (target - start) * progress。
                    self.env_progress = (self.env_progress + self.env_step).min(self.one);
                    let env = self.env_start - self.env_span * self.env_progress;
                    sl = sl * self.gain_l * env;
                    sr = sr * self.gain_r * env;
                }
                if MODE >= 3 {
                    // 3) biquad：lane = voice，跨 voice 的依赖链消失（DF1，与真实实现同公式）。
                    let ol = self.b0 * sl + self.b1 * self.x1l + self.b2 * self.x2l
                        - self.a1 * self.y1l
                        - self.a2 * self.y2l;
                    self.x2l = self.x1l;
                    self.x1l = sl;
                    self.y2l = self.y1l;
                    self.y1l = ol;
                    let or = self.b0 * sr + self.b1 * self.x1r + self.b2 * self.x2r
                        - self.a1 * self.y1r
                        - self.a2 * self.y2r;
                    self.x2r = self.x1r;
                    self.x1r = sr;
                    self.y2r = self.y1r;
                    self.y1r = or;
                    sl = ol;
                    sr = or;
                }
                // 4) 混音：8 个 voice 水平求和 → 共享缓冲（生产语义：同通道同缓冲）。
                let lsum = sl.horizontal_add();
                let rsum = sr.horizontal_add();
                unsafe {
                    *out.get_unchecked_mut(2 * step) += lsum;
                    *out.get_unchecked_mut(2 * step + 1) += rsum;
                }
            }
        });
    }
}

#[test]
fn batch_proto_kernel_vs_per_voice() {
    simd_runtime_generate!(
        fn run() {
            let lanes = S::Vf32::WIDTH;
            let left = sample_buffer(0.0);
            let right = sample_buffer(1.7);

            let params = EnvelopeDescriptor {
                start_percent: 0.0,
                delay: 0.0,
                attack: 0.01,
                hold: 0.0,
                decay: DECAY_SECS,
                sustain_percent: SUSTAIN,
                release: 0.5,
            }
            .to_envelope_params(SR as u32, EnvelopeOptions::default());

            // ── 基线：真实逐声部链路（时间维 SIMD），共享输出缓冲。
            //    stages: 1=采样+混音 2=+增益/包络 3=+biquad（完整，生产链路）──
            let mk_voice = |speed: f32, stages: u32| -> Box<dyn VoiceSampleGenerator> {
                let l_reader = SampleReaderLoopSustain::new_raw(
                    BufferSamplers::new_f32(left.clone()),
                    0,
                    LOOP_START,
                    LOOP_END,
                );
                let r_reader = SampleReaderLoopSustain::new_raw(
                    BufferSamplers::new_f32(right.clone()),
                    0,
                    LOOP_START,
                    LOOP_END,
                );
                let l_grab = SIMDNearestSampleGrabber::<S, _>::new(l_reader);
                let r_grab = SIMDNearestSampleGrabber::<S, _>::new(r_reader);
                let pitch = SIMDConstant::<S>::new(speed);
                let sampler = SIMDStereoVoiceSampler::<S, _, _>::new(l_grab, r_grab, pitch, false);
                if stages <= 1 {
                    return Box::new(SIMDStereoVoice::<S, _>::new(sampler, false));
                }
                let gain = SIMDConstantStereo::<S>::new(0.5, 0.5);
                let gen = VoiceCombineSIMD::<S>::mult(gain, sampler);
                let env = SIMDVoiceEnvelope::<S>::new(params, params, false, SR, false);
                let gen = VoiceCombineSIMD::<S>::mult(env, gen);
                if stages == 2 {
                    return Box::new(SIMDStereoVoice::<S, _>::new(gen, false));
                }
                let filter = BiQuadFilter::new(FilterType::LowPass, 9000.0, SR, Some(0.7));
                let gen = SIMDStereoVoiceCutoff::<S, _>::new(gen, &filter, false);
                Box::new(SIMDStereoVoice::<S, _>::new(gen, false))
            };
            let mut variant_voices: Vec<Vec<Box<dyn VoiceSampleGenerator>>> = Vec::new();
            for stages in 1..=3u32 {
                let mut vs: Vec<Box<dyn VoiceSampleGenerator>> = Vec::new();
                for k in 0..lanes {
                    vs.push(mk_voice(1.0 + k as f32 * 0.01, stages));
                }
                variant_voices.push(vs);
            }
            let mut baseline_buf = vec![0.0f32; FRAMES * 2];

            // ── 内核：同参数 SoA 状态 ──
            let filter = BiQuadFilter::new(FilterType::LowPass, 9000.0, SR, Some(0.7));
            let c = filter.coefficients();
            let mk_kernel = || {
                let mut k = KernelState::<S> {
                    time: [0.0; MAX_LANES],
                    speed: [0.0; MAX_LANES],
                    env_progress: S::Vf32::zeroes(),
                    env_step: S::Vf32::set1(1.0 / (DECAY_SECS * SR)),
                    env_start: S::Vf32::set1(1.0),
                    env_span: S::Vf32::set1(1.0 - SUSTAIN),
                    one: S::Vf32::set1(1.0),
                    gain_l: S::Vf32::set1(0.5),
                    gain_r: S::Vf32::set1(0.5),
                    x1l: S::Vf32::zeroes(),
                    x2l: S::Vf32::zeroes(),
                    y1l: S::Vf32::zeroes(),
                    y2l: S::Vf32::zeroes(),
                    x1r: S::Vf32::zeroes(),
                    x2r: S::Vf32::zeroes(),
                    y1r: S::Vf32::zeroes(),
                    y2r: S::Vf32::zeroes(),
                    b0: S::Vf32::set1(c.b0),
                    b1: S::Vf32::set1(c.b1),
                    b2: S::Vf32::set1(c.b2),
                    a1: S::Vf32::set1(c.a1),
                    a2: S::Vf32::set1(c.a2),
                };
                for i in 0..lanes {
                    k.time[i] = i as f64 * 7.3;
                    k.speed[i] = 1.0 + i as f32 * 0.01;
                }
                k
            };
            let mut kernel_buf = vec![0.0f32; FRAMES * 2];

            // ── 测量：min-of-N 交错（压制机器漂移；每轮重建内核状态）──
            fn bench_round<F: FnMut()>(iters: usize, f: &mut F) -> f64 {
                let t = Instant::now();
                for _ in 0..iters {
                    f();
                }
                t.elapsed().as_nanos() as f64 / iters as f64
            }

            let mut best_base = [f64::MAX; 3];
            let mut best_kern = [f64::MAX; 4];
            for round in 0..ROUNDS {
                macro_rules! bench_baseline_variant {
                    ($v:expr, $best:expr) => {{
                        let voices = &mut variant_voices[$v];
                        let mut f = || {
                            baseline_buf.fill(0.0);
                            for voice in voices.iter_mut() {
                                voice.render_to(&mut baseline_buf);
                            }
                        };
                        if round == 0 {
                            f();
                        }
                        $best = $best.min(bench_round(ITERS, &mut f));
                    }};
                }
                macro_rules! bench_kernel_mode {
                    ($mode:literal, $best:expr) => {{
                        let mut k = mk_kernel();
                        let mut f = || {
                            kernel_buf.fill(0.0);
                            k.render::<$mode>(&mut kernel_buf, &left, &right, FRAMES);
                        };
                        if round == 0 {
                            f();
                        }
                        $best = $best.min(bench_round(ITERS, &mut f));
                    }};
                }
                bench_baseline_variant!(0, best_base[0]);
                bench_kernel_mode!(0, best_kern[0]);
                bench_baseline_variant!(1, best_base[1]);
                bench_kernel_mode!(1, best_kern[1]);
                bench_baseline_variant!(2, best_base[2]);
                bench_kernel_mode!(2, best_kern[2]);
                bench_kernel_mode!(3, best_kern[3]);
            }

            // ── 输出（per-voice 已按 lanes 归一）──
            println!("[B0] lanes={lanes} frames={FRAMES} iters={ITERS} rounds={ROUNDS}");
            println!(
                "[B0] baseline per-voice ns: sampler+mix={:.0} +gain/env={:.0} +biquad_full={:.0}",
                best_base[0] / lanes as f64,
                best_base[1] / lanes as f64,
                best_base[2] / lanes as f64
            );
            for mode in 0..4usize {
                println!(
                    "[B0] kernel mode={mode} per-voice={:.0}ns",
                    best_kern[mode] / lanes as f64
                );
            }
            println!(
                "[B0] speedup full-kernel vs full-baseline = {:.2}x",
                best_base[2] / best_kern[3]
            );

            // 输出量级 sanity（内核为计时模型，不逐位对照）。
            baseline_buf.fill(0.0);
            for voice in variant_voices[2].iter_mut() {
                voice.render_to(&mut baseline_buf);
            }
            let mut k = mk_kernel();
            kernel_buf.fill(0.0);
            k.render::<3>(&mut kernel_buf, &left, &right, FRAMES);
            println!(
                "[B0] rms: baseline={:.4} kernel={:.4}",
                rms(&baseline_buf),
                rms(&kernel_buf)
            );
        }
    );

    run();
}
