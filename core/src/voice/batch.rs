//! B1：跨 voice SoA 批处理渲染（lane = voice）。
//!
//! # 架构
//!
//! `StereoBatchVoice` 在 spawn 期由 `StereoSampledVoiceSpawner` 直接构造：lane 状态
//! 就是该 voice 的**权威状态**，不再保留原链式生成器。渲染时 `VoiceChannel` 把连续的
//! 可批 voice 按运行时 SIMD 宽度（本机 AVX2 = 8）分块交给 [`render_batch_chunk`]：
//!
//! - 标量部分：逐 lane 推进采样（时间推进 + 回绕 + 抓取）、包络（7 阶段状态机）、增益；
//! - 向量部分：跨 voice 并行完成 biquad（DF1）。时间维串行依赖链 → 跨 voice 并行，
//!   这是 B0 决策门测出的结构性收益来源（biquad 阶段 4.8-6.4×）；
//! - 混音：`horizontal_add` 求和后一次 `+=` 进通道缓冲（与 B0 原型语义一致）。
//!
//! # 逐位等价口径（B1 决策门）
//!
//! - **单 lane 信号链逐位一致**：采样/包络/增益/biquad 的每一步运算顺序与标量实现
//!   完全相同，并用差分测试逐位验证（见本文件 tests）：
//!   `BatchLane::render_scalar` ≡ 真实链式生成器；内核单 lane ≡ `render_scalar`。
//! - 包络复刻真实实现的「8 帧组」机制：组内是否走 SIMD 曲线公式还是逐帧 scalar 公式，
//!   取决于组起点（`t + WIDTH - 1 >= end`），与真实实现完全一致，因此 concave/convex
//!   曲线在阶段末尾的取值切换点也逐位相同。
//! - **混音求和顺序不同**：批内 8 lane 经 `horizontal_add` 树形求和，与逐 voice 顺序
//!   累加在 f32 下不满足结合律 → 与 `LUMINO_BATCH=0` 存在 ulp 级差异（端到端由
//!   1e-6 容差 + RMS/峰值对照覆盖）；只有「单 lane 独占一个 batch」时逐位一致。
//!
//! # 开关
//!
//! `LUMINO_BATCH=1`（默认关）。关闭时 spawner 走原链路，渲染路径完全不变。

use std::sync::{Arc, OnceLock};

use simdeez::prelude::*;
use simdeez::simd_runtime_generate;
use xsynth_soundfonts::LoopMode;

use crate::effects::BiQuadFilter;
use crate::soundfont::Interpolator;
use crate::voice::{
    BufferSampler, BufferSamplers, EnvelopeControlData, EnvelopeParameters, EnvelopePart,
    EnvelopeStage, ReleaseType, Voice, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator,
};

/// 运行时所有 SIMD 宽度（Scalar=1 … AVX512=16）的上界。
const MAX_LANES: usize = 16;
/// 启用批渲染所需的最小 SIMD 宽度（更低宽度下跨 voice 并行收益不足）。
const MIN_BATCH_WIDTH: usize = 8;

/// 批处理开关：`LUMINO_BATCH` 存在且不为 `0`/`false` 时启用（默认关）。
pub(crate) fn batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LUMINO_BATCH").is_some_and(|v| {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        })
    })
}

/// 运行时 SIMD 宽度 = 一个 batch chunk 的 lane 数（本机 AVX2 = 8）。
pub(crate) fn batch_chunk_width() -> usize {
    static WIDTH: OnceLock<usize> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        simd_runtime_generate!(
            fn width() -> usize {
                S::Vf32::WIDTH
            }
        );
        width()
    })
}

/// 批渲染是否可用：开关打开 + SIMD 宽度足够。
pub(crate) fn batching_supported() -> bool {
    batch_enabled() && batch_chunk_width() >= MIN_BATCH_WIDTH
}

/// spawn 期传入的静态参数（由 `StereoSampledVoiceSpawner` 填充）。
pub(crate) struct BatchLaneInit {
    pub speed_mult: f32,
    pub gain_l: f32,
    pub gain_r: f32,
    pub samples_l: Arc<[f32]>,
    pub samples_r: Arc<[f32]>,
    pub loop_mode: LoopMode,
    pub loop_offset: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_stop: Option<usize>,
    pub interpolator: Interpolator,
    pub filter: Option<BiQuadFilter>,
    pub envelope: EnvelopeParameters,
    pub sample_rate: f32,
    pub group_len: u8,
    pub velocity: u8,
    pub exclusive_class: Option<u8>,
}

/// 一侧采样读取器状态（镜像 `SampleReader*` 的动态字段）。
struct SideReader {
    buffer: BufferSamplers,
    offset: usize,
    /// `SampleReaderLoopSustain` 的 `last`（最后一次未释放时的索引，含 offset）。
    last: usize,
    /// `loop_params.stop` 或缓冲长度（未设置 stop 时）。
    length: Option<usize>,
}

/// 包络曲线（标量版 `StageData`）。
#[derive(Clone, Copy)]
enum LaneCurve {
    /// `start + length * factor`（`SIMDLerper`）
    Lerp { start: f32, length: f32 },
    /// `length * (1 - factor)^8 + end`（`SIMDLerperConcave`）
    Concave { length: f32, end: f32 },
    /// `length * factor^8 + start`（`SIMDLerperConvex`）
    Convex { start: f32, length: f32 },
    /// `EnvelopePart::Hold(value)`
    Constant(f32),
}

/// 逐 lane 包络状态机（镜像 `SIMDVoiceEnvelope`）。
///
/// 关键：真实实现以「8 帧组」为单位推进（`next_sample` 一次产生 `WIDTH` 个值），
/// 组内若跨越阶段末尾则退回逐帧 scalar 公式（`manually_build_simd_sample`）。
/// 两种路径的 concave/convex 曲线公式**写法不同**（`powi(8)` vs 三次平方），
/// 因此必须复刻组的划分方式才能逐位一致。
struct LaneEnvelope {
    original: EnvelopeParameters,
    params: EnvelopeParameters,
    sample_rate: f32,
    allow_release: bool,
    killed: bool,
    stage: EnvelopeStage,
    data: LaneCurve,
    /// 当前阶段内已产生的帧数（真实实现的 `StageTime::simd_array_start`，lane 0）。
    t: u32,
    /// 当前阶段时长（`Constant` 阶段为 0）。
    end: u32,
    /// 组大小（= 运行时 SIMD 宽度）。
    group_len: u8,
    /// 当前组剩余帧数（`group_len` … 1）。
    group_left: u8,
    /// 当前组是否已进入逐帧 scalar 公式路径。
    group_manual: bool,
}

impl LaneEnvelope {
    fn new(
        original: EnvelopeParameters,
        params: EnvelopeParameters,
        allow_release: bool,
        sample_rate: f32,
        group_len: u8,
    ) -> LaneEnvelope {
        let mut env = LaneEnvelope {
            original,
            params,
            sample_rate,
            allow_release,
            killed: false,
            stage: EnvelopeStage::Delay,
            data: LaneCurve::Constant(0.0),
            t: 0,
            end: 0,
            group_len,
            group_left: group_len,
            group_manual: false,
        };
        let start = params.start_amplitude();
        env.set_stage(EnvelopeStage::Delay, start);
        env
    }

    /// 镜像 `EnvelopeParameters::get_stage_data`：`duration == 0` 递归到下一阶段。
    fn set_stage(&mut self, mut stage: EnvelopeStage, mut amp: f32) {
        loop {
            match self.params.parts[stage.as_usize()] {
                EnvelopePart::Lerp { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Lerp {
                        start: amp,
                        length: target - amp,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::LerpConcave { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Concave {
                        length: amp - target,
                        end: target,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::LerpConvex { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Convex {
                        start: amp,
                        length: target - amp,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::Hold(value) => {
                    self.data = LaneCurve::Constant(value);
                    self.t = 0;
                    self.end = 0;
                }
            }
            self.stage = stage;
            return;
        }
    }

    #[inline(always)]
    fn factor(&self) -> f32 {
        self.t as f32 / self.end as f32
    }

    /// 逐帧 scalar 公式（真实路径 `get_value_at_current_time` / 手动组路径）。
    #[inline(always)]
    fn value_scalar(&self) -> f32 {
        let factor = self.factor();
        match self.data {
            LaneCurve::Lerp { start, length } => start + length * factor,
            LaneCurve::Concave { length, end } => {
                let mult = (1.0 - factor).powi(8);
                length * mult + end
            }
            LaneCurve::Convex { start, length } => {
                let mult = factor.powi(8);
                length * mult + start
            }
            LaneCurve::Constant(value) => value,
        }
    }

    /// 组内 SIMD 公式（真实路径 `lerp_simd`：显式三次平方，非 `powi`）。
    #[inline(always)]
    fn value_simd(&self) -> f32 {
        let factor = self.factor();
        match self.data {
            LaneCurve::Lerp { start, length } => start + length * factor,
            LaneCurve::Concave { length, end } => {
                let r1 = 1.0 - factor;
                let r2 = r1 * r1;
                let r3 = r2 * r2;
                let mult = r3 * r3;
                length * mult + end
            }
            LaneCurve::Convex { start, length } => {
                let r1 = factor * factor;
                let r2 = r1 * r1;
                let mult = r2 * r2;
                length * mult + start
            }
            LaneCurve::Constant(value) => value,
        }
    }

    #[inline(always)]
    fn advance_group(&mut self) {
        if self.group_left <= 1 {
            self.group_left = self.group_len;
        } else {
            self.group_left -= 1;
        }
    }

    fn switch_stage(&mut self) {
        let amp = self.value_scalar();
        self.set_stage(self.stage.next_stage(), amp);
    }

    /// 镜像 `SIMDVoiceEnvelope::update_stage`（控制消息改包络时保留当前阶段与当前幅度）。
    fn update_stage(&mut self) {
        let amp = self.value_scalar();
        self.set_stage(self.stage, amp);
    }

    /// 产生下一帧的包络值（镜像 `next_sample_inner` + `manually_build_simd_sample`）。
    #[inline(always)]
    fn next_frame(&mut self) -> f32 {
        loop {
            match self.data {
                LaneCurve::Constant(value) => {
                    // `StageData::Constant` 分支不推进阶段时间。
                    self.advance_group();
                    return value;
                }
                _ => {
                    if self.t >= self.end {
                        // 组起点已越过阶段末尾：切换阶段，不消耗帧。
                        self.switch_stage();
                        continue;
                    }
                    if self.group_left == self.group_len {
                        // 组入口：按当前阶段判定本组是否退回逐帧 scalar 公式。
                        self.group_manual = self.t + (self.group_len as u32 - 1) >= self.end;
                    }
                    let value = if self.group_manual {
                        self.value_scalar()
                    } else {
                        self.value_simd()
                    };
                    self.t += 1;
                    if self.group_manual && self.t >= self.end {
                        self.switch_stage();
                    }
                    self.advance_group();
                    return value;
                }
            }
        }
    }

    fn signal_release(&mut self, rel_type: ReleaseType) {
        if rel_type == ReleaseType::Kill {
            self.params.modify_stage_data(
                5,
                EnvelopePart::lerp(0.0, (0.001 * self.sample_rate) as u32),
            );
            self.update_stage();
            self.killed = true;
        }
        if self.allow_release || self.killed {
            let amp = self.value_scalar();
            self.set_stage(EnvelopeStage::Release, amp);
        }
    }

    fn modify_envelope(&mut self, envelope: EnvelopeControlData) {
        if !self.killed {
            self.params = self
                .original
                .with_envelope_control(envelope, self.sample_rate);
            self.update_stage();
        }
    }

    #[inline(always)]
    fn ended(&self) -> bool {
        self.stage == EnvelopeStage::Finished
    }
}

/// 一个可批 voice 的全部状态（lane = voice）。
pub struct BatchLane {
    // ── 采样 ──
    left: SideReader,
    right: SideReader,
    loop_mode: LoopMode,
    loop_start: usize,
    loop_end: usize,
    interpolator: Interpolator,
    time: f64,
    speed: f32,
    /// spawn 期速度倍率（`SIMDConstantControl` 的 `base`，控制消息变化时重算 speed）。
    speed_mult: f32,
    is_released: bool,
    // ── 包络 ──
    env: LaneEnvelope,
    // ── 增益（力度 × 等功率声像，spawn 期合并）──
    gain_l: f32,
    gain_r: f32,
    // ── 滤波（biquad DF1，无滤波器时 filter_enabled = false）──
    filter_enabled: bool,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1l: f32,
    x2l: f32,
    y1l: f32,
    y2l: f32,
    x1r: f32,
    x2r: f32,
    y1r: f32,
    y2r: f32,
    // ── 治理/生命周期 ──
    releasing: bool,
    killed: bool,
    velocity: u8,
    exclusive_class: Option<u8>,
}

impl BatchLane {
    pub(crate) fn new(init: &BatchLaneInit, control: &VoiceControlData) -> BatchLane {
        let length_l = Some(init.loop_stop.unwrap_or(init.samples_l.len()));
        let length_r = Some(init.loop_stop.unwrap_or(init.samples_r.len()));
        let modified = init
            .envelope
            .with_envelope_control(control.envelope, init.sample_rate);
        let allow_release = init.loop_mode != LoopMode::OneShot;

        let mut lane = BatchLane {
            left: SideReader {
                buffer: BufferSamplers::new_f32(init.samples_l.clone()),
                offset: init.loop_offset,
                last: 0,
                length: length_l,
            },
            right: SideReader {
                buffer: BufferSamplers::new_f32(init.samples_r.clone()),
                offset: init.loop_offset,
                last: 0,
                length: length_r,
            },
            loop_mode: init.loop_mode,
            loop_start: init.loop_start,
            loop_end: init.loop_end,
            interpolator: init.interpolator,
            time: 0.0,
            // 镜像 `SIMDConstantControl::new(speed_mult, control, |vc| vc.voice_pitch_multiplier)`。
            speed: init.speed_mult * control.voice_pitch_multiplier,
            speed_mult: init.speed_mult,
            is_released: false,
            env: LaneEnvelope::new(
                init.envelope,
                modified,
                allow_release,
                init.sample_rate,
                init.group_len,
            ),
            gain_l: init.gain_l,
            gain_r: init.gain_r,
            filter_enabled: init.filter.is_some(),
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1l: 0.0,
            x2l: 0.0,
            y1l: 0.0,
            y2l: 0.0,
            x1r: 0.0,
            x2r: 0.0,
            y1r: 0.0,
            y2r: 0.0,
            releasing: false,
            killed: false,
            velocity: init.velocity,
            exclusive_class: init.exclusive_class,
        };
        if let Some(filter) = &init.filter {
            let c = filter.coefficients();
            lane.b0 = c.b0;
            lane.b1 = c.b1;
            lane.b2 = c.b2;
            lane.a1 = c.a1;
            lane.a2 = c.a2;
        }
        lane
    }

    /// 该 lane 是否走 biquad（同一 chunk 内必须一致，否则无法整块向量化）。
    #[inline(always)]
    pub(crate) fn filter_enabled(&self) -> bool {
        self.filter_enabled
    }

    /// 一侧的采样读取（逐位镜像 `SampleReaderLoop`/`SampleReaderLoopSustain`/`SampleReaderNoLoop`）。
    #[inline(always)]
    fn read_side(
        side: &mut SideReader,
        loop_mode: LoopMode,
        loop_start: usize,
        loop_end: usize,
        is_released: bool,
        index: usize,
    ) -> f32 {
        let mut pos = index + side.offset;
        match loop_mode {
            LoopMode::NoLoop | LoopMode::OneShot => {}
            LoopMode::LoopContinuous => {
                if pos > loop_end {
                    let span = loop_end - loop_start;
                    let d = pos - loop_end - 1;
                    pos = loop_start + if d >= span { d % span } else { d };
                }
            }
            LoopMode::LoopSustain => {
                if !is_released {
                    side.last = pos;
                    if pos > loop_end {
                        let span = loop_end - loop_start;
                        let d = pos - loop_end - 1;
                        pos = loop_start + if d >= span { d % span } else { d };
                    }
                } else {
                    // 释放后：冻结 last、按剩余偏移线性推进（release 模式回绕语义，
                    // 用 wrapping 运算避免 debug 下 usize 下溢 panic，与 release 一致）。
                    pos = pos.wrapping_sub(side.last).wrapping_add(loop_end);
                }
            }
        }
        side.buffer.get(pos)
    }

    #[inline(always)]
    fn read_left(&mut self, index: usize) -> f32 {
        Self::read_side(
            &mut self.left,
            self.loop_mode,
            self.loop_start,
            self.loop_end,
            self.is_released,
            index,
        )
    }

    #[inline(always)]
    fn read_right(&mut self, index: usize) -> f32 {
        Self::read_side(
            &mut self.right,
            self.loop_mode,
            self.loop_start,
            self.loop_end,
            self.is_released,
            index,
        )
    }

    /// 采样 → 包络 → 增益（标量，逐帧；与真实链式顺序一致）。
    #[inline(always)]
    fn next_pre_filter(&mut self) -> (f32, f32) {
        let t = self.time;
        self.time = t + self.speed as f64;
        let index = t as i32;
        let (sl, sr) = match self.interpolator {
            Interpolator::Nearest => {
                let l = self.read_left(index as usize);
                let r = self.read_right(index as usize);
                (l, r)
            }
            Interpolator::Linear => {
                // 与 `SIMDLinearSampleGrabber::get` 一致：先取 index 再取 index+1
                // （`SampleReaderLoopSustain` 的 last 会被两次读取更新）。
                let frac = (t - index as f64) as f32;
                let l0 = self.read_left(index as usize);
                let l1 = self.read_left(index as usize + 1);
                let r0 = self.read_right(index as usize);
                let r1 = self.read_right(index as usize + 1);
                (l0 * (1.0 - frac) + l1 * frac, r0 * (1.0 - frac) + r1 * frac)
            }
        };
        let env = self.env.next_frame();
        // 真实链顺序：gain 常量 × 采样 → 包络 × 结果。
        (env * (self.gain_l * sl), env * (self.gain_r * sr))
    }

    /// biquad（DF1，与 `BiQuadFilter::process` 逐项一致）。
    #[inline(always)]
    fn filter_frame(&mut self, l: f32, r: f32) -> (f32, f32) {
        if !self.filter_enabled {
            return (l, r);
        }
        let ol = self.b0 * l + self.b1 * self.x1l + self.b2 * self.x2l
            - self.a1 * self.y1l
            - self.a2 * self.y2l;
        self.x2l = self.x1l;
        self.x1l = l;
        self.y2l = self.y1l;
        self.y1l = ol;

        let or = self.b0 * r + self.b1 * self.x1r + self.b2 * self.x2r
            - self.a1 * self.y1r
            - self.a2 * self.y2r;
        self.x2r = self.x1r;
        self.x1r = r;
        self.y2r = self.y1r;
        self.y1r = or;

        (ol, or)
    }

    /// 标量渲染（单 lane），用于：非批路径（如通道使用 key 级线程池时）/ 尾部不足
    /// 一个 chunk 的 voice。逐位等价于真实链式生成器。
    pub(crate) fn render_scalar(&mut self, buffer: &mut [f32]) {
        let frames = buffer.len() / 2;
        for f in 0..frames {
            let (l, r) = self.next_pre_filter();
            let (l, r) = self.filter_frame(l, r);
            buffer[2 * f] += l;
            buffer[2 * f + 1] += r;
        }
    }

    #[inline(always)]
    fn past_end_side(&self, side: &SideReader, pos: usize) -> bool {
        match self.loop_mode {
            LoopMode::NoLoop | LoopMode::OneShot => match side.length {
                Some(len) => pos >= len.saturating_sub(side.offset),
                None => false,
            },
            // `SampleReaderLoop::is_past_end` 恒为 false；
            // `SampleReaderLoopSustain` 用 stop/length 与冻结的 last 判定（原样镜像）。
            LoopMode::LoopContinuous => false,
            LoopMode::LoopSustain => match side.length {
                Some(len) => pos >= len.saturating_sub(side.last.saturating_add(side.offset)),
                None => false,
            },
        }
    }

    /// 镜像 `SIMDStereoVoiceSampler::ended`。
    #[inline(always)]
    fn past_end(&self) -> bool {
        let pos = self.time as usize;
        self.past_end_side(&self.left, pos) || self.past_end_side(&self.right, pos)
    }

    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.releasing = true;
        if rel_type == ReleaseType::Kill {
            self.killed = true;
        }
        self.env.signal_release(rel_type);
        // 镜像 `SIMDSampleGrabber::signal_release` → reader：仅 `SampleReaderLoopSustain`
        // 有状态（Loop/NoLoop 为空实现），且与 rel_type / allow_release 无关。
        if self.loop_mode == LoopMode::LoopSustain {
            self.is_released = true;
        }
    }

    fn process_controls(&mut self, control: &VoiceControlData) {
        self.speed = self.speed_mult * control.voice_pitch_multiplier;
        self.env.modify_envelope(control.envelope);
    }
}

/// 标量渲染入口（内核单 chunk）。
///
/// 要求 `lanes.len()` 等于运行时 SIMD 宽度，且所有 lane 的 `filter_enabled` 一致。
pub(crate) fn render_batch_chunk(
    lanes: &mut [&mut BatchLane],
    out: &mut [f32],
    frames: usize,
) -> bool {
    simd_runtime_generate!(
        fn render(lanes: &mut [&mut BatchLane], out: &mut [f32], frames: usize) -> bool {
            if lanes.len() != S::Vf32::WIDTH {
                return false;
            }
            render_inner::<S>(lanes, out, frames);
            true
        }
    );

    render(lanes, out, frames)
}

fn render_inner<S: Simd>(lanes: &mut [&mut BatchLane], out: &mut [f32], frames: usize) {
    let width = S::Vf32::WIDTH;
    debug_assert_eq!(lanes.len(), width);
    let filter_on = lanes[0].filter_enabled;

    // 必须在 `simd_invoke!` 内执行：它是唯一建立 `#[target_feature(enable = "avx2,fma")]`
    // 的入口（B0 关键坑：泛型方法体直接调用 intrinsic 会退化为函数调用，实测 0.55× → 3.48×）。
    simd_invoke!(S, {
        let mut sl_arr = [0.0f32; MAX_LANES];
        let mut sr_arr = [0.0f32; MAX_LANES];

        macro_rules! load_lane_vec {
            ($field:ident) => {{
                let mut arr = [0.0f32; MAX_LANES];
                for k in 0..width {
                    arr[k] = lanes[k].$field;
                }
                S::Vf32::load_from_slice(&arr)
            }};
        }
        macro_rules! store_lane_vec {
            ($field:ident, $value:expr) => {{
                let mut arr = [0.0f32; MAX_LANES];
                $value.copy_to_slice(&mut arr);
                for k in 0..width {
                    lanes[k].$field = arr[k];
                }
            }};
        }

        // 滤波状态/系数装载到向量寄存器（每 chunk 一次，热循环内不再碰内存）。
        let b0 = load_lane_vec!(b0);
        let b1 = load_lane_vec!(b1);
        let b2 = load_lane_vec!(b2);
        let a1 = load_lane_vec!(a1);
        let a2 = load_lane_vec!(a2);
        let mut x1l = load_lane_vec!(x1l);
        let mut x2l = load_lane_vec!(x2l);
        let mut y1l = load_lane_vec!(y1l);
        let mut y2l = load_lane_vec!(y2l);
        let mut x1r = load_lane_vec!(x1r);
        let mut x2r = load_lane_vec!(x2r);
        let mut y1r = load_lane_vec!(y1r);
        let mut y2r = load_lane_vec!(y2r);

        for step in 0..frames {
            // 1) 标量：逐 lane 采样 + 包络 + 增益。
            unsafe {
                for k in 0..width {
                    let lane = lanes.get_unchecked_mut(k);
                    let (l, r) = lane.next_pre_filter();
                    *sl_arr.get_unchecked_mut(k) = l;
                    *sr_arr.get_unchecked_mut(k) = r;
                }
            }
            let mut sl = S::Vf32::load_from_slice(&sl_arr);
            let mut sr = S::Vf32::load_from_slice(&sr_arr);

            // 2) 向量：biquad（lane = voice，时间维依赖链消失）。
            if filter_on {
                let ol = b0 * sl + b1 * x1l + b2 * x2l - a1 * y1l - a2 * y2l;
                x2l = x1l;
                x1l = sl;
                y2l = y1l;
                y1l = ol;
                let or = b0 * sr + b1 * x1r + b2 * x2r - a1 * y1r - a2 * y2r;
                x2r = x1r;
                x1r = sr;
                y2r = y1r;
                y1r = or;
                sl = ol;
                sr = or;
            }

            // 3) 混音：水平求和后一次写入共享通道缓冲。
            let lsum = sl.horizontal_add();
            let rsum = sr.horizontal_add();
            unsafe {
                *out.get_unchecked_mut(2 * step) += lsum;
                *out.get_unchecked_mut(2 * step + 1) += rsum;
            }
        }

        store_lane_vec!(x1l, x1l);
        store_lane_vec!(x2l, x2l);
        store_lane_vec!(y1l, y1l);
        store_lane_vec!(y2l, y2l);
        store_lane_vec!(x1r, x1r);
        store_lane_vec!(x2r, x2r);
        store_lane_vec!(y1r, y1r);
        store_lane_vec!(y2r, y2r);
    });
}

/// 批处理 voice：持有 [`BatchLane`]，`Voice` 接口直通 lane 状态。
pub(crate) struct StereoBatchVoice {
    lane: BatchLane,
}

impl StereoBatchVoice {
    pub(crate) fn new(init: &BatchLaneInit, control: &VoiceControlData) -> StereoBatchVoice {
        StereoBatchVoice {
            lane: BatchLane::new(init, control),
        }
    }
}

impl VoiceGeneratorBase for StereoBatchVoice {
    #[inline(always)]
    fn ended(&self) -> bool {
        self.lane.env.ended() || self.lane.past_end()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.lane.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.lane.process_controls(control);
    }
}

impl VoiceSampleGenerator for StereoBatchVoice {
    fn render_to(&mut self, buffer: &mut [f32]) {
        self.lane.render_scalar(buffer);
    }
}

impl Voice for StereoBatchVoice {
    #[inline(always)]
    fn is_releasing(&self) -> bool {
        self.lane.releasing
    }

    #[inline(always)]
    fn is_killed(&self) -> bool {
        self.lane.killed
    }

    #[inline(always)]
    fn velocity(&self) -> u8 {
        self.lane.velocity
    }

    #[inline(always)]
    fn exclusive_class(&self) -> Option<u8> {
        self.lane.exclusive_class
    }

    #[inline(always)]
    fn batch_lane(&mut self) -> Option<&mut BatchLane> {
        Some(&mut self.lane)
    }
}
