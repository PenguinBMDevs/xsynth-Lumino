use std::{marker::PhantomData, ops::Mul, sync::Arc};

use simdeez::Simd;

use crate::{
    effects::BiQuadFilter,
    voice::{
        BufferSampler, SIMDSample, SIMDSampleGrabber, SIMDSampleMono, SIMDSampleStereo,
        SIMDStereoVoiceCutoff, SIMDVoiceGenerator,
    },
    AudioStreamParams,
};
use crate::{
    voice::VoiceControlData,
    voice::{
        BatchLaneInit, BufferSamplers, EnvelopeParameters, SIMDConstantControl, SIMDConstantStereo,
        SIMDLinearSampleGrabber, SIMDNearestSampleGrabber, SIMDStereoVoice, SIMDStereoVoiceSampler,
        SIMDVoiceEnvelope, SampleReader, SampleReaderLoop, SampleReaderLoopSustain,
        SampleReaderNoLoop, StereoBatchVoice, Voice, VoiceBase, VoiceCombineSIMD,
    },
};

use xsynth_soundfonts::LoopMode;

use crate::soundfont::{Interpolator, LoopParams, SampleVoiceSpawnerParams, VoiceSpawner};

pub struct StereoSampledVoiceSpawner<S: 'static + Simd + Send + Sync> {
    speed_mult: f32,
    filter: Option<BiQuadFilter>,
    loop_params: LoopParams,
    amp: f32,
    pan: f32,
    volume_envelope_params: Arc<EnvelopeParameters>,
    samples: Arc<[Arc<[f32]>]>,
    interpolator: Interpolator,
    exclusive_class: Option<u8>,
    vel: u8,
    stream_params: AudioStreamParams,
    _s: PhantomData<S>,
}

impl<S: Simd + Send + Sync> StereoSampledVoiceSpawner<S> {
    pub fn new(
        params: &SampleVoiceSpawnerParams,
        vel: u8,
        stream_params: AudioStreamParams,
    ) -> Self {
        let amp = params.volume;

        let filter = params.cutoff.map(|cutoff| {
            BiQuadFilter::new(
                params.filter_type,
                cutoff,
                stream_params.sample_rate as f32,
                Some(params.resonance),
            )
        });

        Self {
            speed_mult: params.speed_mult,
            filter,
            loop_params: params.loop_params.clone(),
            amp,
            pan: params.pan,
            volume_envelope_params: params.envelope.clone(),
            samples: params.sample.clone(),
            interpolator: params.interpolator,
            exclusive_class: params.exclusive_class,
            vel,
            stream_params,
            _s: PhantomData,
        }
    }

    fn begin_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        // 性能探针：按 1/N 采样 voice（仅 voice_probe feature 且运行期开启时命中）。
        // 包装在 `convert_to_voice` 中完成，这里只把采样标记透传下去。
        let probe = crate::voice_probe::should_probe();
        if probe {
            crate::voice_probe::record_voice_kind(self.filter.is_some());
        }
        self.begin_voice_impl(control, !probe && crate::voice::batching_supported(), probe)
    }

    /// 构造 voice。`batch = true` 时优先构造 B1 批处理 voice（lane 即权威状态），
    /// 否则构造原链式生成器。差分测试直接调用本函数对比两条路径（逐位一致）。
    fn begin_voice_impl(
        &self,
        control: &VoiceControlData,
        batch: bool,
        probe: bool,
    ) -> Box<dyn Voice> {
        // B1 批处理：可批时 lane 直接作为权威状态（跳过整条链式生成器）。
        // 被采样的 voice 不批（否则 Tracy 的阶段归因失真）。
        if batch {
            if let Some(voice) = self.make_batch_voice(control) {
                return voice;
            }
        }

        // Currently there's only the f32 buffer samples, more could be added in the future.
        #[allow(clippy::redundant_closure)]
        self.make_sample_reader(control, |s| BufferSamplers::new_f32(s), probe)
    }

    /// B1：构造可批 voice。`LUMINO_BATCH=0`、宿主闸门关闭或 SIMD 宽度不足时
    /// 返回 `None`（走原链路）。
    fn make_batch_voice(&self, control: &VoiceControlData) -> Option<Box<dyn Voice>> {
        if self.samples.len() < 2 {
            return None;
        }
        let (gain_l, gain_r) = self.stereo_gains();
        let init = BatchLaneInit {
            speed_mult: self.speed_mult,
            gain_l,
            gain_r,
            samples_l: self.samples[0].clone(),
            samples_r: self.samples[1].clone(),
            loop_mode: self.loop_params.mode,
            loop_offset: self.loop_params.offset as usize,
            loop_start: self.loop_params.start as usize,
            loop_end: self.loop_params.end as usize,
            loop_stop: self.loop_params.stop.map(|stop| stop as usize),
            interpolator: self.interpolator,
            filter: self.filter.clone(),
            envelope: *self.volume_envelope_params,
            sample_rate: self.stream_params.sample_rate as f32,
            // 包络「8 帧组」大小必须与真实链路的运行时 SIMD 宽度一致（见 batch.rs）。
            group_len: crate::voice::batch_chunk_width() as u8,
            velocity: self.vel,
            exclusive_class: self.exclusive_class,
        };
        Some(Box::new(StereoBatchVoice::new(&init, control)))
    }

    fn make_sample_reader<BS: 'static + BufferSampler>(
        &self,
        control: &VoiceControlData,
        make_bs: impl Fn(Arc<[f32]>) -> BS,
        probe: bool,
    ) -> Box<dyn Voice> {
        match self.loop_params.mode {
            LoopMode::LoopContinuous => self.make_sample_grabber(
                control,
                move |s| SampleReaderLoop::new(make_bs(s), self.loop_params.clone()),
                probe,
            ),
            LoopMode::LoopSustain => self.make_sample_grabber(
                control,
                move |s| SampleReaderLoopSustain::new(make_bs(s), self.loop_params.clone()),
                probe,
            ),
            LoopMode::NoLoop | LoopMode::OneShot => self.make_sample_grabber(
                control,
                move |s| SampleReaderNoLoop::new(make_bs(s), self.loop_params.clone()),
                probe,
            ),
        }
    }

    fn make_sample_grabber<SR: 'static + SampleReader>(
        &self,
        control: &VoiceControlData,
        make_bs: impl Fn(Arc<[f32]>) -> SR,
        probe: bool,
    ) -> Box<dyn Voice> {
        match self.interpolator {
            Interpolator::Nearest => self.generate_sampler(
                control,
                |s| SIMDNearestSampleGrabber::new(make_bs(s)),
                probe,
            ),
            Interpolator::Linear => {
                self.generate_sampler(control, |s| SIMDLinearSampleGrabber::new(make_bs(s)), probe)
            }
        }
    }

    fn generate_sampler<SG: 'static + SIMDSampleGrabber<S>>(
        &self,
        control: &VoiceControlData,
        make_sampler: impl Fn(Arc<[f32]>) -> SG,
        probe: bool,
    ) -> Box<dyn Voice> {
        let left = make_sampler(self.samples[0].clone());
        let right = make_sampler(self.samples[1].clone());

        let pitch_fac = self.create_pitch_fac(control);

        let sampler = SIMDStereoVoiceSampler::new(left, right, pitch_fac, probe);
        self.apply_voice_params(sampler, control, probe)
    }

    /// 增益合成：力度 `amp` 与等功率声像增益在 spawn 期合并为一个 stereo 常量，
    /// 每 8 帧只做一次 SIMD 乘。
    ///
    /// 原实现是 `amp`（单声道常量乘）+ 声像增益（立体声乘）两层 `SIMDVoiceCombine`，
    /// 组合链上每层每 8 帧都要走一遍嵌套 next_sample；两者都是常量，相乘是等价的。
    fn stereo_gains(&self) -> (f32, f32) {
        let pan = self.pan * std::f32::consts::PI / 2.0;
        let left = (pan.cos() * 1.42).min(1.0) * self.amp;
        let right = (pan.sin() * 1.42).min(1.0) * self.amp;
        (left, right)
    }

    fn apply_gain<Gen>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, SIMDSampleStereo<S>>
    where
        Gen: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let (left, right) = self.stereo_gains();
        let gains = SIMDConstantStereo::<S>::new(left, right);
        VoiceCombineSIMD::mult(gains, gen)
    }

    fn create_pitch_fac(
        &self,
        control: &VoiceControlData,
    ) -> impl SIMDVoiceGenerator<S, SIMDSampleMono<S>> {
        // 单层「常量 × 控制值」生成器：等价语义，组合链上少一层 next_sample。
        SIMDConstantControl::<S>::new(self.speed_mult, control, |vc| vc.voice_pitch_multiplier)
    }

    fn apply_envelope<Gen, Sample>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
        probe: bool,
    ) -> impl SIMDVoiceGenerator<S, Sample>
    where
        Sample: SIMDSample<S>,
        SIMDSampleMono<S>: Mul<Sample, Output = Sample>,
        Gen: SIMDVoiceGenerator<S, Sample>,
    {
        let modified_params = SIMDVoiceEnvelope::<S>::get_modified_envelope(
            *self.volume_envelope_params.clone(),
            control.envelope,
            self.stream_params.sample_rate as f32,
        );

        let allow_release = self.loop_params.mode != LoopMode::OneShot;

        let volume_envelope = SIMDVoiceEnvelope::new(
            *self.volume_envelope_params.clone(),
            modified_params,
            allow_release,
            self.stream_params.sample_rate as f32,
            probe,
        );

        let amp = VoiceCombineSIMD::mult(volume_envelope, gen);
        amp
    }

    fn convert_to_voice<Gen>(&self, gen: Gen, probe: bool) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let flattened = SIMDStereoVoice::new(gen, probe);
        let base = VoiceBase::new(self.vel, self.exclusive_class(), flattened);

        if probe {
            Box::new(crate::voice_probe::ProbeVoice::new(Box::new(base)))
        } else {
            Box::new(base)
        }
    }

    fn apply_voice_params<Gen>(
        &self,
        gen: Gen,
        control: &VoiceControlData,
        probe: bool,
    ) -> Box<dyn Voice>
    where
        Gen: 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let gen = self.apply_gain(gen);
        let gen = self.apply_envelope(gen, control, probe);

        self.apply_cutoff_effect(gen, probe)
    }

    fn apply_cutoff_effect(
        &self,
        gen: impl 'static + SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
        probe: bool,
    ) -> Box<dyn Voice> {
        if let Some(filter) = &self.filter {
            let gen = SIMDStereoVoiceCutoff::new(gen, filter, probe);
            self.convert_to_voice(gen, probe)
        } else {
            self.convert_to_voice(gen, probe)
        }
    }
}

impl<S: 'static + Sync + Send + Simd> VoiceSpawner for StereoSampledVoiceSpawner<S> {
    fn spawn_voice(&self, control: &VoiceControlData) -> Box<dyn Voice> {
        self.begin_voice(control)
    }

    fn exclusive_class(&self) -> Option<u8> {
        self.exclusive_class
    }
}

#[cfg(test)]
mod tests {
    //! B1 差分验证：批处理 lane 与真实链式生成器逐位一致。
    //!
    //! 三条独立口径：
    //! 1. `BatchLane::render_scalar`（标量 lane）≡ 原链式生成器（逐位）；
    //! 2. 批内核（向量 biquad + 水平求和）≡ 标量 lane（逐位，用「单活跃 lane +
    //!    其余静音 lane」隔离混音求和顺序）；
    //! 3. 8 lane 全部活跃时，内核与逐 voice 顺序渲染的差异仅为混音求和顺序造成的
    //!    f32 舍入（1e-6 相对量级）。

    use std::marker::PhantomData;

    use simdeez::prelude::*;
    use simdeez::simd_runtime_generate;

    use super::*;
    use crate::soundfont::{EnvelopeCurveType, EnvelopeDescriptor, EnvelopeOptions};
    use crate::voice::{
        render_batch_chunk, render_batch_chunk_mode, BatchLane, ReleaseType, Voice,
    };

    const SR: u32 = 48_000;
    const FRAMES: usize = 480;

    fn sample_buffer(seed: f32) -> Arc<[f32]> {
        (0..8_192)
            .map(|i| {
                let t = i as f32 * 0.01 + seed;
                (t.sin() * 0.7 + (t * 2.13).sin() * 0.3) * 0.5
            })
            .collect::<Vec<_>>()
            .into()
    }

    /// 覆盖全部阶段（Delay/Attack/Hold/Decay/Sustain/Release/Finished）的包络。
    fn descriptor() -> EnvelopeDescriptor {
        EnvelopeDescriptor {
            start_percent: 0.0,
            delay: 0.004,
            attack: 0.01,
            hold: 0.006,
            decay: 0.03,
            sustain_percent: 0.35,
            release: 0.05,
        }
    }

    /// 长 Decay 包络：整个测量期间停留在 Decay（`LerpConcave`，标量路径最贵），
    /// 贴近真实钢琴长衰减声部的稳态负载。
    fn long_descriptor() -> EnvelopeDescriptor {
        EnvelopeDescriptor {
            start_percent: 0.0,
            delay: 0.0,
            attack: 0.001,
            hold: 0.0,
            decay: 1_000.0,
            sustain_percent: 0.5,
            release: 0.05,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn spawner<S: 'static + Simd + Send + Sync>(
        loop_mode: LoopMode,
        interpolator: Interpolator,
        filter: bool,
        options: EnvelopeOptions,
        amp: f32,
        vel: u8,
    ) -> StereoSampledVoiceSpawner<S> {
        let stream_params = AudioStreamParams::new(SR, crate::ChannelCount::Stereo);
        StereoSampledVoiceSpawner::<S> {
            speed_mult: 1.0,
            filter: filter.then(|| {
                BiQuadFilter::new(
                    crate::effects::FilterType::LowPass,
                    9_000.0,
                    SR as f32,
                    Some(0.7),
                )
            }),
            loop_params: LoopParams {
                mode: loop_mode,
                offset: 0,
                start: 100,
                end: 3_000,
                stop: None,
            },
            amp,
            pan: 0.3,
            volume_envelope_params: Arc::new(descriptor().to_envelope_params(SR, options)),
            samples: Arc::new([sample_buffer(0.0), sample_buffer(1.7)]),
            interpolator,
            exclusive_class: None,
            vel,
            stream_params,
            _s: PhantomData,
        }
    }

    fn all_cases() -> Vec<(LoopMode, Interpolator, bool)> {
        let mut cases = Vec::new();
        for mode in [
            LoopMode::LoopSustain,
            LoopMode::LoopContinuous,
            LoopMode::NoLoop,
            LoopMode::OneShot,
        ] {
            for interpolator in [Interpolator::Nearest, Interpolator::Linear] {
                for filter in [true, false] {
                    cases.push((mode, interpolator, filter));
                }
            }
        }
        cases
    }

    fn envelope_option_sets() -> [EnvelopeOptions; 2] {
        [
            EnvelopeOptions::default(),
            EnvelopeOptions {
                attack_curve: EnvelopeCurveType::Linear,
                decay_curve: EnvelopeCurveType::Exponential,
                release_curve: EnvelopeCurveType::Exponential,
            },
        ]
    }

    /// 口径 1：标量 lane ≡ 真实链式生成器（逐位），覆盖
    /// 4 loop mode × 2 interpolator × 2 filter × 2 曲线组合 × 14 块，
    /// 中途注入控制消息（pitch/attack/release）与 release/kill。
    ///
    /// 关键：包络用**长 Decay**（`long_descriptor`），保证采样位置跨越 loop 边界
    /// （`loop_end` 之后）时包络仍非零——否则 `env == 0` 会掩盖采样位置错误
    /// （历史上 NoLoop 不回绕语义的正确性正是被这一点掩盖过）。
    #[test]
    fn batch_lane_matches_real_chain_bitwise() {
        simd_runtime_generate!(
            fn run() {
                for (mode, interpolator, filter) in all_cases() {
                    for options in envelope_option_sets() {
                        for kill in [false, true] {
                            let mut s = spawner::<S>(mode, interpolator, filter, options, 0.4, 100);
                            s.volume_envelope_params =
                                Arc::new(long_descriptor().to_envelope_params(SR, options));
                            let control = VoiceControlData::new_defaults();
                            let mut real = s.begin_voice_impl(&control, false, false);
                            let mut batch = s.begin_voice_impl(&control, true, false);
                            assert!(
                                batch.batch_lane().is_some(),
                                "批 voice 未构造成功: {mode:?} {interpolator:?} filter={filter}"
                            );

                            let mut ctl = VoiceControlData::new_defaults();
                            ctl.voice_pitch_multiplier = 1.25;
                            ctl.envelope.attack = Some(90);
                            ctl.envelope.release = Some(30);

                            let mut real_buf = vec![0.0f32; FRAMES * 2];
                            let mut batch_buf = vec![0.0f32; FRAMES * 2];
                            for block in 0..14 {
                                if block == 2 {
                                    real.process_controls(&ctl);
                                    batch.process_controls(&ctl);
                                }
                                if block == 9 {
                                    real.signal_release(ReleaseType::Standard);
                                    batch.signal_release(ReleaseType::Standard);
                                }
                                if block == 12 && kill {
                                    real.signal_release(ReleaseType::Kill);
                                    batch.signal_release(ReleaseType::Kill);
                                }
                                real_buf.fill(0.0);
                                batch_buf.fill(0.0);
                                real.render_to(&mut real_buf);
                                batch.render_to(&mut batch_buf);

                                assert_eq!(
                                    real_buf, batch_buf,
                                    "第 {block} 块输出不一致: {mode:?} {interpolator:?} \
                                     filter={filter} kill={kill}"
                                );
                                assert_eq!(
                                    real.ended(),
                                    batch.ended(),
                                    "第 {block} 块 ended() 不一致: {mode:?} {interpolator:?} \
                                     filter={filter} kill={kill}"
                                );
                            }

                            // 结束后仍应保持同步：OneShot 时 `allow_release == false` 且
                            // 缓冲足够长（未越界结束），包络不会进入 Release（两路径一致地
                            // 不结束）；其余组合应已 Finished。
                            if kill || mode != LoopMode::OneShot {
                                assert!(real.ended() && batch.ended());
                            }
                        }
                    }
                }
            }
        );

        run();
    }

    /// 口径 2：批内核 ≡ 标量 lane（逐位）。
    ///
    /// 活跃 lane 放在中间下标（验证任意 lane 位置），其余 lane 全部静音（`amp = 0`
    /// → 增益为 0 → 经 biquad 后为 +0.0），因此水平求和 = 活跃 lane 的值。
    ///
    /// 覆盖全部 (loop mode × interpolator) 组合，且包络用**长 Decay**：采样位置会
    /// 越过 `loop_end`（3000）后才比较，专门锁死 NoLoop「不回绕」与 Loop「回绕」的
    /// 语义差异（历史盲区：静音 lane 的样本被 0 增益掩盖，导致该差异逃脱测试）。
    #[test]
    fn batch_kernel_matches_scalar_lane_bitwise() {
        simd_runtime_generate!(
            fn run() {
                let width = S::Vf32::WIDTH;
                let active_idx = (width / 2).min(width - 1);

                for (mode, interpolator) in [
                    (LoopMode::LoopSustain, Interpolator::Nearest),
                    (LoopMode::LoopSustain, Interpolator::Linear),
                    (LoopMode::LoopContinuous, Interpolator::Nearest),
                    (LoopMode::LoopContinuous, Interpolator::Linear),
                    (LoopMode::NoLoop, Interpolator::Nearest),
                    (LoopMode::NoLoop, Interpolator::Linear),
                    (LoopMode::OneShot, Interpolator::Nearest),
                    (LoopMode::OneShot, Interpolator::Linear),
                ] {
                    let options = EnvelopeOptions::default();
                    let mut active_spawner =
                        spawner::<S>(mode, interpolator, true, options, 0.4, 100);
                    active_spawner.volume_envelope_params =
                        Arc::new(long_descriptor().to_envelope_params(SR, options));
                    let mut silent_spawner =
                        spawner::<S>(mode, interpolator, true, options, 0.0, 100);
                    silent_spawner.volume_envelope_params =
                        Arc::new(long_descriptor().to_envelope_params(SR, options));

                    let control = VoiceControlData::new_defaults();
                    let mut voices: Vec<Box<dyn Voice>> = (0..width)
                        .map(|k| {
                            if k == active_idx {
                                active_spawner.begin_voice_impl(&control, true, false)
                            } else {
                                silent_spawner.begin_voice_impl(&control, true, false)
                            }
                        })
                        .collect();
                    // 参考 voice：同一参数、独立状态、只走标量 lane 渲染（`read_side` 语义）。
                    let mut reference = active_spawner.begin_voice_impl(&control, true, false);

                    let mut kernel_buf = vec![0.0f32; FRAMES * 2];
                    let mut scalar_buf = vec![0.0f32; FRAMES * 2];
                    for block in 0..16 {
                        kernel_buf.fill(0.0);
                        scalar_buf.fill(0.0);

                        let mut lanes: Vec<&mut BatchLane> =
                            voices.iter_mut().filter_map(|v| v.batch_lane()).collect();
                        assert_eq!(lanes.len(), width);
                        assert!(render_batch_chunk(&mut lanes, &mut kernel_buf, FRAMES));
                        drop(lanes);

                        reference
                            .batch_lane()
                            .expect("参考 voice 应为批 voice")
                            .render_scalar(&mut scalar_buf);

                        assert_eq!(
                            kernel_buf, scalar_buf,
                            "第 {block} 块内核与标量 lane 不一致: {mode:?} {interpolator:?}"
                        );
                    }
                }
            }
        );

        run();
    }

    /// 口径 2b：统一 chunk 专用内核 ≡ 通用内核（逐位）。
    ///
    /// 通用内核已由口径 1/2 验证与标量参考逐位一致；本测试专门覆盖 B1.3 引入的
    /// 无逐 lane 分支专用内核（模式/插值/释放状态 const 泛型化）各组合的等价性。
    #[test]
    fn batch_uniform_kernel_matches_generic_kernel_bitwise() {
        simd_runtime_generate!(
            fn run() {
                let width = S::Vf32::WIDTH;
                let options = EnvelopeOptions::default();
                for (mode, interpolator) in [
                    (LoopMode::LoopSustain, Interpolator::Nearest),
                    (LoopMode::LoopSustain, Interpolator::Linear),
                    (LoopMode::LoopContinuous, Interpolator::Nearest),
                    (LoopMode::LoopContinuous, Interpolator::Linear),
                    (LoopMode::NoLoop, Interpolator::Linear),
                    (LoopMode::OneShot, Interpolator::Nearest),
                ] {
                    let s = spawner::<S>(mode, interpolator, true, options, 0.4, 100);
                    let control = VoiceControlData::new_defaults();
                    let mut uniform_voices: Vec<Box<dyn Voice>> = (0..width)
                        .map(|_| s.begin_voice_impl(&control, true, false))
                        .collect();
                    let mut generic_voices: Vec<Box<dyn Voice>> = (0..width)
                        .map(|_| s.begin_voice_impl(&control, true, false))
                        .collect();

                    let mut uniform_buf = vec![0.0f32; FRAMES * 2];
                    let mut generic_buf = vec![0.0f32; FRAMES * 2];
                    for block in 0..14 {
                        uniform_buf.fill(0.0);
                        generic_buf.fill(0.0);
                        // 第 6 块统一释放（覆盖 LoopSustain 的释放后分支）。
                        if block == 6 {
                            for voice in uniform_voices.iter_mut() {
                                voice.signal_release(ReleaseType::Standard);
                            }
                            for voice in generic_voices.iter_mut() {
                                voice.signal_release(ReleaseType::Standard);
                            }
                        }
                        let mut lanes: Vec<&mut BatchLane> = uniform_voices
                            .iter_mut()
                            .filter_map(|v| v.batch_lane())
                            .collect();
                        assert!(render_batch_chunk(&mut lanes, &mut uniform_buf, FRAMES));
                        drop(lanes);

                        let mut lanes: Vec<&mut BatchLane> = generic_voices
                            .iter_mut()
                            .filter_map(|v| v.batch_lane())
                            .collect();
                        assert!(render_batch_chunk_mode(
                            &mut lanes,
                            &mut generic_buf,
                            FRAMES,
                            3
                        ));
                        drop(lanes);

                        assert_eq!(
                            uniform_buf, generic_buf,
                            "第 {block} 块统一内核与通用内核不一致: {mode:?} {interpolator:?}"
                        );
                    }
                }
            }
        );

        run();
    }

    /// 口径 3：8 lane 全活跃时，内核与逐 voice 顺序渲染的差异仅为混音求和顺序
    /// （树形 vs 顺序）带来的 f32 舍入，必须在 1e-6 相对量级内，且 RMS 一致。
    #[test]
    fn batch_kernel_mixing_order_within_tolerance() {
        simd_runtime_generate!(
            fn run() {
                let width = S::Vf32::WIDTH;
                let options = EnvelopeOptions::default();
                let s = spawner::<S>(
                    LoopMode::LoopSustain,
                    Interpolator::Linear,
                    true,
                    options,
                    0.4,
                    100,
                );
                let control = VoiceControlData::new_defaults();

                let mut kernel_voices: Vec<Box<dyn Voice>> = (0..width)
                    .map(|_| s.begin_voice_impl(&control, true, false))
                    .collect();
                let mut scalar_voices: Vec<Box<dyn Voice>> = (0..width)
                    .map(|_| s.begin_voice_impl(&control, true, false))
                    .collect();

                let mut kernel_buf = vec![0.0f32; FRAMES * 2];
                let mut scalar_buf = vec![0.0f32; FRAMES * 2];
                for block in 0..6 {
                    kernel_buf.fill(0.0);
                    scalar_buf.fill(0.0);

                    let mut lanes: Vec<&mut BatchLane> = kernel_voices
                        .iter_mut()
                        .filter_map(|v| v.batch_lane())
                        .collect();
                    assert_eq!(lanes.len(), width);
                    assert!(render_batch_chunk(&mut lanes, &mut kernel_buf, FRAMES));
                    drop(lanes);

                    for voice in scalar_voices.iter_mut() {
                        voice.render_to(&mut scalar_buf);
                    }

                    let mut max_diff = 0.0f32;
                    let mut peak = 0.0f32;
                    let mut energy_kernel = 0.0f64;
                    let mut energy_scalar = 0.0f64;
                    for (k, s) in kernel_buf.iter().zip(scalar_buf.iter()) {
                        max_diff = max_diff.max((k - s).abs());
                        peak = peak.max(s.abs());
                        energy_kernel += (*k as f64) * (*k as f64);
                        energy_scalar += (*s as f64) * (*s as f64);
                    }
                    let tolerance = 1e-6 + 1e-6 * peak;
                    assert!(
                        max_diff <= tolerance,
                        "第 {block} 块混音顺序差异超容差: max_diff={max_diff} peak={peak}"
                    );
                    let rms_kernel = (energy_kernel / kernel_buf.len() as f64).sqrt();
                    let rms_scalar = (energy_scalar / scalar_buf.len() as f64).sqrt();
                    assert!(
                        (rms_kernel - rms_scalar).abs() <= 1e-9 + 1e-6 * rms_scalar,
                        "第 {block} 块 RMS 偏离: kernel={rms_kernel} scalar={rms_scalar}"
                    );
                }
            }
        );

        run();
    }

    /// B1 内核 vs 真实链路（同参数、同长度、min-of-N 交错）——真实 lane 实现的
    /// 单 voice 成本，用于判断批处理收益是否被标量部分（采样/包络）吃掉。
    ///
    /// 手动运行：
    /// `cargo test --release -p xsynth-core --lib batch_kernel_speed -- --ignored --nocapture`
    #[test]
    #[ignore = "性能基准，手动运行"]
    fn batch_kernel_speed_vs_real_chain() {
        use std::time::Instant;

        simd_runtime_generate!(
            fn run() {
                let width = S::Vf32::WIDTH;
                if width < 8 {
                    println!("[B1] 跳过：运行时 SIMD 宽度 {width} < 8");
                    return;
                }
                const ROUNDS: usize = 5;
                const ITERS: usize = 200;

                for (name, mode, interpolator, filter) in [
                    (
                        "loop_sustain+filter",
                        LoopMode::LoopSustain,
                        Interpolator::Nearest,
                        true,
                    ),
                    (
                        "loop_continuous+filter",
                        LoopMode::LoopContinuous,
                        Interpolator::Nearest,
                        true,
                    ),
                    (
                        "loop_sustain+nofilter",
                        LoopMode::LoopSustain,
                        Interpolator::Nearest,
                        false,
                    ),
                    (
                        "loop_sustain+filter+linear",
                        LoopMode::LoopSustain,
                        Interpolator::Linear,
                        true,
                    ),
                ] {
                    // 长 Decay：测量期间停在 LerpConcave 阶段（标量路径最贵）。
                    let s = spawner::<S>(
                        mode,
                        interpolator,
                        filter,
                        EnvelopeOptions::default(),
                        0.4,
                        100,
                    );
                    // 用长衰减包络替换（spawner 字段为私有，测试内可直接重建）。
                    let mut s = s;
                    s.volume_envelope_params = Arc::new(
                        long_descriptor().to_envelope_params(SR, EnvelopeOptions::default()),
                    );
                    let control = VoiceControlData::new_defaults();

                    let mut chain: Vec<Box<dyn Voice>> = (0..width)
                        .map(|_| s.begin_voice_impl(&control, false, false))
                        .collect();
                    let mut kernel: Vec<Box<dyn Voice>> = (0..width)
                        .map(|_| s.begin_voice_impl(&control, true, false))
                        .collect();
                    if kernel[0].batch_lane().is_none() {
                        println!("[B1] 跳过 {name}：批 voice 未构造成功");
                        continue;
                    }

                    let mut chain_buf = vec![0.0f32; FRAMES * 2];
                    let mut kernel_buf = vec![0.0f32; FRAMES * 2];
                    let mut best_chain = f64::MAX;
                    let mut best_modes = [f64::MAX; 6];
                    macro_rules! bench_mode {
                        ($mode:literal) => {{
                            let t = Instant::now();
                            for _ in 0..ITERS {
                                kernel_buf.fill(0.0);
                                let mut lanes: Vec<&mut BatchLane> =
                                    kernel.iter_mut().filter_map(|v| v.batch_lane()).collect();
                                assert!(render_batch_chunk_mode(
                                    &mut lanes,
                                    &mut kernel_buf,
                                    FRAMES,
                                    $mode
                                ));
                            }
                            best_modes[$mode] =
                                best_modes[$mode].min(t.elapsed().as_nanos() as f64 / ITERS as f64);
                        }};
                    }
                    for _ in 0..ROUNDS {
                        let t = Instant::now();
                        for _ in 0..ITERS {
                            chain_buf.fill(0.0);
                            for voice in chain.iter_mut() {
                                voice.render_to(&mut chain_buf);
                            }
                        }
                        best_chain = best_chain.min(t.elapsed().as_nanos() as f64 / ITERS as f64);

                        bench_mode!(0);
                        bench_mode!(1);
                        bench_mode!(2);
                        bench_mode!(3);
                        bench_mode!(4);
                        // 生产入口（统一 chunk 走无逐 lane 分支的专用内核）
                        {
                            let t = Instant::now();
                            for _ in 0..ITERS {
                                kernel_buf.fill(0.0);
                                let mut lanes: Vec<&mut BatchLane> =
                                    kernel.iter_mut().filter_map(|v| v.batch_lane()).collect();
                                assert!(render_batch_chunk(&mut lanes, &mut kernel_buf, FRAMES));
                            }
                            best_modes[5] =
                                best_modes[5].min(t.elapsed().as_nanos() as f64 / ITERS as f64);
                        }
                    }

                    println!(
                        "[B1] {name}: chain={:.0} mode0(sample)={:.0} mode1(+env)={:.0} mode2(+gain)={:.0} mode3(generic)={:.0} mode4(flat)={:.0} prod(uniform)={:.0} ns/voice-block; speedup={:.2}x",
                        best_chain / width as f64,
                        best_modes[0] / width as f64,
                        best_modes[1] / width as f64,
                        best_modes[2] / width as f64,
                        best_modes[3] / width as f64,
                        best_modes[4] / width as f64,
                        best_modes[5] / width as f64,
                        best_chain / best_modes[5]
                    );
                }
            }
        );

        run();
    }
}
