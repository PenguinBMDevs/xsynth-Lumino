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
        BufferSamplers, EnvelopeParameters, SIMDConstantControl, SIMDConstantStereo,
        SIMDLinearSampleGrabber, SIMDNearestSampleGrabber, SIMDStereoVoice, SIMDStereoVoiceSampler,
        SIMDVoiceEnvelope, SampleReader, SampleReaderLoop, SampleReaderLoopSustain,
        SampleReaderNoLoop, Voice, VoiceBase, VoiceCombineSIMD,
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

        // Currently there's only the f32 buffer samples, more could be added in the future.
        #[allow(clippy::redundant_closure)]
        self.make_sample_reader(control, |s| BufferSamplers::new_f32(s), probe)
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
            Interpolator::Linear => self.generate_sampler(
                control,
                |s| SIMDLinearSampleGrabber::new(make_bs(s)),
                probe,
            ),
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
    fn apply_gain<Gen>(&self, gen: Gen) -> impl SIMDVoiceGenerator<S, SIMDSampleStereo<S>>
    where
        Gen: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
    {
        let pan = self.pan * std::f32::consts::PI / 2.0;
        let left = (pan.cos() * 1.42).min(1.0) * self.amp;
        let right = (pan.sin() * 1.42).min(1.0) * self.amp;

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
