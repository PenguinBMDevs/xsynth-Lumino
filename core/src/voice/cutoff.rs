use std::marker::PhantomData;

use simdeez::prelude::*;

use crate::{
    effects::BiQuadFilter,
    voice::{ReleaseType, SIMDVoiceGenerator, VoiceControlData},
};

use super::{SIMDSampleMono, SIMDSampleStereo, VoiceGeneratorBase};

pub struct SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    v: V,
    cutoff: BiQuadFilter,
    _s: PhantomData<S>,
}

impl<S, V> SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    pub fn new(v: V, filter: &BiQuadFilter) -> Self {
        SIMDMonoVoiceCutoff {
            v,
            cutoff: filter.clone(),
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.v.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.v.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.v.process_controls(control);
    }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleMono<S>> for SIMDMonoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();
            next_sample.0 = self.cutoff.process_simd::<S>(next_sample.0);
            next_sample
        })
    }
}

pub struct SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    v: V,
    cutoff1: BiQuadFilter,
    cutoff2: BiQuadFilter,
    /// 性能探针采样标记（仅 `voice_probe` feature 且命中采样时计时 biquad）。
    #[cfg_attr(not(feature = "voice_probe"), allow(dead_code))]
    probe: bool,
    _s: PhantomData<S>,
}

impl<S, V> SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    pub fn new(v: V, filter: &BiQuadFilter, probe: bool) -> Self {
        SIMDStereoVoiceCutoff {
            v,
            cutoff1: filter.clone(),
            cutoff2: filter.clone(),
            probe,
            _s: PhantomData,
        }
    }
}

impl<S, V> VoiceGeneratorBase for SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.v.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.v.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.v.process_controls(control);
    }
}

impl<S, V> SIMDVoiceGenerator<S, SIMDSampleStereo<S>> for SIMDStereoVoiceCutoff<S, V>
where
    S: Simd,
    V: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        simd_invoke!(S, {
            let mut next_sample = self.v.next_sample();

            // 性能探针：计时两路 biquad 的处理耗时（不含内层生成器）。
            #[cfg(feature = "voice_probe")]
            let probe_start = if self.probe {
                crate::voice_probe::tick()
            } else {
                None
            };

            next_sample.0 = self.cutoff1.process_simd::<S>(next_sample.0);
            next_sample.1 = self.cutoff2.process_simd::<S>(next_sample.1);

            #[cfg(feature = "voice_probe")]
            if let (Some(t0), Some(t1)) = (probe_start, crate::voice_probe::tick()) {
                crate::voice_probe::record_cut_ticks(t1.saturating_sub(t0));
            }

            next_sample
        })
    }
}
