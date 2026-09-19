use std::marker::PhantomData;

use simdeez::prelude::*;

use crate::voice::{ReleaseType, VoiceControlData};

use super::{
    SIMDSample, SIMDSampleMono, SIMDSampleStereo, SIMDVoiceGenerator, VoiceGeneratorBase,
    VoiceSampleGenerator,
};

pub struct SIMDStereoVoice<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>> {
    generator: T,
    remainder: SIMDSampleStereo<S>,
    remainder_pos: usize,
    /// 性能探针采样标记（仅 `voice_probe` feature 且命中采样时计时链式生成器）。
    #[cfg_attr(not(feature = "voice_probe"), allow(dead_code))]
    probe: bool,
    _s: PhantomData<S>,
}

impl<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>> SIMDStereoVoice<S, T> {
    pub fn new(generator: T, probe: bool) -> SIMDStereoVoice<S, T> {
        SIMDStereoVoice {
            generator,
            remainder: SIMDSampleStereo::<S>::zero(),
            remainder_pos: S::Vf32::WIDTH,
            probe,
            _s: PhantomData,
        }
    }
}

impl<S, T> VoiceGeneratorBase for SIMDStereoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.generator.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.generator.signal_release(rel_type)
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.generator.process_controls(control)
    }
}

impl<S, T> VoiceSampleGenerator for SIMDStereoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleStereo<S>>,
{
    fn render_to(&mut self, buffer: &mut [f32]) {
        simd_invoke!(S, {
            // 性能探针：累计链式生成器（envelope/pan/amp/sampler）耗时；
            // 与整 voice 耗时（ProbeVoice）之差即混音循环 + 每帧写入成本。
            #[cfg(feature = "voice_probe")]
            let mut chain_ticks: u64 = 0;

            let width = S::Vf32::WIDTH;
            let len = buffer.len();
            let mut i = 0usize;

            // 1) 前导：消费当前 remainder 中未用完的帧（跨块残留 / 非整块对齐）。
            while i + 1 < len && self.remainder_pos < width {
                unsafe {
                    *buffer.get_unchecked_mut(i) +=
                        self.remainder.0.get_unchecked(self.remainder_pos);
                    *buffer.get_unchecked_mut(i + 1) +=
                        self.remainder.1.get_unchecked(self.remainder_pos);
                }
                self.remainder_pos += 1;
                i += 2;
            }

            // 2) 整块快路径：每次取一组 width 帧（remainder 必然已用尽）。
            //    向量只在取数时读一次，固定展开的 lane 访问不会逐帧从 self 重载；
            //    每帧的 `remainder_pos == width` 分支也一并消除。
            while i + 2 * width <= len {
                #[cfg(feature = "voice_probe")]
                let chain_t0 = if self.probe {
                    crate::voice_probe::tick()
                } else {
                    None
                };
                self.remainder = self.generator.next_sample();
                #[cfg(feature = "voice_probe")]
                if let Some(t0) = chain_t0 {
                    if let Some(t1) = crate::voice_probe::tick() {
                        chain_ticks += t1.saturating_sub(t0);
                    }
                }

                let rem_l = self.remainder.0;
                let rem_r = self.remainder.1;
                for k in 0..width {
                    unsafe {
                        *buffer.get_unchecked_mut(i + 2 * k) += rem_l.get_unchecked(k);
                        *buffer.get_unchecked_mut(i + 2 * k + 1) += rem_r.get_unchecked(k);
                    }
                }
                i += 2 * width;
                self.remainder_pos = width;
            }

            // 3) 尾部：不足一组时逐帧取数（生产路径 480 % width == 0 通常不走到）。
            while i + 1 < len {
                if self.remainder_pos == width {
                    #[cfg(feature = "voice_probe")]
                    let chain_t0 = if self.probe {
                        crate::voice_probe::tick()
                    } else {
                        None
                    };
                    self.remainder = self.generator.next_sample();
                    #[cfg(feature = "voice_probe")]
                    if let Some(t0) = chain_t0 {
                        if let Some(t1) = crate::voice_probe::tick() {
                            chain_ticks += t1.saturating_sub(t0);
                        }
                    }
                    self.remainder_pos = 0;
                }
                unsafe {
                    *buffer.get_unchecked_mut(i) +=
                        self.remainder.0.get_unchecked(self.remainder_pos);
                    *buffer.get_unchecked_mut(i + 1) +=
                        self.remainder.1.get_unchecked(self.remainder_pos);
                }
                self.remainder_pos += 1;
                i += 2;
            }

            #[cfg(feature = "voice_probe")]
            if self.probe {
                crate::voice_probe::record_chain_ticks(chain_ticks);
            }
        })
    }
}

pub struct SIMDMonoVoice<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>> {
    generator: T,
    remainder: SIMDSampleMono<S>,
    remainder_pos: usize,
    _s: PhantomData<S>,
}

impl<S: Simd, T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>> SIMDMonoVoice<S, T> {
    pub fn new(generator: T) -> SIMDMonoVoice<S, T> {
        SIMDMonoVoice {
            generator,
            remainder: SIMDSampleMono::<S>::zero(),
            remainder_pos: S::Vf32::WIDTH,
            _s: PhantomData,
        }
    }
}

impl<S, T> VoiceGeneratorBase for SIMDMonoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.generator.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.generator.signal_release(rel_type)
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.generator.process_controls(control)
    }
}

impl<S, T> VoiceSampleGenerator for SIMDMonoVoice<S, T>
where
    S: Simd,
    T: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
{
    fn render_to(&mut self, buffer: &mut [f32]) {
        simd_invoke!(S, {
            let mut i = 0;
            while i < buffer.len() {
                if self.remainder_pos == S::Vf32::WIDTH {
                    self.remainder = self.generator.next_sample();
                    self.remainder_pos = 0;
                }

                buffer[i] += self.remainder.0[self.remainder_pos];
                i += 1;

                self.remainder_pos += 1;
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use simdeez::prelude::*;
    use simdeez::simd_runtime_generate;

    use super::SIMDStereoVoice;
    use crate::voice::{
        ReleaseType, SIMDSampleStereo, SIMDVoiceGenerator, VoiceControlData, VoiceGeneratorBase,
        VoiceSampleGenerator,
    };

    /// 生成「左 = 连续帧序号、右 = 相反数」的立体声样本，序号跨 `next_sample`/`render_to` 连续。
    struct SeqGen<S: Simd> {
        call: u32,
        _s: PhantomData<S>,
    }

    impl<S: Simd> SeqGen<S> {
        fn new() -> Self {
            SeqGen {
                call: 0,
                _s: PhantomData,
            }
        }
    }

    impl<S: Simd> VoiceGeneratorBase for SeqGen<S> {
        fn ended(&self) -> bool {
            false
        }

        fn signal_release(&mut self, _rel_type: ReleaseType) {}

        fn process_controls(&mut self, _control: &VoiceControlData) {}
    }

    impl<S: Simd> SIMDVoiceGenerator<S, SIMDSampleStereo<S>> for SeqGen<S> {
        fn next_sample(&mut self) -> SIMDSampleStereo<S> {
            simd_invoke!(S, {
                let mut left = S::Vf32::zeroes();
                let mut right = S::Vf32::zeroes();
                for i in 0..S::Vf32::WIDTH {
                    let v = (self.call as usize * S::Vf32::WIDTH + i) as f32;
                    left[i] = v;
                    right[i] = -v;
                }
                self.call += 1;
                SIMDSampleStereo(left, right)
            })
        }
    }

    /// 混音块化改写的行为验证：
    /// - 非整块对齐的缓冲区（尾部走慢路径）；
    /// - 跨两次 `render_to` 的 remainder 连续性（第二次的前导走慢路径）；
    /// - 每个输出帧的值必须与生成器的全局样本序号一一对应。
    #[test]
    fn stereo_mixer_matches_sample_sequence_across_blocks_and_calls() {
        simd_runtime_generate!(
            fn run() {
                let width = S::Vf32::WIDTH;
                let mut voice = SIMDStereoVoice::new(SeqGen::<S>::new(), false);

                // 非整块对齐：3 个整块 + 3 帧尾部
                let n = width * 3 + 3;
                let mut first = vec![0.0f32; n * 2];
                voice.render_to(&mut first);
                for frame in 0..n {
                    let expect = frame as f32;
                    assert_eq!(first[frame * 2], expect, "首块 L 帧 {frame}");
                    assert_eq!(first[frame * 2 + 1], -expect, "首块 R 帧 {frame}");
                }

                // 跨块连续：第二块从前一块的 remainder 中间接续
                let mut second = vec![0.0f32; width * 2];
                voice.render_to(&mut second);
                for frame in 0..width {
                    let expect = (n + frame) as f32;
                    assert_eq!(second[frame * 2], expect, "次块 L 帧 {frame}");
                    assert_eq!(second[frame * 2 + 1], -expect, "次块 R 帧 {frame}");
                }
            }
        );

        run();
    }
}
