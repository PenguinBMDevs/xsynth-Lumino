use std::{marker::PhantomData, sync::Arc};

use simdeez::prelude::*;

use crate::soundfont::LoopParams;
use crate::voice::{ReleaseType, VoiceControlData};

use super::{SIMDSampleMono, SIMDSampleStereo, SIMDVoiceGenerator, VoiceGeneratorBase};

mod linear;
pub use linear::*;

mod nearest;
pub use nearest::*;

// I believe some terminology reference is relevant for this one.
//
// BufferSampler: Something that grabs a sample based on an index
//
// SampleReader: Something that grabs the sample value at an arbitrary index,
// and implements sample start/end/looping
//
// SIMDSampleGrabber: Something that takes a SIMD array of float64 locations and
// returns a SIMD array of f32 interpolated sample values

// Base traits

pub trait BufferSampler: Send + Sync {
    fn get(&self, pos: usize) -> f32;
    fn length(&self) -> usize;
}

pub trait SIMDSampleGrabber<S: Simd>: Send + Sync {
    /// Indexes: the rounded index of the sample
    ///
    /// Fractional: The fractional part of the index, i.e. the 0-1 range decimal
    fn get(&mut self, indexes: S::Vi32, fractional: S::Vf32) -> S::Vf32;

    fn is_past_end(&self, pos: f64) -> bool;

    fn signal_release(&mut self);
}

// F32 sampler

pub struct F32BufferSampler(Arc<[f32]>);

impl F32BufferSampler {
    /// B1 批处理：底层切片（见 `BufferSamplers::slice`）。
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl BufferSampler for F32BufferSampler {
    #[inline(always)]
    fn get(&self, pos: usize) -> f32 {
        match self.0.get(pos) {
            Some(v) => *v,
            None => 0.0,
        }
    }

    fn length(&self) -> usize {
        self.0.len()
    }
}

// Generalized enum sampler

pub enum BufferSamplers {
    F32(F32BufferSampler),
}

impl BufferSamplers {
    #[inline(always)]
    pub fn new_f32(sample: Arc<[f32]>) -> BufferSamplers {
        BufferSamplers::F32(F32BufferSampler(sample))
    }

    /// B1 批处理：取出底层样本缓冲切片（lane 在 chunk 开始时裁剪一次，
    /// 热循环内不再做枚举匹配 / Arc 间接）。
    #[inline(always)]
    pub(crate) fn slice(&self) -> &[f32] {
        match self {
            BufferSamplers::F32(sampler) => sampler.as_slice(),
        }
    }
}

impl BufferSampler for BufferSamplers {
    #[inline(always)]
    fn get(&self, pos: usize) -> f32 {
        match self {
            BufferSamplers::F32(sampler) => sampler.get(pos),
        }
    }

    fn length(&self) -> usize {
        match self {
            BufferSamplers::F32(sampler) => sampler.length(),
        }
    }
}

// Enum sampler reader

/// B0 原型用：绕过 `LoopParams`（pub(super)）直接构造读取器（测试构建）。
#[cfg(test)]
impl<Sampler: BufferSampler> SampleReaderLoopSustain<Sampler> {
    pub(crate) fn new_raw(
        buffer: Sampler,
        offset: usize,
        loop_start: usize,
        loop_end: usize,
    ) -> Self {
        let length = Some(buffer.length());
        Self {
            buffer,
            length,
            offset,
            loop_start,
            loop_end,
            last: 0,
            is_released: false,
        }
    }
}

pub trait SampleReader: Send + Sync {
    fn get(&mut self, pos: usize) -> f32;
    fn is_past_end(&self, pos: usize) -> bool;
    fn signal_release(&mut self);
}

pub struct SampleReaderNoLoop<Sampler: BufferSampler> {
    buffer: Sampler,
    length: Option<usize>,
    offset: usize,
}

impl<Sampler: BufferSampler> SampleReaderNoLoop<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        let stop = loop_params
            .stop
            .map(|stop| stop as usize)
            .unwrap_or_else(|| buffer.length());
        let length = Some(stop);
        Self {
            buffer,
            length,
            offset: loop_params.offset as usize,
        }
    }
}

impl<Sampler: BufferSampler> SampleReader for SampleReaderNoLoop<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        self.buffer.get(pos + self.offset)
    }

    fn is_past_end(&self, pos: usize) -> bool {
        if let Some(len) = self.length {
            pos >= len.saturating_sub(self.offset)
        } else {
            false
        }
    }

    fn signal_release(&mut self) {}
}

pub struct SampleReaderLoop<Sampler: BufferSampler> {
    buffer: Sampler,
    offset: usize,
    loop_start: usize,
    loop_end: usize,
}

impl<Sampler: BufferSampler> SampleReaderLoop<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        Self {
            buffer,
            offset: loop_params.offset as usize,
            loop_start: loop_params.start as usize,
            loop_end: loop_params.end as usize,
        }
    }
}

impl<Sampler: BufferSampler> SampleReader for SampleReaderLoop<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        let mut pos = pos + self.offset;
        let end = self.loop_end;
        let start = self.loop_start;

        if pos > end {
            // 循环回绕：常见情况只越界少量样本，用比较 + 减法避免整数除法；
            // 极端越界（跨多个循环周期）回退取模，语义与原实现一致。
            let span = end - start;
            let d = pos - end - 1;
            pos = start + if d >= span { d % span } else { d };
        }

        self.buffer.get(pos)
    }

    fn is_past_end(&self, _pos: usize) -> bool {
        false
    }

    fn signal_release(&mut self) {}
}

pub struct SampleReaderLoopSustain<Sampler: BufferSampler> {
    buffer: Sampler,
    length: Option<usize>,
    offset: usize,
    loop_start: usize,
    loop_end: usize,
    last: usize,
    is_released: bool,
}

impl<Sampler: BufferSampler> SampleReaderLoopSustain<Sampler> {
    pub fn new(buffer: Sampler, loop_params: LoopParams) -> Self {
        let stop = loop_params
            .stop
            .map(|stop| stop as usize)
            .unwrap_or_else(|| buffer.length());
        let length = Some(stop);
        Self {
            buffer,
            length,
            offset: loop_params.offset as usize,
            loop_start: loop_params.start as usize,
            loop_end: loop_params.end as usize,
            last: 0,
            is_released: false,
        }
    }
}

impl<Sampler: BufferSampler> SampleReader for SampleReaderLoopSustain<Sampler> {
    fn get(&mut self, pos: usize) -> f32 {
        let mut pos = pos + self.offset;
        let end = self.loop_end;
        let start = self.loop_start;

        if !self.is_released {
            self.last = pos;
            if pos > end {
                // 同 SampleReaderLoop：常见越界走比较 + 减法，极端越界回退取模。
                let span = end - start;
                let d = pos - end - 1;
                pos = start + if d >= span { d % span } else { d };
            }
        } else {
            pos = pos - self.last + self.loop_end;
        }

        self.buffer.get(pos)
    }

    /// 越界判据：**「距上次读取位置推进了多远」是否已超过声明长度**。
    ///
    /// - 未释放：`get` 每次都把 `last` 更新为当前（未回绕）位置，`pos - last` 恒为
    ///   0/1 → **永不越界**。Sustain 循环在按键期间可以无限播放，声部只由包络
    ///   （release → Finished）结束。**这一条是长音能播完的前提**。
    /// - 已释放：`last` 冻结在释放点，`pos - last` 随线性外推增长，越过 `len`
    ///   （`loop_params.stop` 或缓冲长度）即越界，声部结束。
    ///
    /// 必须写成 `pos - last - offset` 的左结合形式：`pos >= len - (last + offset)`
    /// 是**错误的代数改写**（符号翻转，`last` 未释放时无界增长会让右边饱和到 0 →
    /// 恒为真，所有 sustain 长音在播放到样本长度一半时被硬切）。
    /// 用 saturating 减法表达同样的数学含义，同时避免 `offset > 0` 时的 usize 下溢 panic。
    fn is_past_end(&self, pos: usize) -> bool {
        if let Some(len) = self.length {
            pos.saturating_sub(self.last).saturating_sub(self.offset) >= len
        } else {
            false
        }
    }

    fn signal_release(&mut self) {
        self.is_released = true;
    }
}

// Sample grabbers enum

pub enum SIMDSampleGrabbers<S: Simd, Reader: SampleReader> {
    Nearest(SIMDNearestSampleGrabber<S, Reader>),
    Linear(SIMDLinearSampleGrabber<S, Reader>),
}

impl<S: Simd, Reader: SampleReader> SIMDSampleGrabbers<S, Reader> {
    pub fn nearest(reader: Reader) -> Self {
        SIMDSampleGrabbers::Nearest(SIMDNearestSampleGrabber::new(reader))
    }

    pub fn linear(reader: Reader) -> Self {
        SIMDSampleGrabbers::Linear(SIMDLinearSampleGrabber::new(reader))
    }
}

impl<S: Simd, Reader: SampleReader> SIMDSampleGrabber<S> for SIMDSampleGrabbers<S, Reader> {
    #[inline(always)]
    fn get(&mut self, indexes: S::Vi32, fractional: S::Vf32) -> S::Vf32 {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.get(indexes, fractional),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.get(indexes, fractional),
        }
    }

    #[inline(always)]
    fn is_past_end(&self, pos: f64) -> bool {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.is_past_end(pos),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.is_past_end(pos),
        }
    }

    #[inline(always)]
    fn signal_release(&mut self) {
        match self {
            SIMDSampleGrabbers::Linear(grabber) => grabber.signal_release(),
            SIMDSampleGrabbers::Nearest(grabber) => grabber.signal_release(),
        }
    }
}

// Sampler generator

pub struct SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    grabber: Grabber,

    pitch_gen: Pitch,

    time: f64,

    _s: PhantomData<S>,
}

impl<S, Pitch, Grabber> SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    pub fn new(grabber: Grabber, pitch_gen: Pitch) -> Self {
        SIMDMonoVoiceSampler {
            grabber,
            pitch_gen,
            time: 0.0,
            _s: PhantomData,
        }
    }

    fn increment_time(&mut self, by: f64) -> f64 {
        let time = self.time;
        self.time += by;
        time
    }
}

impl<S, Pitch, Grabber> VoiceGeneratorBase for SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.grabber.is_past_end(self.time)
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.pitch_gen.signal_release(rel_type);
        self.grabber.signal_release();
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.pitch_gen.process_controls(control);
    }
}

impl<S, Pitch, Grabber> SIMDVoiceGenerator<S, SIMDSampleMono<S>>
    for SIMDMonoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleMono<S> {
        simd_invoke!(S, {
            let speed = self.pitch_gen.next_sample().0;
            let mut indexes = S::Vi32::zeroes();
            let mut fractionals = S::Vf32::zeroes();

            unsafe {
                for i in 0..S::Vf32::WIDTH {
                    let time = self.increment_time(speed.get_unchecked(i) as f64);
                    // `time % 1.0` 会为每个样本触发 fmod 库调用；改用「截断整数 + 差值」：
                    // 对 0 ≤ time < 2^31 与 fmod 结果逐位等价（两者都是精确运算）。
                    let index = time as i32;
                    *indexes.get_unchecked_mut(i) = index;
                    *fractionals.get_unchecked_mut(i) = (time - index as f64) as f32;
                }
            }

            let sample = self.grabber.get(indexes, fractionals);

            SIMDSampleMono(sample)
        })
    }
}

pub struct SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    grabber_left: Grabber,
    grabber_right: Grabber,

    pitch_gen: Pitch,

    time: f64,

    /// 性能探针采样标记（仅 `voice_probe` feature 且命中采样步长时为 true）。
    probe: bool,

    _s: PhantomData<S>,
}

impl<S, Pitch, Grabber> SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    pub fn new(
        grabber_left: Grabber,
        grabber_right: Grabber,
        pitch_gen: Pitch,
        probe: bool,
    ) -> Self {
        SIMDStereoVoiceSampler {
            grabber_left,
            grabber_right,
            pitch_gen,
            time: 0.0,
            probe,
            _s: PhantomData,
        }
    }
}

impl<S, Pitch, Grabber> VoiceGeneratorBase for SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn ended(&self) -> bool {
        self.grabber_left.is_past_end(self.time) || self.grabber_right.is_past_end(self.time)
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.pitch_gen.signal_release(rel_type);
        self.grabber_left.signal_release();
        self.grabber_right.signal_release();
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.pitch_gen.process_controls(control);
    }
}

impl<S, Pitch, Grabber> SIMDVoiceGenerator<S, SIMDSampleStereo<S>>
    for SIMDStereoVoiceSampler<S, Pitch, Grabber>
where
    S: Simd,
    Pitch: SIMDVoiceGenerator<S, SIMDSampleMono<S>>,
    Grabber: SIMDSampleGrabber<S>,
{
    #[inline(always)]
    fn next_sample(&mut self) -> SIMDSampleStereo<S> {
        simd_invoke!(S, {
            // 性能探针：仅被采样的 voice 走计时路径（pitch → 时间推进 → 采样抓取）。
            let mut marks = crate::voice_probe::sampler_begin(self.probe);

            let speed = self.pitch_gen.next_sample().0;
            crate::voice_probe::sampler_mark(&mut marks, 1);
            let mut indexes = S::Vi32::zeroes();
            let mut fractionals = S::Vf32::zeroes();

            // 时间推进在局部 f64 中累加，循环结束再写回 `self.time`：
            // 运算顺序与逐 lane `increment_time` 完全一致（逐位等价），
            // 但省去每 lane 一次结构体读写。
            let mut time = self.time;
            unsafe {
                for i in 0..S::Vf32::WIDTH {
                    let t = time;
                    time += speed.get_unchecked(i) as f64;
                    // `time % 1.0` 会为每个样本触发 fmod 库调用；改用「截断整数 + 差值」：
                    // 对 0 ≤ time < 2^31 与 fmod 结果逐位等价（两者都是精确运算）。
                    let index = t as i32;
                    *indexes.get_unchecked_mut(i) = index;
                    *fractionals.get_unchecked_mut(i) = (t - index as f64) as f32;
                }
            }
            self.time = time;

            crate::voice_probe::sampler_mark(&mut marks, 2);
            let left = self.grabber_left.get(indexes, fractionals);
            let right = self.grabber_right.get(indexes, fractionals);
            crate::voice_probe::sampler_end(&mut marks, S::Vf32::WIDTH);

            SIMDSampleStereo(left, right)
        })
    }
}

#[cfg(test)]
mod tests {
    use xsynth_soundfonts::LoopMode;

    use super::*;

    fn ramp_buffer(len: usize) -> BufferSamplers {
        let data: Arc<[f32]> = (0..len)
            .map(|i| i as f32 * 0.001)
            .collect::<Vec<_>>()
            .into();
        BufferSamplers::new_f32(data)
    }

    /// 回归（**长音被直接杀掉** 的根因）：未释放的 Sustain 循环**永不**判定越界。
    ///
    /// 缺陷来自 `2d8a15a` 的代数改写：`pos - last - offset >= len` 被改成
    /// `pos >= len - (last + offset)`——符号翻转。`last` 是「未回绕」的读取位置，
    /// 未释放时随 `pos` 一起增长，于是阈值 `len - last - offset` 一路下降：
    /// 大约播放到样本长度**一半**时 `is_past_end` 就变真 → `VoiceBuffer::remove_ended_voices`
    /// 直接把声部移除（不是 release，是硬切）→ 听感即「长音播不完、音符被杀」。
    ///
    /// 本测试锁死语义：
    /// - 未释放：任意播放位置都不越界（Sustain 循环可无限播放，声部只由包络结束）；
    /// - 释放后：`last` 冻结，位置线性外推越过样本末尾 → 必须越界（否则声部滞留）。
    #[test]
    fn sustain_loop_is_never_past_end_while_held() {
        const LEN: usize = 2_048;
        for offset in [0u32, 37] {
            let mut reader = SampleReaderLoopSustain::new(
                ramp_buffer(LEN),
                LoopParams {
                    mode: LoopMode::LoopSustain,
                    offset,
                    start: 256,
                    end: 1_024,
                    stop: None,
                },
            );

            // 播放 32 倍样本长度：覆盖「未回绕位置远超 len」与「回绕若干圈」两种状态。
            for pos in 0..LEN * 32 {
                let _ = reader.get(pos);
                assert!(
                    !reader.is_past_end(pos),
                    "未释放的 sustain 循环在 pos={pos}（offset={offset}）被判越界——长音会被硬切"
                );
            }

            reader.signal_release();
            let release_pos = LEN * 32;
            let ended_after = (release_pos..release_pos + LEN * 8)
                .find(|&pos| {
                    let _ = reader.get(pos);
                    reader.is_past_end(pos)
                })
                .unwrap_or_else(|| panic!("释放后应最终判越界（offset={offset}）"));
            // 释放后必须至少再走完「声明长度」（-1 帧取整）才允许越界；提前触发 = 尾巴被砍。
            assert!(
                ended_after >= release_pos + LEN - 1,
                "释放后越界触发过早（release 尾巴被砍）: ended_after={ended_after} \
                 release_pos={release_pos} offset={offset}"
            );
        }
    }

    /// 回归（防止同一处再被"简化"错）：NoLoop 的越界判据是「读取位置 + offset 越过声明长度」。
    #[test]
    fn no_loop_past_end_uses_position_plus_offset() {
        const LEN: usize = 1_024;
        for offset in [0u32, 37] {
            let buffer = ramp_buffer(LEN);
            let mut reader = SampleReaderNoLoop::new(
                buffer,
                LoopParams {
                    mode: LoopMode::NoLoop,
                    offset,
                    start: 0,
                    end: 0,
                    stop: None,
                },
            );
            for pos in 0..LEN * 2 {
                let _ = reader.get(pos);
                assert_eq!(
                    reader.is_past_end(pos),
                    pos + offset as usize >= LEN,
                    "NoLoop 越界判据必须等价于「pos + offset >= len」: pos={pos} offset={offset}"
                );
            }
        }
    }

    /// 优化前后行为一致性：`time % 1.0` 与「截断 + 差值」在非负时间上逐位等价。
    #[test]
    fn fraction_optimization_matches_fmod() {
        let mut t = 0.0f64;
        let mut step = 0.123_456_789f64;
        let mut checks = 0u32;
        while t < 200_000.0 {
            let index = t as i32;
            let fast = (t - index as f64) as f32;
            let slow = (t % 1.0) as f32;
            assert_eq!(fast, slow, "t={t}");
            t += step;
            step = 0.1 + (step * std::f64::consts::GOLDEN_RATIO) % 1.0;
            checks += 1;
        }
        assert!(checks > 100_000, "采样点太少: {checks}");
    }
}
