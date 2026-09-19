#![allow(dead_code)]
#![allow(non_camel_case_types)] // For the SIMD library

mod envelopes;
pub(crate) use envelopes::*;

mod simd;
pub(crate) use simd::*;

mod simdvoice;
pub(crate) use simdvoice::*;

pub mod batch;
pub use batch::BatchLane;
pub(crate) use batch::{
    BatchLaneInit, StereoBatchVoice, batch_chunk_width, batching_supported, render_batch_chunk,
};

#[cfg(test)]
mod batch_proto;

mod base;
pub(crate) use base::*;

mod squarewave;
#[allow(unused_imports)]
pub(crate) use squarewave::*;

mod channels;
#[allow(unused_imports)]
pub(crate) use channels::*;

mod constant;
pub(crate) use constant::*;

mod sampler;
pub(crate) use sampler::*;

mod control;
pub(crate) use control::*;

mod cutoff;
pub(crate) use cutoff::*;

/// Options to modify the envelope of a voice.
#[derive(Copy, Clone)]
pub struct EnvelopeControlData {
    /// Controls the attack. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub attack: Option<u8>,

    /// Controls the release. Can take values from 0 to 128
    /// according to the MIDI CC spec.
    pub release: Option<u8>,
}

/// How a voice should be released.
#[derive(Copy, Clone, PartialEq)]
pub enum ReleaseType {
    /// Standard release. Uses the voice's envelope.
    Standard,

    /// Kills the voice with a fadeout of 1ms.
    Kill,
}

/// Options to control the parameters of a voice.
#[derive(Copy, Clone)]
pub struct VoiceControlData {
    /// Pitch multiplier
    pub voice_pitch_multiplier: f32,

    /// Envelope control
    pub envelope: EnvelopeControlData,
}

impl VoiceControlData {
    pub fn new_defaults() -> Self {
        VoiceControlData {
            voice_pitch_multiplier: 1.0,
            envelope: EnvelopeControlData {
                attack: None,
                release: None,
            },
        }
    }
}

pub trait VoiceGeneratorBase: Sync + Send {
    fn ended(&self) -> bool;
    fn signal_release(&mut self, rel_type: ReleaseType);
    fn process_controls(&mut self, control: &VoiceControlData);
}

pub trait VoiceSampleGenerator: VoiceGeneratorBase {
    fn render_to(&mut self, buffer: &mut [f32]);
}

pub trait Voice: VoiceSampleGenerator + Send + Sync {
    fn is_releasing(&self) -> bool;
    fn is_killed(&self) -> bool;

    fn velocity(&self) -> u8;
    fn exclusive_class(&self) -> Option<u8>;

    /// B1 批处理：返回可批 lane（仅批处理 voice 实现返回 `Some`，其余走默认 `None`）。
    ///
    /// 由 `VoiceChannel` 的批渲染路径调用；返回 `Some` 表示该 voice 的权威状态在
    /// lane 内，渲染时由批内核驱动。
    #[doc(hidden)]
    fn batch_lane(&mut self) -> Option<&mut BatchLane> {
        None
    }
}
