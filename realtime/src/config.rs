use std::ops::RangeInclusive;
pub use xsynth_core::{
    channel::ChannelInitOptions,
    channel_group::{SynthFormat, ThreadCount},
};

/// Options for initializing a new RealtimeSynth.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct XSynthRealtimeConfig {
    /// Channel initialization options (same for all channels).
    /// See the `ChannelInitOptions` documentation for more information.
    pub channel_init_options: ChannelInitOptions,

    /// The length of the buffer reader in ms.
    ///
    /// Also the event timing quantization: MIDI events are applied at render
    /// block boundaries, so smaller blocks give tighter note timing (at the
    /// cost of more per-block overhead).
    ///
    /// Default: `10.0`
    pub render_window_ms: f64,

    /// Target amount of rendered-but-unconsumed audio kept buffered, in ms.
    ///
    /// Acts as a cushion against render spikes and OS scheduling jitter.
    /// Larger values increase output latency but drastically reduce dropouts
    /// and the associated stutter. Should be >= `render_window_ms`.
    ///
    /// Default: `100.0`
    pub cushion_ms: f64,

    /// Maximum number of active voices summed across **all** channels
    /// (`None` = unlimited, `Some(0)` is treated as unlimited too).
    ///
    /// Enforced by the render pipe: when exceeded, the busiest channels are
    /// asked to steal their oldest/quietest/releasing voices, so newly played
    /// notes keep sounding. This is the cross-channel counterpart of
    /// `channel_init_options.max_voices` (which is per channel).
    ///
    /// Default: `None`
    pub global_max_voices: Option<usize>,

    /// Defines the format that the synthesizer will use. See the `SynthFormat`
    /// documentation for more information.
    ///
    /// Default: `SynthFormat::Midi`
    pub format: SynthFormat,

    /// Controls the multithreading used for rendering per-voice audio for all
    /// the voices stored in a key for a channel. See the `ThreadCount` documentation
    /// for the available options.
    ///
    /// Default: `ThreadCount::None`
    pub multithreading: ThreadCount,

    /// A range of velocities that will not be played.
    ///
    /// Default: `0..=0`
    pub ignore_range: RangeInclusive<u8>,

    /// Maximum estimated notes-per-second for the realtime note limiter.
    ///
    /// A `NoteOn` is skipped when the estimated NPS exceeds
    /// `max_nps * velocity / 127` (quiet notes are dropped first).
    /// Set to `0` to disable the limiter entirely (no notes are dropped
    /// by the NPS guard).
    ///
    /// Default: `10_000`
    pub max_nps: u64,
}

impl Default for XSynthRealtimeConfig {
    fn default() -> Self {
        Self {
            channel_init_options: Default::default(),
            render_window_ms: 10.0,
            cushion_ms: 100.0,
            global_max_voices: None,
            format: Default::default(),
            multithreading: ThreadCount::None,
            ignore_range: 0..=0,
            max_nps: 10_000,
        }
    }
}
