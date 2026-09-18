use std::ops::RangeInclusive;
pub use xsynth_core::{
    channel::ChannelInitOptions,
    channel_group::{SynthFormat, ThreadCount},
};

/// `global_max_voices` 为 `None`/`0`（自动模式）时使用的默认硬上限。
///
/// 依据实测标定：96kHz + 重型音色库下，本机 load≈0.96 对应约 1 万声部；
/// 软目标再乘 `voice_target_ratio`（默认 `1 - 1/e ≈ 0.632`）留出暂态余量。
pub const DEFAULT_HARD_MAX_VOICES: usize = 10_000;

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

    /// Maximum number of active voices summed across **all** channels.
    ///
    /// `None` or `Some(0)` means **automatic**: the engine uses
    /// [`DEFAULT_HARD_MAX_VOICES`] as the hard ceiling and lets the load
    /// governor pick the actual runtime target. A positive value is the
    /// user-set hard ceiling (量程): the governor's soft target never exceeds
    /// `voice_target_ratio * global_max_voices`.
    ///
    /// Enforced by the render pipe: when the runtime target is exceeded, the
    /// busiest channels are asked to steal their oldest/quietest/releasing
    /// voices, so newly played notes keep sounding. This is the cross-channel
    /// counterpart of `channel_init_options.max_voices` (which is per channel).
    ///
    /// Default: `None` (automatic)
    pub global_max_voices: Option<usize>,

    /// 软目标比例：运行目标 = `voice_target_ratio * 硬上限`（负载反馈只会更低，
    /// 不会更高）。默认 `1 - 1/e ≈ 0.632`，即留出约 37% 的暂态余量；
    /// 更激进的 `1 - 1/e^2 ≈ 0.865` 也可用，但突发余量仅 13.5%。
    ///
    /// Default: `1 - 1/e ≈ 0.632`
    pub voice_target_ratio: f64,

    /// 软 NPS 闸（保命开关，默认关闭）：仅当负载持续超限（L2 以上）时，
    /// 由渲染管线临时启用全局限速令牌桶，避免"事件洪峰 → 更过载"的正反馈；
    /// 负载回落后自动解除。关闭时**不存在**任何 NoteOn 丢弃路径。
    ///
    /// Default: `false`
    pub soft_nps_gate: bool,

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
            voice_target_ratio: 1.0 - 1.0 / std::f64::consts::E,
            soft_nps_gate: false,
            format: Default::default(),
            multithreading: ThreadCount::None,
            ignore_range: 0..=0,
            max_nps: 10_000,
        }
    }
}
