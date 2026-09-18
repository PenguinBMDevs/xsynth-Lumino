use std::sync::{atomic::AtomicU64, Arc};

use crate::{
    effects::MultiChannelBiQuad,
    helpers::{prepapre_cache_vec, sum_simd},
    voice::VoiceControlData,
    AudioStreamParams, ChannelCount,
};

use xsynth_soundfonts::FilterType;

use self::{control::ControlEventData, key::KeyData, params::VoiceChannelParams};

use super::AudioPipe;

use rayon::prelude::*;

mod channel_sf;
mod control;
mod key;
mod params;
mod voice_buffer;
mod voice_spawner;

mod event;
pub use event::*;

pub(crate) use control::ValueLerp;
pub use params::VoiceChannelStatsReader;

struct Key {
    data: KeyData,
    audio_cache: Vec<f32>,
    event_cache: Vec<KeyNoteEvent>,
}

impl Key {
    pub fn new(key: u8, shared_voice_counter: Arc<AtomicU64>, options: ChannelInitOptions) -> Self {
        Key {
            data: KeyData::new(key, shared_voice_counter, options),
            audio_cache: Vec::new(),
            event_cache: Vec::new(),
        }
    }
}

/// Options for initializing a new VoiceChannel.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize),
    serde(default)
)]
pub struct ChannelInitOptions {
    /// If set to true, the voices killed due to the voice limit will fade out.
    /// If set to false, they will be killed immediately, usually causing clicking
    /// but improving performance.
    ///
    /// Default: `false`
    pub fade_out_killing: bool,

    /// Maximum number of active voices per channel (`None` = unlimited,
    /// `Some(0)` is treated as unlimited too).
    ///
    /// When exceeded, the oldest voice groups of the busiest keys are released
    /// first, so newly played notes keep sounding (dense black-MIDI safety
    /// valve that keeps the render load bounded without dropping new notes).
    ///
    /// Default: `None`
    pub max_voices: Option<usize>,
}

#[allow(clippy::derivable_impls)]
impl Default for ChannelInitOptions {
    fn default() -> Self {
        Self {
            fade_out_killing: false,
            max_voices: None,
        }
    }
}

/// Represents a single MIDI channel within XSynth.
///
/// Keeps track and manages MIDI events and the active voices of a channel.
///
/// MIDI CC Support Chart:
/// - `CC0`: Bank Select
/// - `CC6`, `CC38`, `CC100`, `CC101`: RPN & NRPN
/// - `CC7`: Volume
/// - `CC8`: Balance
/// - `CC10`: Pan
/// - `CC11`: Expression
/// - `CC64`: Damper pedal
/// - `CC71`: Cutoff resonance
/// - `CC72`: Release time multiplier
/// - `CC73`: Attack time multiplier
/// - `CC74`: Cutoff frequency
/// - `CC120`: All sounds off
/// - `CC121`: Reset all controllers
/// - `CC123`: All notes off
pub struct VoiceChannel {
    key_voices: Vec<Key>,

    params: VoiceChannelParams,
    threadpool: Option<Arc<rayon::ThreadPool>>,

    stream_params: AudioStreamParams,

    /// Channel configuration (fade-out kills, per-channel voice cap).
    options: ChannelInitOptions,

    /// The helper struct for keeping track of MIDI control event data
    control_event_data: ControlEventData,

    /// Processed control data, ready to feed to voices
    voice_control_data: VoiceControlData,

    /// Effects
    cutoff: MultiChannelBiQuad,
}

impl VoiceChannel {
    /// Initializes a new voice channel.
    ///
    /// - `options`: Channel configuration
    /// - `stream_params`: Parameters of the output audio
    /// - `threadpool`: The thread-pool that will be used to render the individual
    ///   keys' voices concurrently. If None is used, the voices will be
    ///   rendered on the same thread.
    pub fn new(
        options: ChannelInitOptions,
        stream_params: AudioStreamParams,
        threadpool: Option<Arc<rayon::ThreadPool>>,
    ) -> VoiceChannel {
        fn fill_key_array<T, F: Fn(u8) -> T>(func: F) -> Vec<T> {
            let mut vec = Vec::with_capacity(128);
            for i in 0..128 {
                vec.push(func(i));
            }
            vec
        }

        let params = VoiceChannelParams::new(stream_params);
        let shared_voice_counter = params.stats.voice_counter.clone();

        VoiceChannel {
            params,
            key_voices: fill_key_array(|i| Key::new(i, shared_voice_counter.clone(), options)),

            threadpool,

            stream_params,

            options,

            control_event_data: ControlEventData::new_defaults(stream_params.sample_rate),
            voice_control_data: VoiceControlData::new_defaults(),

            cutoff: MultiChannelBiQuad::new(
                stream_params.channels.count() as usize,
                FilterType::LowPass,
                stream_params.sample_rate as f32 / 2.0,
                stream_params.sample_rate as f32,
                None,
            ),
        }
    }

    fn apply_channel_effects(&mut self, out: &mut [f32]) {
        let control = &mut self.control_event_data;

        match self.stream_params.channels {
            ChannelCount::Mono => {
                // Volume
                for sample in out.iter_mut() {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    *sample *= vol;
                }
            }
            ChannelCount::Stereo => {
                // Volume
                for sample in out.chunks_mut(2) {
                    let vol = control.volume.get_next() * control.expression.get_next();
                    let vol = vol.powi(2);
                    sample[0] *= vol;
                    sample[1] *= vol;
                }

                // Pan
                for sample in out.chunks_mut(2) {
                    let pan = control.pan.get_next();
                    sample[0] *= ((pan * std::f32::consts::PI / 2.0).cos()).min(1.0);
                    sample[1] *= ((pan * std::f32::consts::PI / 2.0).sin()).min(1.0);
                }
            }
        }

        // Cutoff
        if let Some(cutoff) = control.cutoff {
            self.cutoff
                .set_filter_type(FilterType::LowPass, cutoff, control.resonance);
            self.cutoff.process(out);
        }
    }

    fn push_key_events_and_render(&mut self, out: &mut [f32]) {
        self.params.load_program();

        // 1) 应用本块的全部事件（单线程，代价低），使声部统计反映本块最新状态。
        for key in self.key_voices.iter_mut() {
            for e in key.event_cache.drain(..) {
                key.data.send_event(
                    e,
                    &self.voice_control_data,
                    &self.params.channel_sf,
                    self.params.layers,
                );
            }
        }

        // 2) 每通道声部上限治理：超限时优先杀"最老"的声部组（保留最新音符），
        //    使渲染负载有上界，同时不丢刚触发的音符。
        self.enforce_max_voices();

        // 3) 渲染（可并行）。
        out.fill(0.0);
        match self.threadpool.as_ref() {
            Some(pool) => {
                let len = out.len();
                let key_voices = &mut self.key_voices;
                pool.install(|| {
                    key_voices.par_iter_mut().for_each(move |key| {
                        prepapre_cache_vec(&mut key.audio_cache, len, 0.0);
                        key.data.render_to(&mut key.audio_cache);
                    });
                });

                for key in self.key_voices.iter() {
                    sum_simd(&key.audio_cache, out);
                }
            }
            None => {
                for key in self.key_voices.iter_mut() {
                    key.data.render_to(out);
                }
            }
        }

        self.apply_channel_effects(out);
    }

    /// 每通道声部上限：超过 `options.max_voices` 时，从"声部最多"的键里
    /// 反复释放最老的一组声部（`VecDeque` 前端），直到回到上限内。
    ///
    /// 选择最忙的键可把抢占集中在压力最大的键上；选择最老的一组保证
    /// 新音符（当前正在演奏）不被丢弃——避免"杀还没播放的音符"造成断续。
    ///
    /// 性能：活跃数只统计一次（O(total)），之后在 128 个计数上增量维护
    /// （每次抢占 O(128)），避免"每杀一个都全量重扫"导致的二次复杂度卡死。
    fn enforce_max_voices(&mut self) {
        let Some(cap) = self.options.max_voices else {
            return;
        };
        // 0 视为不限，避免歧义。
        if cap == 0 {
            return;
        }

        let mut counts: Vec<usize> = self
            .key_voices
            .iter()
            .map(|key| key.data.active_voice_count())
            .collect();
        let mut total: usize = counts.iter().sum();
        if total <= cap {
            return;
        }

        while total > cap {
            let Some(idx) = counts
                .iter()
                .enumerate()
                .filter(|(_, &count)| count > 0)
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
            else {
                break;
            };
            if !self.key_voices[idx].data.release_oldest_voice_group() {
                break;
            }
            counts[idx] -= 1;
            total -= 1;
        }
    }

    fn propagate_voice_controls(&mut self) {
        for key in self.key_voices.iter_mut() {
            key.data.process_controls(&self.voice_control_data);
        }
    }

    fn kill_voices_in_exclusive_class(&mut self, class: u8) {
        for key in self.key_voices.iter_mut() {
            key.data.kill_by_exclusive_class(class);
        }
    }

    /// Sends a ChannelEvent to the channel.
    /// See the `ChannelEvent` documentation for more information.
    pub fn process_event(&mut self, event: ChannelEvent) {
        self.push_events_iter(std::iter::once(event));
    }

    /// Sends multiple ChannelEvent items to the channel as an iterator.
    pub fn push_events_iter<T: Iterator<Item = ChannelEvent>>(&mut self, iter: T) {
        for e in iter {
            match e {
                ChannelEvent::Audio(audio) => match audio {
                    ChannelAudioEvent::NoteOn { key, vel } => {
                        let classes: Vec<_> = self
                            .params
                            .channel_sf
                            .exclusive_classes_attack(key, vel)
                            .collect();
                        for class in classes {
                            self.kill_voices_in_exclusive_class(class);
                        }
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::On(vel);
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::NoteOff { key } => {
                        if let Some(key) = self.key_voices.get_mut(key as usize) {
                            let ev = KeyNoteEvent::Off;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesOff => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllOff;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::AllNotesKilled => {
                        for key in self.key_voices.iter_mut() {
                            let ev = KeyNoteEvent::AllKilled;
                            key.event_cache.push(ev);
                        }
                    }
                    ChannelAudioEvent::ResetControl => {
                        self.reset_control();
                    }
                    ChannelAudioEvent::Control(control) => {
                        self.process_control_event(control);
                    }
                    ChannelAudioEvent::ProgramChange(preset) => {
                        self.params.set_preset(preset);
                    }
                    ChannelAudioEvent::SystemReset => {
                        for key in self.key_voices.iter_mut() {
                            key.event_cache.clear();
                            key.event_cache.push(KeyNoteEvent::AllKilled);
                        }
                        self.reset_control();
                        self.reset_program();
                    }
                },
                ChannelEvent::Config(config) => self.params.process_config_event(config),
            }
        }
    }

    /// Returns a reader for the VoiceChannel statistics.
    /// See the `VoiceChannelStatsReader` documentation for more information.
    pub fn get_channel_stats(&self) -> VoiceChannelStatsReader {
        let stats = self.params.stats.clone();
        VoiceChannelStatsReader::new(stats)
    }
}

impl AudioPipe for VoiceChannel {
    fn stream_params(&self) -> &AudioStreamParams {
        &self.params.constant.stream_params
    }

    fn read_samples_unchecked(&mut self, out: &mut [f32]) {
        self.push_key_events_and_render(out);
    }
}
