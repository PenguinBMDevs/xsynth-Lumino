use std::sync::{atomic::AtomicU64, Arc};

use crate::{
    effects::MultiChannelBiQuad,
    helpers::{prepapre_cache_vec, sum_simd},
    voice::{BatchLane, Voice, VoiceControlData},
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
        crate::profiling::tracy_zone!("ch_apply_events", {
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
        });

        // A/B 归一化基准：本块渲染前的活跃声部数（= 本块工作量）。
        // 跨 run 比较时用它把 DSP 耗时换算成 ns/voice-block，消除"渲染内容漂移"
        // 造成的负载差异（见 profiling.rs `tracy_plot` 说明）。未启用 tracy 时零开销。
        #[cfg(feature = "tracy")]
        crate::profiling::tracy_plot!("ch_voices", self.count_voices() as f64);

        // 3) 渲染（可并行）。
        out.fill(0.0);
        crate::profiling::tracy_zone!("channel_keys", {
            // B1 批渲染：仅在单线程（无 key 级线程池）且立体声、SIMD 宽度足够时启用。
            if self.batched_render_enabled() {
                crate::profiling::tracy_zone!("ch_keys_render", {
                    self.render_batched(out);
                });
                crate::profiling::tracy_zone!("channel_effects", {
                    self.apply_channel_effects(out);
                });
                return;
            }
            match self.threadpool.as_ref() {
                Some(pool) => {
                    let len = out.len();
                    let key_voices = &mut self.key_voices;
                    crate::profiling::tracy_zone!("ch_keys_render", {
                        pool.install(|| {
                            key_voices.par_iter_mut().for_each(move |key| {
                                prepapre_cache_vec(&mut key.audio_cache, len, 0.0);
                                key.data.render_to(&mut key.audio_cache);
                            });
                        });
                    });

                    crate::profiling::tracy_zone!("ch_keys_sum", {
                        for key in self.key_voices.iter() {
                            sum_simd(&key.audio_cache, out);
                        }
                    });
                }
                None => {
                    crate::profiling::tracy_zone!("ch_keys_render", {
                        for key in self.key_voices.iter_mut() {
                            key.data.render_to(out);
                        }
                    });
                }
            }
        });

        crate::profiling::tracy_zone!("channel_effects", {
            self.apply_channel_effects(out);
        });
    }

    /// B1 批渲染是否可用：`LUMINO_BATCH`（默认开）+ 无 key 级线程池（只有单线程渲染才能
    /// 跨 key 收集连续的可批 lane）+ 立体声输出 + SIMD 宽度足够。
    fn batched_render_enabled(&self) -> bool {
        self.threadpool.is_none()
            && self.stream_params.channels == ChannelCount::Stereo
            && crate::voice::batching_supported()
    }

    /// 跨 key 批渲染。
    ///
    /// 遍历「key 顺序 → key 内 voice 顺序」的扁平序列（与 `LUMINO_BATCH=0` 的
    /// 渲染顺序一致），把连续的可批 voice 按运行时 SIMD 宽度分块交给批内核；
    /// 前导/后继/孤立（不足一个 chunk）的 voice 仍逐 voice 渲染，因此逐 voice
    /// 的累加顺序不变（批内 lane 之间为树形求和，见 `voice/batch.rs`）。
    fn render_batched(&mut self, out: &mut [f32]) {
        let frames = out.len() / 2;
        let width = crate::voice::batch_chunk_width();
        {
            let mut all: Vec<&mut Box<dyn Voice>> = self
                .key_voices
                .iter_mut()
                .flat_map(|key| key.data.iter_voices_mut())
                .collect();
            let mut i = 0usize;
            while i < all.len() {
                if all[i].batch_lane().is_none() {
                    all[i].render_to(out);
                    i += 1;
                    continue;
                }
                let start = i;
                while i < all.len() && all[i].batch_lane().is_some() {
                    i += 1;
                }
                let end = i;

                let mut j = start;
                while j + width <= end {
                    if !render_voice_chunk(&mut all[j..j + width], out, frames) {
                        for voice in all[j..j + width].iter_mut() {
                            voice.render_to(out);
                        }
                    }
                    j += width;
                }
                for voice in all[j..end].iter_mut() {
                    voice.render_to(out);
                }
            }
        }

        for key in self.key_voices.iter_mut() {
            if key.data.has_voices() {
                key.data.remove_ended_voices();
            }
        }
    }

    /// 按分级策略抢占 `count` 个活跃声部（供跨通道全局治理调用）。
    ///
    /// 与 `enforce_max_voices` 相同：从最忙的键、按 T1 释放中最轻 → T2 最轻
    /// 依次抢占；实际抢占数可能小于 `count`（活跃声部不足时）。
    pub fn steal_voices(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let mut counts: Vec<usize> = self
            .key_voices
            .iter()
            .map(|key| key.data.active_voice_count())
            .collect();
        let total: usize = counts.iter().sum();
        let mut remaining = count.min(total);
        while remaining > 0 {
            let Some(idx) = counts
                .iter()
                .enumerate()
                .filter(|(_, &count)| count > 0)
                .max_by_key(|(_, &count)| count)
                .map(|(idx, _)| idx)
            else {
                break;
            };
            if self.key_voices[idx].data.steal_voice_group().is_none() {
                break;
            }
            counts[idx] -= 1;
            remaining -= 1;
        }
    }

    /// 硬抢占 `count` 个活跃声部（L2 重度治理：跳过淡出，立即移除）。
    ///
    /// 批量化：按各键声部数降序，从最满的键整段弹出，避免"每弹一个都重扫全部键"。
    pub fn steal_voices_hard(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let mut order: Vec<usize> = (0..self.key_voices.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(self.key_voices[i].data.voice_count()));
        let mut remaining = count;
        for idx in order {
            if remaining == 0 {
                break;
            }
            while remaining > 0 && self.key_voices[idx].data.hard_steal_oldest() {
                remaining -= 1;
            }
        }
    }

    /// 看门狗自愈：每键仅保留最新 `keep` 组声部，其余立即移除。
    pub fn trim_to_newest(&mut self, keep: usize) {
        for key in self.key_voices.iter_mut() {
            key.data.trim_to_newest(keep);
        }
    }

    /// 每通道声部上限：超过 `options.max_voices` 时，从"活跃声部最多的键"里
    /// 按分级策略抢占**活跃**声部（T1 释放中最轻 → T2 最轻），直到回到上限内。
    ///
    /// - 活跃数不含已 Kill 的组：被抢的组走 1ms 淡出 + 死期限（见
    ///   `VoiceBuffer::kill_voice_fade_out`），约 2 块后强制移除；
    /// - 已 Kill 的组不会被再次抢占（否则活跃计数虚降、治理失效，
    ///   此前实测导致活跃声部无界增长）；
    /// - 自然保护：持续长音/低音（力度响、未释放）不会因"最老"被优先命中，
    ///   只有连衰减音与轻音都不存在时才会被抢。
    ///
    /// 性能：计数只统计一次（O(total)），此后在 128 个计数上增量维护；
    /// 单次抢占 = 选键 O(128) + 在单个键内扫描 O(cap)。
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

        // 诊断（XSYNTH_GOV_DEBUG=1）：区分"活跃数未压住"与"缓冲残留滞留"。
        // 开关只解析一次：每通道每块读取环境变量是 syscall 级开销（重载时 16×100 次/秒）。
        static GOV_DEBUG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let debug = *GOV_DEBUG.get_or_init(|| std::env::var_os("XSYNTH_GOV_DEBUG").is_some());
        let active_before = total;
        let buffer_before: usize = self
            .key_voices
            .iter()
            .map(|key| key.data.voice_count())
            .sum();

        let mut steals = 0usize;
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
            if self.key_voices[idx].data.steal_voice_group().is_none() {
                break;
            }
            counts[idx] -= 1;
            total -= 1;
            steals += 1;
        }

        if debug {
            use std::sync::atomic::{AtomicU64, Ordering};
            static EVENTS: AtomicU64 = AtomicU64::new(0);
            let n = EVENTS.fetch_add(1, Ordering::Relaxed);
            if n.is_multiple_of(200) {
                let buffer_after: usize = self
                    .key_voices
                    .iter()
                    .map(|key| key.data.voice_count())
                    .sum();
                eprintln!(
                    "[GOV] #{n} cap={cap} active_before={active_before} active_after={total} buffer_before={buffer_before} buffer_after={buffer_after} steals={steals}",
                );
            }
        }
    }

    /// 通道内缓冲中的声部总数：A/B 渲染工作量归一化基准（仅 tracy 构建调用）。
    #[cfg(feature = "tracy")]
    fn count_voices(&self) -> usize {
        self.key_voices
            .iter()
            .map(|key| key.data.voice_count())
            .sum()
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

/// 把一个 chunk（恰好 SIMD 宽度个 voice）交给批内核。
///
/// 返回 `false` 表示不可批（存在非 lane voice，或 biquad 开关不一致导致无法整块
/// 向量化），由调用方回退到逐 voice 渲染；回退不改变任何状态。
fn render_voice_chunk(voices: &mut [&mut Box<dyn Voice>], out: &mut [f32], frames: usize) -> bool {
    let mut lanes: Vec<&mut BatchLane> = Vec::with_capacity(voices.len());
    for voice in voices.iter_mut() {
        match voice.batch_lane() {
            Some(lane) => lanes.push(lane),
            None => return false,
        }
    }
    let Some(first) = lanes.first() else {
        return false;
    };
    let filter_enabled = first.filter_enabled();
    if lanes
        .iter()
        .any(|lane| lane.filter_enabled() != filter_enabled)
    {
        return false;
    }
    crate::voice::render_batch_chunk(&mut lanes, out, frames)
}

#[cfg(test)]
mod batch_render_tests {
    //! B1 批渲染接线测试（`VoiceChannel::render_batched`）。
    //!
    //! 目标：验证「扁平 voice 序列 → 连续可批 run → 按 SIMD 宽度分块 → 尾部/非可批
    //! voice 回退」的接线不丢 voice、不重复渲染，并与逐 voice 路径输出一致
    //! （差异仅来自批内树形求和的 f32 舍入）。

    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;
    use crate::effects::{BiQuadFilter, FilterType};
    use crate::voice::{
        BatchLaneInit, EnvelopeDescriptor, ReleaseType, StereoBatchVoice, VoiceControlData,
        VoiceGeneratorBase, VoiceSampleGenerator,
    };
    use xsynth_soundfonts::LoopMode;

    const SR: u32 = 48_000;
    const FRAMES: usize = 480;

    /// 常量 voice：每帧给左右声道各加 `value`；统计 `render_to` 次数以验证
    /// 「既不跳过也不重复渲染」。
    struct ConstVoice {
        value: f32,
        renders: Arc<AtomicUsize>,
    }

    impl VoiceGeneratorBase for ConstVoice {
        fn ended(&self) -> bool {
            false
        }
        fn signal_release(&mut self, _rel_type: ReleaseType) {}
        fn process_controls(&mut self, _control: &VoiceControlData) {}
    }

    impl VoiceSampleGenerator for ConstVoice {
        fn render_to(&mut self, buffer: &mut [f32]) {
            self.renders.fetch_add(1, Ordering::Relaxed);
            for sample in buffer.iter_mut() {
                *sample += self.value;
            }
        }
    }

    impl Voice for ConstVoice {
        fn is_releasing(&self) -> bool {
            false
        }
        fn is_killed(&self) -> bool {
            false
        }
        fn velocity(&self) -> u8 {
            100
        }
        fn exclusive_class(&self) -> Option<u8> {
            None
        }
    }

    fn batch_voice(amp: f32) -> Box<dyn Voice> {
        batch_voice_with_velocity(amp, 100)
    }

    /// 同上，但指定力度（复现「低力度持音 vs 高力度新音」的抢占优先级）。
    fn batch_voice_with_velocity(amp: f32, velocity: u8) -> Box<dyn Voice> {
        batch_voice_impl(amp, velocity, LoopMode::LoopSustain)
    }

    /// NoLoop 形态（app 音源 TWGMD/Nexus/GarageBand 的区域全是 NoLoop）：
    /// `signal_release` 不改写读取位置，因此可用来单独观察「抢占是否做了淡出」。
    fn batch_voice_noloop(amp: f32, velocity: u8) -> Box<dyn Voice> {
        batch_voice_impl(amp, velocity, LoopMode::NoLoop)
    }

    fn batch_voice_impl(amp: f32, velocity: u8, loop_mode: LoopMode) -> Box<dyn Voice> {
        let samples: Arc<[f32]> = (0..4_096)
            .map(|i| (i as f32 * 0.01).sin() * 0.5)
            .collect::<Vec<_>>()
            .into();
        let env = EnvelopeDescriptor {
            start_percent: 0.0,
            delay: 0.0,
            attack: 0.01,
            hold: 0.0,
            decay: 0.3,
            sustain_percent: 0.5,
            release: 0.05,
        }
        .to_envelope_params(SR, Default::default());
        let init = BatchLaneInit {
            speed_mult: 1.0,
            gain_l: amp,
            gain_r: amp,
            samples_l: samples.clone(),
            samples_r: samples,
            loop_mode,
            loop_offset: 0,
            loop_start: 100,
            loop_end: 3_000,
            loop_stop: None,
            interpolator: crate::soundfont::Interpolator::Nearest,
            filter: Some(BiQuadFilter::new(
                FilterType::LowPass,
                9_000.0,
                SR as f32,
                Some(0.7),
            )),
            envelope: env,
            sample_rate: SR as f32,
            group_len: crate::voice::batch_chunk_width() as u8,
            velocity,
            exclusive_class: None,
        };
        Box::new(StereoBatchVoice::new(
            &init,
            &VoiceControlData::new_defaults(),
        ))
    }

    /// 构造一个带脚本化 voice 布局的通道：跨 key 的连续批 run、run 被非可批 voice
    /// 打断、以及不足一个 chunk 的尾部。
    fn build_channel(counters: &[Arc<AtomicUsize>]) -> VoiceChannel {
        let mut channel = VoiceChannel::new(
            ChannelInitOptions::default(),
            AudioStreamParams::new(SR, ChannelCount::Stereo),
            None,
        );
        // key 0：8 个可批（正好一个 chunk）
        for _ in 0..8 {
            channel.key_voices[0].data.push_voice_test(batch_voice(0.4));
        }
        // key 1：2 个可批 + 常量 voice（打断 run）+ 8 个可批
        channel.key_voices[1].data.push_voice_test(batch_voice(0.4));
        channel.key_voices[1].data.push_voice_test(batch_voice(0.4));
        channel.key_voices[1]
            .data
            .push_voice_test(Box::new(ConstVoice {
                value: 1.0,
                renders: counters[0].clone(),
            }));
        for _ in 0..8 {
            channel.key_voices[1]
                .data
                .push_voice_test(batch_voice(0.25));
        }
        // key 2：3 个可批（不足一个 chunk 的尾部）+ 常量 voice
        for _ in 0..3 {
            channel.key_voices[2].data.push_voice_test(batch_voice(0.3));
        }
        channel.key_voices[2]
            .data
            .push_voice_test(Box::new(ConstVoice {
                value: 2.0,
                renders: counters[1].clone(),
            }));
        channel
    }

    /// 端到端回归（用户症状）：通道渲染路径下，按住不放的 sustain 长音不得被
    /// `remove_ended_voices` 提前移除。
    ///
    /// 判据：`voice_count()` 在每一块都必须保持不变（声部被移除 = 硬切，无 release
    /// 尾巴），且末块仍在发声。批渲染路径与逐 voice 路径都要成立——缺陷在采样读取器
    /// 本身，两条路径共享同一语义（批 lane 是镜像）。
    #[test]
    fn held_sustain_voice_survives_channel_voice_reaping() {
        const VOICES: usize = 8;
        const BLOCKS: usize = 40;

        let mut batched = VoiceChannel::new(
            ChannelInitOptions::default(),
            AudioStreamParams::new(SR, ChannelCount::Stereo),
            None,
        );
        let mut per_voice = VoiceChannel::new(
            ChannelInitOptions::default(),
            AudioStreamParams::new(SR, ChannelCount::Stereo),
            None,
        );
        for _ in 0..VOICES {
            batched.key_voices[0].data.push_voice_test(batch_voice(0.4));
            per_voice.key_voices[0]
                .data
                .push_voice_test(batch_voice(0.4));
        }

        let mut batched_buf = vec![0.0f32; FRAMES * 2];
        let mut per_voice_buf = vec![0.0f32; FRAMES * 2];
        for block in 0..BLOCKS {
            batched_buf.fill(0.0);
            per_voice_buf.fill(0.0);
            batched.render_batched(&mut batched_buf);
            for key in per_voice.key_voices.iter_mut() {
                key.data.render_to(&mut per_voice_buf);
            }

            assert_eq!(
                batched.key_voices[0].data.voice_count(),
                VOICES,
                "批渲染：第 {block} 块后长音声部被提前移除（硬切）"
            );
            assert_eq!(
                per_voice.key_voices[0].data.voice_count(),
                VOICES,
                "逐 voice：第 {block} 块后长音声部被提前移除（硬切）"
            );
        }

        assert!(
            batched_buf.iter().any(|s| s.abs() > 1e-3),
            "末块批渲染输出应为非静音（声部仍在发声）"
        );
        assert!(
            per_voice_buf.iter().any(|s| s.abs() > 1e-3),
            "末块逐 voice 输出应为非静音（声部仍在发声）"
        );
    }

    /// 音频级回归（复现 app 配置：`SetLayerCount(Some(4))` + `fade_out_killing = false`）：
    /// 同键反复触发导致每键上限抢占时，输出**不得出现采样级硬切**。
    ///
    /// 用 NoLoop 形态（= app 音源的实际形态）：`signal_release` 不改写读取位置，
    /// 因此块内跳变只可能来自「抢占是否做了淡出」。
    /// 判据：块内相邻采样最大跳变（同声道）。正常信号（~76Hz 正弦）每采样斜率 ~1e-3；
    /// 零淡出硬切 = 被抢声部的瞬时幅值（~1e-1）直接消失 → 跳变放大两个数量级；
    /// 1ms 淡出（48 帧 @48k）只贡献 ~幅值/48 的斜率，与信号自身同量级。
    #[test]
    fn layer_cap_steal_fades_instead_of_hard_cutting_audio() {
        fn max_step_same_channel(buf: &[f32], prev_last: (f32, f32)) -> f32 {
            let frames = buf.len() / 2;
            // 跨块边界：抢占是在块间发生的，硬切首先出现在边界上（历史盲区：只测块内会漏）。
            let mut best = (buf[0] - prev_last.0)
                .abs()
                .max((buf[1] - prev_last.1).abs());
            for i in 1..frames {
                best = best.max((buf[i * 2] - buf[(i - 1) * 2]).abs());
                best = best.max((buf[i * 2 + 1] - buf[(i - 1) * 2 + 1]).abs());
            }
            best
        }

        fn last_frame(buf: &[f32]) -> (f32, f32) {
            let frames = buf.len() / 2;
            (buf[(frames - 1) * 2], buf[(frames - 1) * 2 + 1])
        }

        const MAX_LAYERS: usize = 4;
        let mut channel = VoiceChannel::new(
            ChannelInitOptions {
                fade_out_killing: false,
                max_voices: None,
            },
            AudioStreamParams::new(SR, ChannelCount::Stereo),
            None,
        );

        // 按住的长音（低力度 → 若不保护，会因"最轻"被优先抢走）。
        channel.key_voices[0]
            .data
            .push_voice_test_capped(batch_voice_noloop(0.4, 10), MAX_LAYERS);

        let mut buf = vec![0.0f32; FRAMES * 2];
        let mut prev = (0.0f32, 0.0f32);
        buf.fill(0.0);
        channel.render_batched(&mut buf);
        let calm_step = max_step_same_channel(&buf, prev);
        prev = last_frame(&buf);

        // 同键再触发 MAX_LAYERS 次：第 4 次会让活跃数超过上限 → 触发抢占。
        let mut steal_step = 0.0f32;
        for _ in 0..MAX_LAYERS {
            channel.key_voices[0]
                .data
                .push_voice_test_capped(batch_voice_noloop(0.4, 100), MAX_LAYERS);
            buf.fill(0.0);
            channel.render_batched(&mut buf);
            steal_step = steal_step.max(max_step_same_channel(&buf, prev));
            prev = last_frame(&buf);
        }

        assert!(
            steal_step < 0.03,
            "抢占块出现了采样级硬切：steal_step={steal_step}（calm_step={calm_step}）\
             —— 每键上限抢占必须走 1ms 淡出"
        );
        assert!(
            channel.key_voices[0].data.active_voice_count() <= MAX_LAYERS,
            "每键活跃声部上限必须维持"
        );
    }
    #[test]
    fn render_batched_matches_per_voice_render_and_visits_every_voice() {
        let batched_counters = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let scalar_counters = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let mut batched = build_channel(&batched_counters);
        let mut scalar = build_channel(&scalar_counters);

        let mut batched_buf = vec![0.0f32; FRAMES * 2];
        let mut scalar_buf = vec![0.0f32; FRAMES * 2];

        for block in 0..4 {
            batched_buf.fill(0.0);
            scalar_buf.fill(0.0);
            batched.render_batched(&mut batched_buf);
            for key in scalar.key_voices.iter_mut() {
                key.data.render_to(&mut scalar_buf);
            }

            let mut max_diff = 0.0f32;
            let mut peak = 0.0f32;
            for (b, s) in batched_buf.iter().zip(scalar_buf.iter()) {
                max_diff = max_diff.max((b - s).abs());
                peak = peak.max(s.abs());
            }
            assert!(
                max_diff <= 1e-5 + 1e-5 * peak,
                "第 {block} 块批/逐 voice 输出差异超容差: max_diff={max_diff} peak={peak}"
            );
        }

        // 每块恰好渲染一次：4 块 → 每个常量 voice 各 4 次。
        for (i, counter) in batched_counters.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::Relaxed),
                4,
                "批路径第 {i} 个非可批 voice 的渲染次数异常（跳过或重复）"
            );
        }
        for (i, counter) in scalar_counters.iter().enumerate() {
            assert_eq!(
                counter.load(Ordering::Relaxed),
                4,
                "标量路径第 {i} 个 voice 次数异常"
            );
        }

        // 常量贡献必须完整出现在输出里（0.4/0.25/0.3 的批 voice 与之叠加，
        // 用「去掉常量 voice 的通道」做差验证会更复杂，这里退而验证信号非零且量级正确）。
        assert!(
            batched_buf.iter().any(|s| s.abs() > 1.0),
            "输出中应包含常量 voice 的贡献"
        );
    }
}
