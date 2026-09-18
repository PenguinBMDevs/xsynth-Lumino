use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use super::{
    channel_sf::ChannelSoundfont, event::KeyNoteEvent, voice_buffer::StealTier,
    voice_buffer::VoiceBuffer, ChannelInitOptions, VoiceControlData,
};

pub struct KeyData {
    key: u8,
    voices: VoiceBuffer,
    last_voice_count: usize,
    shared_voice_counter: Arc<AtomicU64>,
}

impl KeyData {
    pub fn new(
        key: u8,
        shared_voice_counter: Arc<AtomicU64>,
        options: ChannelInitOptions,
    ) -> KeyData {
        KeyData {
            key,
            voices: VoiceBuffer::new(options),
            last_voice_count: 0,
            shared_voice_counter,
        }
    }

    pub fn send_event(
        &mut self,
        event: KeyNoteEvent,
        control: &VoiceControlData,
        channel_sf: &ChannelSoundfont,
        max_layers: Option<usize>,
    ) {
        match event {
            KeyNoteEvent::On(vel) => {
                let voices = channel_sf.spawn_voices_attack(control, self.key, vel);
                self.voices.push_voices(voices, max_layers);
            }
            KeyNoteEvent::Off => {
                let vel = self.voices.release_next_voice();
                if let Some(vel) = vel {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
                }
            }
            KeyNoteEvent::AllOff => {
                while let Some(vel) = self.voices.release_next_voice() {
                    let voices = channel_sf.spawn_voices_release(control, self.key, vel);
                    self.voices.push_voices(voices, max_layers);
                }
            }
            KeyNoteEvent::AllKilled => {
                self.voices.kill_all_voices();
            }
        }
    }

    pub fn process_controls(&mut self, control: &VoiceControlData) {
        for voice in &mut self.voices.iter_voices_mut() {
            voice.process_controls(control);
        }
    }

    pub fn render_to(&mut self, out: &mut [f32]) {
        if self.has_voices() {
            for voice in &mut self.voices.iter_voices_mut() {
                voice.render_to(out);
            }
            self.voices.remove_ended_voices();
        }

        let voice_count = self.voices.voice_count();
        let change = voice_count as i64 - self.last_voice_count as i64;
        if change < 0 {
            self.shared_voice_counter
                .fetch_sub((-change) as u64, Ordering::SeqCst);
        } else {
            self.shared_voice_counter
                .fetch_add(change as u64, Ordering::SeqCst);
        }
        self.last_voice_count = voice_count;
    }

    pub fn has_voices(&self) -> bool {
        self.voices.has_voices()
    }

    /// 当前活跃（未被 Kill）的声部组数量（用于每通道声部上限治理）。
    pub fn active_voice_count(&self) -> usize {
        self.voices.active_voice_count()
    }

    /// 当前缓冲中的声部组总数（含已 Kill 待移除的组，用于诊断）。
    pub fn voice_count(&self) -> usize {
        self.voices.voice_count()
    }

    /// 按分级策略抢占一组声部（短淡出 + 死期限）。返回被抢层级。
    pub fn steal_voice_group(&mut self) -> Option<StealTier> {
        self.voices.steal_voice_group()
    }

    /// 硬移除最老的一组声部（L2 重度治理）。
    pub fn hard_steal_oldest(&mut self) -> bool {
        self.voices.hard_steal_oldest()
    }

    /// 看门狗：仅保留最新 `keep` 组声部。
    pub fn trim_to_newest(&mut self, keep: usize) {
        self.voices.trim_to_newest(keep);
    }

    pub fn set_damper(&mut self, damper: bool) {
        self.voices.set_damper(damper);
    }

    pub fn kill_by_exclusive_class(&mut self, class: u8) {
        self.voices.kill_by_exclusive_class(class);
    }
}
