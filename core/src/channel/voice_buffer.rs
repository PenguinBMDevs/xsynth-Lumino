use super::ChannelInitOptions;
use crate::voice::{ReleaseType, Voice};
use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

struct GroupVoice {
    pub id: usize,
    pub voice: Box<dyn Voice>,
}

impl Deref for GroupVoice {
    type Target = Box<dyn Voice>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.voice
    }
}

impl DerefMut for GroupVoice {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Box<dyn Voice> {
        &mut self.voice
    }
}

impl Debug for GroupVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("")
            .field(&self.id)
            .field(&self.voice.velocity())
            .field(&self.voice.is_killed())
            .finish()
    }
}

/// 淡出期被杀 voice 的相对保留上限：`max_voices * 2 + 8`。
///
/// `fade_out_killing = true` 时被杀 voice 会保留在 buffer 中，直到本次渲染结束
/// 由 `remove_ended_voices` 清除。高密度事件积压时（一个渲染块内持续到达大量
/// NoteOn），每次 `push_voices` 都会触发全 buffer 扫描（`get_active_count` /
/// `pop_quietest_voice_group`），若保留量无界增长，单块事件处理会退化为
/// O(n²)（黑 MIDI 密集段实测渲染负载可达 13~28 倍实时，且长时间无法恢复）。
/// 给保留量设上界后 buffer 长度被限制在 `active + limit`，扫描成本回到常量级。
fn fading_retention_limit(max_voices: usize) -> usize {
    max_voices.saturating_mul(2).saturating_add(8)
}

pub struct VoiceBuffer {
    options: ChannelInitOptions,
    id_counter: usize,
    buffer: VecDeque<GroupVoice>,
    damper_held: bool,
    held_by_damper: Vec<usize>,
}

impl VoiceBuffer {
    pub fn new(options: ChannelInitOptions) -> Self {
        VoiceBuffer {
            options,
            id_counter: 0,
            buffer: VecDeque::new(),
            damper_held: false,
            held_by_damper: Vec::new(),
        }
    }

    fn get_id(&mut self) -> usize {
        self.id_counter += 1;
        self.id_counter
    }

    /// Pops the quietest voice group. Multiple voices can be part of the same group
    /// based on their ID (e.g. a note and a hammer playing at the same time for a note on event)
    fn pop_quietest_voice_group(&mut self, ignored_id: usize) {
        if self.buffer.is_empty() {
            return;
        }

        let mut quietest = u8::MAX;
        let mut quietest_index = 0;
        let mut quietest_id = 0;
        let mut count = 0;
        for i in 0..self.buffer.len() {
            let voice = &self.buffer[i];
            if voice.id == ignored_id || voice.is_killed() {
                continue;
            }
            let vel = voice.velocity();
            if quietest_id == voice.id {
                count += 1;
            } else if vel < quietest || i == 0 {
                quietest = vel;
                quietest_index = i;
                quietest_id = voice.id;
                count = 1;
            }
        }

        if count > 0 {
            if self.options.fade_out_killing {
                for i in quietest_index..(quietest_index + count) {
                    self.kill_voice_fade_out(i);
                }
            } else {
                self.buffer.drain(quietest_index..(quietest_index + count));
            }

            if let Some(index) = self.held_by_damper.iter().position(|&x| x == quietest_id) {
                self.held_by_damper.remove(index);
            }
        }
    }

    fn kill_voice_fade_out(&mut self, index: usize) {
        self.buffer[index]
            .deref_mut()
            .signal_release(ReleaseType::Kill);
    }

    pub fn kill_all_voices(&mut self) {
        if self.options.fade_out_killing {
            for i in 0..self.buffer.len() {
                self.kill_voice_fade_out(i);
            }
            self.id_counter = 0;
        } else {
            self.buffer.clear();
        }
    }

    pub fn kill_by_exclusive_class(&mut self, class: u8) {
        for voice in &mut self.buffer {
            if voice.exclusive_class() == Some(class) {
                voice.signal_release(ReleaseType::Kill);
            }
        }
    }

    fn get_active_count(&mut self) -> usize {
        let mut active = 0;
        for i in 0..self.buffer.len() {
            if !self.buffer[i].deref().is_killed() {
                active += 1;
            }
        }
        active
    }

    /// Pushes a new set of voices for a single note on event. Multiple voices can be part of the same group
    /// based on their ID (e.g. a note and a hammer playing at the same time for a note on event)
    pub fn push_voices(
        &mut self,
        voices: impl Iterator<Item = Box<dyn Voice>>,
        max_voices: Option<usize>,
    ) {
        let mut len = 0;

        let id = self.get_id();
        for voice in voices {
            self.buffer.push_back(GroupVoice { id, voice });
            len += 1;
        }

        if let Some(max_voices) = max_voices {
            if len > max_voices {
                self.pop_quietest_voice_group(id);
            } else if self.options.fade_out_killing {
                while self.get_active_count() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            } else {
                while self.buffer.len() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            }

            // 淡出保留上限：防止被杀 voice 在事件积压时无界累积，导致
            // `get_active_count` / `pop_quietest_voice_group` 的全 buffer 扫描
            // 在单个渲染块内退化为 O(n²)。仅截短过载时最老的淡出（听感代价最小），
            // 正常负载下保留量低于上限，淡出质量不受影响。
            if self.options.fade_out_killing {
                self.trim_excess_fading_voices(max_voices);
            }
        }
    }

    /// 将被杀（淡出中）voice 的数量裁剪到 `fading_retention_limit` 以内，
    /// 超出部分优先丢弃最老的（最接近淡出结束，截短听感代价最小）。
    fn trim_excess_fading_voices(&mut self, max_voices: usize) {
        let limit = fading_retention_limit(max_voices);
        // 快速路径：buffer 未超过「active 上限 + 淡出保留上限」时无需扫描。
        if self.buffer.len() <= max_voices.saturating_add(limit) {
            return;
        }
        let mut kept = 0usize;
        self.buffer.retain(|group| {
            if group.voice.is_killed() {
                kept += 1;
                return kept <= limit;
            }
            true
        });
    }

    /// Releases the next voice, and all subsequent voices that have the same ID.
    pub fn release_next_voice(&mut self) -> Option<u8> {
        if !self.damper_held {
            let mut id: Option<usize> = None;
            let mut vel = None;

            // Find the first non releasing voice, get its id and release all voices with that id
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() {
                    continue;
                }

                if id.is_none() {
                    id = Some(voice.id);
                    vel = Some(voice.velocity())
                }

                if id != Some(voice.id) {
                    break;
                }

                voice.signal_release(ReleaseType::Standard);
            }

            vel
        } else {
            // Find the first non releasing voice which also isn't being held in the release buffer, and add it to the release buffer
            for voice in self.buffer.iter_mut() {
                if voice.is_releasing() {
                    continue;
                }

                if self.held_by_damper.contains(&voice.id) {
                    continue;
                }

                self.held_by_damper.push(voice.id);
                break;
            }

            None
        }
    }

    pub fn remove_ended_voices(&mut self) {
        let mut i = 0;
        while i < self.buffer.len() {
            if self.buffer[i].ended() {
                self.buffer.remove(i);
            } else {
                i += 1;
            }
        }
    }

    // pub fn iter_voices<'a>(&'a self) -> impl Iterator<Item = &Box<dyn Voice>> + 'a {
    //     self.buffer.iter().map(|group| &group.voice)
    // }

    pub fn iter_voices_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn Voice>> {
        self.buffer.iter_mut().map(|group| &mut group.voice)
    }

    pub fn has_voices(&self) -> bool {
        !self.buffer.is_empty()
    }

    pub fn voice_count(&self) -> usize {
        self.buffer.len()
    }

    pub fn set_damper(&mut self, damper: bool) {
        if self.damper_held && !damper {
            // Release all voices that are held by the damper
            for voice in self.buffer.iter_mut() {
                if self.held_by_damper.contains(&voice.id) {
                    voice.signal_release(ReleaseType::Standard);
                }
            }
            self.held_by_damper.clear();
        }
        self.damper_held = damper;
    }
}

#[cfg(test)]
mod tests {
    use super::{fading_retention_limit, ChannelInitOptions, VoiceBuffer};
    use crate::voice::{
        ReleaseType, Voice, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator,
    };

    /// 最小 Voice 实现：只关心偷声决策用到的 velocity / is_killed / ended。
    struct MockVoice {
        velocity: u8,
        killed: bool,
        ended: bool,
    }

    impl MockVoice {
        fn new(velocity: u8) -> Self {
            MockVoice {
                velocity,
                killed: false,
                ended: false,
            }
        }
    }

    impl VoiceGeneratorBase for MockVoice {
        fn ended(&self) -> bool {
            self.ended
        }

        fn signal_release(&mut self, rel_type: ReleaseType) {
            match rel_type {
                ReleaseType::Kill => self.killed = true,
                ReleaseType::Standard => self.ended = true,
            }
        }

        fn process_controls(&mut self, _control: &VoiceControlData) {}
    }

    impl VoiceSampleGenerator for MockVoice {
        fn render_to(&mut self, _buffer: &mut [f32]) {}
    }

    impl Voice for MockVoice {
        fn is_releasing(&self) -> bool {
            false
        }

        fn is_killed(&self) -> bool {
            self.killed
        }

        fn velocity(&self) -> u8 {
            self.velocity
        }

        fn exclusive_class(&self) -> Option<u8> {
            None
        }
    }

    fn push_one(buffer: &mut VoiceBuffer, velocity: u8, max_voices: usize) {
        buffer.push_voices(
            std::iter::once(Box::new(MockVoice::new(velocity)) as Box<dyn Voice>),
            Some(max_voices),
        );
    }

    #[test]
    fn fade_killing_retains_at_most_limit_killed_voices() {
        // 回归：高密度 NoteOn（模拟事件积压排空）下，被杀 voice 的保留量必须有界，
        // 否则每次 push 的全 buffer 扫描会退化为 O(n²)（死亡螺旋根因）。
        let mut buffer = VoiceBuffer::new(ChannelInitOptions {
            fade_out_killing: true,
        });
        let max_voices = 2usize;
        for i in 0..1000u32 {
            push_one(&mut buffer, (i % 127 + 1) as u8, max_voices);
            let limit = fading_retention_limit(max_voices);
            assert!(
                buffer.buffer.len() <= max_voices + limit,
                "第 {i} 次 push 后 buffer 无界增长: {}",
                buffer.buffer.len()
            );
        }
        assert!(
            buffer.get_active_count() <= max_voices,
            "活跃 voice 数必须受每键上限约束"
        );
    }

    #[test]
    fn fade_killing_keeps_fading_voices_under_normal_load() {
        // 正常负载（低于保留上限）下，被杀 voice 不被提前丢弃，淡出质量不受影响。
        let mut buffer = VoiceBuffer::new(ChannelInitOptions {
            fade_out_killing: true,
        });
        push_one(&mut buffer, 10, 2);
        push_one(&mut buffer, 20, 2);
        push_one(&mut buffer, 30, 2); // 触发一次偷声（最轻的 10 被淡出）
        assert_eq!(buffer.buffer.len(), 3, "被杀 voice 应保留在 buffer 中淡出");
        assert_eq!(buffer.get_active_count(), 2, "活跃 voice 应被限制为 2");
    }

    #[test]
    fn no_fade_out_drops_stolen_voices_immediately() {
        // fade_out_killing = false 时保持原语义：偷声立即出队，buffer 有界。
        let mut buffer = VoiceBuffer::new(ChannelInitOptions {
            fade_out_killing: false,
        });
        for _ in 0..100 {
            push_one(&mut buffer, 50, 2);
        }
        assert!(buffer.buffer.len() <= 2, "无淡出时 buffer 不得超过每键上限");
    }
}
