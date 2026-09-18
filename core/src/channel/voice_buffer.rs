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
    /// 被 Kill（短淡出）后强制移除的块序号；`None` 表示未被 Kill。
    /// 循环采样 release 后可能永远不报告 `ended()`，靠该期限兜底防滞留。
    pub kill_deadline: Option<u32>,
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

pub struct VoiceBuffer {
    options: ChannelInitOptions,
    id_counter: usize,
    buffer: VecDeque<GroupVoice>,
    damper_held: bool,
    held_by_damper: Vec<usize>,
    /// 渲染块序号（`remove_ended_voices` 每次调用自增），用于 Kill 死期限。
    block_index: u32,
}

/// Kill（1ms 淡出）后最多保留的渲染块数：到期强制移除，防止循环采样滞留。
const KILL_DEADLINE_BLOCKS: u32 = 2;

/// 抢占层级：T0 已杀 → T1 释放中最轻 → T2 最轻（并列取最老）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StealTier {
    /// 已被 Kill（已在淡出，听感代价最低）。
    Killed,
    /// 已进入 release 阶段的组中最轻者。
    Releasing,
    /// 全体最轻者（力度并列时取最老）。
    Quietest,
}

impl VoiceBuffer {
    pub fn new(options: ChannelInitOptions) -> Self {
        VoiceBuffer {
            options,
            id_counter: 0,
            buffer: VecDeque::new(),
            damper_held: false,
            held_by_damper: Vec::new(),
            block_index: 0,
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
        // 死期限：即使采样循环导致 `ended()` 永远为 false，也会在若干块后被强制移除。
        let deadline = self.block_index.saturating_add(KILL_DEADLINE_BLOCKS);
        self.buffer[index].kill_deadline = Some(deadline);
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
        for i in 0..self.buffer.len() {
            if self.buffer[i].exclusive_class() == Some(class) {
                self.kill_voice_fade_out(i);
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
            self.buffer.push_back(GroupVoice {
                id,
                voice,
                kill_deadline: None,
            });
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
        }
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
        self.block_index = self.block_index.saturating_add(1);
        let now = self.block_index;
        let mut i = 0;
        while i < self.buffer.len() {
            let deadline_expired = self.buffer[i]
                .kill_deadline
                .is_some_and(|deadline| now >= deadline);
            if self.buffer[i].ended() || deadline_expired {
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

    /// 当前活跃（未被 Kill）的声部组数量。
    pub fn active_voice_count(&self) -> usize {
        self.buffer.iter().filter(|g| !g.is_killed()).count()
    }

    /// 按分级策略抢占一组声部：短淡出（Kill）+ 死期限强制移除。
    ///
    /// 优先级（听感代价从低到高）：
    /// 1. **T0**：已被 Kill 的组（本身已在淡出/静音）；
    /// 2. **T1**：已进入 release 的组中最轻者（note-off 后正在衰减）；
    /// 3. **T2**：全体最轻者（力度并列时取最老——从队首迭代、严格小于保持首个）。
    ///
    /// 持续低音（长音、力度响）不会被"最老"直接命中：只有连轻音/衰减音
    /// 都不存在时才会轮到它们。
    pub fn steal_voice_group(&mut self) -> Option<StealTier> {
        if self.buffer.is_empty() {
            return None;
        }

        // T0：已 Kill
        if let Some(index) = self.buffer.iter().position(|g| g.is_killed()) {
            self.kill_voice_fade_out(index);
            return Some(StealTier::Killed);
        }

        // T1/T2：一次遍历同时找"释放中最轻"与"全体最轻"（并列取最老 = 先出现者）
        let mut releasing: Option<(usize, u8)> = None;
        let mut quietest: Option<(usize, u8)> = None;
        for (index, voice) in self.buffer.iter().enumerate() {
            let velocity = voice.velocity();
            if voice.is_releasing() && releasing.is_none_or(|(_, v)| velocity < v) {
                releasing = Some((index, velocity));
            }
            if quietest.is_none_or(|(_, v)| velocity < v) {
                quietest = Some((index, velocity));
            }
        }

        let (index, tier) = match releasing {
            Some((index, _)) => (index, StealTier::Releasing),
            None => {
                let (index, _) = quietest?;
                (index, StealTier::Quietest)
            }
        };
        self.kill_voice_fade_out(index);
        Some(tier)
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
