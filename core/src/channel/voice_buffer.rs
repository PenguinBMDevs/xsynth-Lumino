use super::ChannelInitOptions;
use crate::voice::{ReleaseType, Voice};
use std::{
    collections::VecDeque,
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
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
    /// 权威声部计数（与 `buffer.len()` 严格同步维护）。
    ///
    /// 由本缓冲区在所有增删点直接更新；治理器/入场控制读它做实时决策，
    /// 不再依赖"渲染后按差值对账"的滞后统计（那种统计在长时间硬抢占下会失真）。
    voice_counter: Arc<AtomicU64>,
}

/// Kill（1ms 淡出）后最多保留的渲染块数：到期强制移除，防止循环采样滞留。
const KILL_DEADLINE_BLOCKS: u32 = 2;

/// 抢占层级：T1 释放中最轻 → T2 最轻（并列取最老）。
///
/// 只从"未 Kill"的活跃组中选择：已 Kill 的组由死期限负责移除，再次"抢占"
/// 它们既不会降低活跃数（会把治理计数带偏），也没有听感收益。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StealTier {
    /// 已进入 release 阶段的组中最轻者。
    Releasing,
    /// 全体最轻者（力度并列时取最老）。
    Quietest,
}

impl VoiceBuffer {
    pub fn new(voice_counter: Arc<AtomicU64>, options: ChannelInitOptions) -> Self {
        VoiceBuffer {
            options,
            id_counter: 0,
            buffer: VecDeque::new(),
            damper_held: false,
            held_by_damper: Vec::new(),
            block_index: 0,
            voice_counter,
        }
    }

    /// 同步权威计数（所有增删点必须调用）。
    fn adjust_counter(&self, delta: isize) {
        if delta >= 0 {
            self.voice_counter
                .fetch_add(delta as u64, Ordering::Relaxed);
        } else {
            self.voice_counter
                .fetch_sub((-delta) as u64, Ordering::Relaxed);
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
                self.adjust_counter(-(count as isize));
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
            let removed = self.buffer.len();
            self.buffer.clear();
            self.adjust_counter(-(removed as isize));
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
        self.adjust_counter(len as isize);

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
        let mut removed = 0isize;
        let mut i = 0;
        while i < self.buffer.len() {
            let deadline_expired = self.buffer[i]
                .kill_deadline
                .is_some_and(|deadline| now >= deadline);
            if self.buffer[i].ended() || deadline_expired {
                self.buffer.remove(i);
                removed += 1;
            } else {
                i += 1;
            }
        }
        self.adjust_counter(-removed);
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

    /// 按分级策略抢占一组**活跃**（未 Kill）声部：短淡出（Kill）+ 死期限强制移除。
    ///
    /// 优先级（听感代价从低到高）：
    /// 1. **T1**：已进入 release 的组中最轻者（note-off 后正在衰减）；
    /// 2. **T2**：全体最轻者（力度并列时取最老——从队首迭代、严格小于保持首个）。
    ///
    /// 已 Kill 的组会被跳过：它们已计入"非活跃"，再次 kill 不会降低活跃数，
    /// 反而会让治理计数虚降（此前实测导致活跃声部无界增长）；它们的移除由
    /// 死期限在 `remove_ended_voices` 中完成。
    ///
    /// 自然保护：持续低音（长音、力度响、未释放）不会被优先命中。
    pub fn steal_voice_group(&mut self) -> Option<StealTier> {
        if self.buffer.is_empty() {
            return None;
        }

        // T1/T2：一次遍历同时找"释放中最轻"与"全体最轻"（并列取最老 = 先出现者）
        // 保护刚触发的音符：跳过最新一组（队尾，id 最大）。
        let newest_id = self.buffer.back().map(|g| g.id);
        let mut releasing: Option<(usize, u8)> = None;
        let mut quietest: Option<(usize, u8)> = None;
        for (index, voice) in self.buffer.iter().enumerate() {
            if voice.is_killed() || Some(voice.id) == newest_id {
                continue;
            }
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

    /// 硬移除最老的一组声部（L2 重度治理：跳过淡出，立即释放）。
    pub fn hard_steal_oldest(&mut self) -> bool {
        let popped = self.buffer.pop_front().is_some();
        if popped {
            self.adjust_counter(-1);
        }
        popped
    }

    /// 看门狗：每键仅保留最新 `keep` 组声部，其余立即移除（L4 自愈）。
    pub fn trim_to_newest(&mut self, keep: usize) {
        let mut removed = 0isize;
        while self.buffer.len() > keep {
            self.buffer.pop_front();
            removed += 1;
        }
        self.adjust_counter(-removed);
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
