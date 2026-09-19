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

    /// 抢占同键上「最不重要」的一组声部：**优先已进入 release 的组**（正在衰减的
    /// 尾巴），其次最轻者（并列取最老——从队首迭代、严格小于保持首个）。
    ///
    /// 两条硬约束（修复「长音被直接杀掉」）：
    ///
    /// 1. **永远走 1ms 淡出**（`kill_voice_fade_out`），不做零淡出硬切。
    ///    历史实现按 `fade_out_killing` 分支，关闭时用 `buffer.drain()` 立即出队：
    ///    那是采样级不连续 —— 既是可闻 click，也正是用户报告的「长音无法完整播放、
    ///    音符被直接杀掉」（同键第 N+1 次触发把仍在响的持音瞬间抹掉）。
    ///    淡出后的组由 `remove_ended_voices`（同块末）或 `KILL_DEADLINE_BLOCKS`
    ///    死期限清理，保留量由 `trim_excess_fading_voices` 兜底。
    /// 2. **保护持音**：SF2 sustain 循环、延音踏板下的长音都是「未释放」；
    ///    优先抢 release 中的尾巴 → 重复音符杀掉的总是前一次的余音，
    ///    只有在同键所有组都未释放时才会命中持音（复音上限仍然严格维持）。
    fn pop_quietest_voice_group(&mut self, ignored_id: usize) {
        if self.buffer.is_empty() {
            return;
        }

        // 一次遍历选出：T1 = release 中的最轻组；T2 = 全体最轻组（并列取最老）。
        let mut releasing: Option<(usize, usize, u8)> = None; // (index, id, vel)
        let mut quietest: Option<(usize, usize, u8)> = None;
        for (i, voice) in self.buffer.iter().enumerate() {
            if voice.id == ignored_id || voice.is_killed() {
                continue;
            }
            let vel = voice.velocity();
            if voice.is_releasing() && releasing.is_none_or(|(_, _, v)| vel < v) {
                releasing = Some((i, voice.id, vel));
            }
            if quietest.is_none_or(|(_, _, v)| vel < v) {
                quietest = Some((i, voice.id, vel));
            }
        }

        let Some((victim_index, victim_id, _)) = releasing.or(quietest) else {
            return;
        };

        // 组大小 = 该 id 在选中下标处连续出现的个数（同一次 NoteOn 的声部共用 id）。
        let mut count = 0usize;
        while victim_index + count < self.buffer.len()
            && self.buffer[victim_index + count].id == victim_id
        {
            count += 1;
        }

        for i in victim_index..(victim_index + count) {
            self.kill_voice_fade_out(i);
        }

        if let Some(index) = self.held_by_damper.iter().position(|&x| x == victim_id) {
            self.held_by_damper.remove(index);
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
        self.buffer
            .iter()
            .filter(|group| !(**group).is_killed())
            .count()
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
            } else {
                // 上限按**活跃**（未被 Kill）组计，而不是 buffer 长度：被 Kill 的组正在
                // 1ms 淡出，若按 buffer 长度计会导致每次 push 都叠加额外抢占
                // （淡出保留 → buffer 变长 → 再抢一个 → 再淡出 ……）。
                while self.get_active_count() > max_voices {
                    self.pop_quietest_voice_group(id);
                }
            }

            // 淡出保留上限：防止被杀 voice 在事件积压时无界累积，导致
            // `get_active_count` / `pop_quietest_voice_group` 的全 buffer 扫描
            // 在单个渲染块内退化为 O(n²)。每键上限的抢占现在**始终**走淡出，
            // 因此这一步必须无条件执行（不再依赖 `fade_out_killing`）。
            self.trim_excess_fading_voices(max_voices);
        }
    }

    /// 将被杀（淡出中）voice 的数量裁剪到 `fading_retention_limit` 以内。
    ///
    /// 丢弃顺序：**从最老（队首）开始**——它们最接近淡出结束（1ms 淡出 ≈ 块内即完成），
    /// 硬截的听感代价最小。历史实现在此处保留了最老的、丢弃最新的，等于把刚开始
    /// 淡出（电平仍高）的 voice 直接抹掉 —— 又一次采样级不连续。
    fn trim_excess_fading_voices(&mut self, max_voices: usize) {
        let limit = fading_retention_limit(max_voices);
        // 快速路径：buffer 未超过「active 上限 + 淡出保留上限」时无需扫描。
        if self.buffer.len() <= max_voices.saturating_add(limit) {
            return;
        }
        let killed = self
            .buffer
            .iter()
            .filter(|group| group.voice.is_killed())
            .count();
        let mut to_drop = killed.saturating_sub(limit);
        let mut removed = 0isize;
        if to_drop > 0 {
            self.buffer.retain(|group| {
                if to_drop > 0 && group.voice.is_killed() {
                    to_drop -= 1;
                    removed += 1;
                    return false;
                }
                true
            });
        }
        // 权威计数不变量：物理移除必须同步 `voice_counter`
        // （rpn 的治理器/入场控制直接读它，漏计会让声部总数永久虚高）。
        self.adjust_counter(-removed);
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
        // 单趟重建：避免 `VecDeque::remove(i)` 在长缓冲上反复搬移导致 O(n^2)。
        // 保留原有顺序（最老在前），硬抢占/看门狗语义不变。
        let old_len = self.buffer.len();
        let mut alive = VecDeque::with_capacity(old_len);
        for group in self.buffer.drain(..) {
            let deadline_expired = group.kill_deadline.is_some_and(|deadline| now >= deadline);
            if !(group.ended() || deadline_expired) {
                alive.push_back(group);
            }
        }
        let removed = old_len - alive.len();
        self.buffer = alive;
        self.adjust_counter(-(removed as isize));
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

#[cfg(test)]
mod tests {
    use std::sync::{atomic::AtomicU64, Arc};

    use super::{fading_retention_limit, ChannelInitOptions, VoiceBuffer};
    use crate::voice::{
        ReleaseType, Voice, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator,
    };

    /// 最小 Voice 实现：只关心偷声决策用到的 velocity / is_killed / is_releasing / ended。
    struct MockVoice {
        velocity: u8,
        killed: bool,
        releasing: bool,
        ended: bool,
    }

    impl MockVoice {
        fn new(velocity: u8) -> Self {
            MockVoice {
                velocity,
                killed: false,
                releasing: false,
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
            self.releasing
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
        let counter = Arc::new(AtomicU64::new(0));
        let mut buffer = VoiceBuffer::new(
            counter.clone(),
            ChannelInitOptions {
                fade_out_killing: true,
                max_voices: None,
            },
        );
        let max_voices = 2usize;
        for i in 0..1000u32 {
            push_one(&mut buffer, (i % 127 + 1) as u8, max_voices);
            let limit = fading_retention_limit(max_voices);
            assert!(
                buffer.buffer.len() <= max_voices + limit,
                "第 {i} 次 push 后 buffer 无界增长: {}",
                buffer.buffer.len()
            );
            // 权威计数不变量：裁剪淡出 voice 必须同步 voice_counter，
            // 否则治理器读到的声部总数会永久虚高（合并回归）。
            assert_eq!(
                counter.load(std::sync::atomic::Ordering::Relaxed),
                buffer.buffer.len() as u64,
                "第 {i} 次 push 后权威计数与 buffer 失同步"
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
        let mut buffer = VoiceBuffer::new(
            Arc::new(AtomicU64::new(0)),
            ChannelInitOptions {
                fade_out_killing: true,
                max_voices: None,
            },
        );
        push_one(&mut buffer, 10, 2);
        push_one(&mut buffer, 20, 2);
        push_one(&mut buffer, 30, 2); // 触发一次偷声（最轻的 10 被淡出）
        assert_eq!(buffer.buffer.len(), 3, "被杀 voice 应保留在 buffer 中淡出");
        assert_eq!(buffer.get_active_count(), 2, "活跃 voice 应被限制为 2");
    }

    #[test]
    fn layer_cap_steal_is_faded_even_without_fade_out_killing() {
        // 回归（用户症状「长音无法完整播放、音符被直接杀掉」）：
        // `fade_out_killing = false` 时，每键上限抢占**也必须**走 1ms 淡出。
        // 历史实现走 `buffer.drain()` 零淡出硬切 = 采样级不连续（click）。
        let counter = Arc::new(AtomicU64::new(0));
        let mut buffer = VoiceBuffer::new(
            counter.clone(),
            ChannelInitOptions {
                fade_out_killing: false,
                max_voices: None,
            },
        );
        let max_voices = 2usize;
        for _ in 0..100 {
            push_one(&mut buffer, 50, max_voices);
        }

        assert_eq!(
            buffer.get_active_count(),
            2,
            "活跃声部数必须严格受每键上限约束"
        );
        let killed = buffer.buffer.iter().filter(|g| g.voice.is_killed()).count();
        assert!(
            killed >= 1,
            "被抢占的 voice 应停留在 Kill 淡出状态（不是被静默抹掉）"
        );
        assert!(
            killed <= fading_retention_limit(max_voices),
            "淡出保留量必须有界（O(n²) 防护）: killed={killed}"
        );
        assert!(
            buffer.buffer.len() <= max_voices + fading_retention_limit(max_voices),
            "淡出保留量必须有界（O(n²) 防护）: {}",
            buffer.buffer.len()
        );
        assert_eq!(
            counter.load(std::sync::atomic::Ordering::Relaxed),
            buffer.buffer.len() as u64,
            "权威计数必须与 buffer 严格同步"
        );
    }

    #[test]
    fn layer_cap_steal_protects_held_notes_and_takes_releasing_tail() {
        // 保护持音：同键重复触发时，优先抢「已在 release 的尾巴」，
        // 不能因为「最轻」而先把按住不放的长音（低力度）杀掉。
        let mut buffer = VoiceBuffer::new(
            Arc::new(AtomicU64::new(0)),
            ChannelInitOptions {
                fade_out_killing: false,
                max_voices: None,
            },
        );
        buffer.push_voices(
            std::iter::once(Box::new(MockVoice::new(10)) as Box<dyn Voice>),
            Some(2),
        );
        let mut tail = MockVoice::new(120);
        tail.releasing = true;
        buffer.push_voices(std::iter::once(Box::new(tail) as Box<dyn Voice>), Some(2));
        // 第三次触发：活跃 3 > 上限 2 → 抢占。应抢 release 中的尾巴（120），保留持音（10）。
        buffer.push_voices(
            std::iter::once(Box::new(MockVoice::new(100)) as Box<dyn Voice>),
            Some(2),
        );

        let killed_velocities: Vec<u8> = buffer
            .buffer
            .iter()
            .filter(|g| g.voice.is_killed())
            .map(|g| g.voice.velocity())
            .collect();
        assert_eq!(
            killed_velocities,
            vec![120],
            "必须优先抢占 release 中的尾巴，而不是最轻的持音"
        );
        assert_eq!(buffer.get_active_count(), 2);
    }
}
