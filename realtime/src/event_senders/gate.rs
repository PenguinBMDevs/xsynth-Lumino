//! 软 NPS 闸：过载保命的全局令牌桶（默认关闭）。
//!
//! 与旧 NPS 限流的区别：
//! - 只在渲染管线判定"重度过载（L2+）"时**临时**启用，负载回落自动解除；
//! - 关闭时 `allow()` 只做一次原子读，恒返 `true`，不构成任何丢音路径；
//! - 启用时按令牌桶限速，溢出的 NoteOn 被跳过（计入 `skipped_notes`，
//!   对应 NoteOff 会被抵消，避免挂音）。
//!
//! 这是最后一道保命措施，非必要不应触发。

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

/// 令牌桶内部状态（毫令牌为单位的定点数，1 音符 = 1000 毫令牌）。
struct GateState {
    tokens_milli: i64,
    burst_milli: i64,
    last_ms: u64,
}

/// 全局软 NPS 闸。
pub struct EmergencyGate {
    active: AtomicBool,
    rate_per_sec: AtomicU64,
    state: Mutex<GateState>,
    epoch: Instant,
}

impl EmergencyGate {
    /// 创建未启用的闸门。
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            active: AtomicBool::new(false),
            rate_per_sec: AtomicU64::new(1),
            state: Mutex::new(GateState {
                tokens_milli: 0,
                burst_milli: 1000,
                last_ms: 0,
            }),
            epoch: Instant::now(),
        })
    }

    fn now_ms(&self) -> u64 {
        self.epoch.elapsed().as_millis() as u64
    }

    /// 由渲染管线每块调用：设置启用状态与限速（NoteOn/秒）。
    ///
    /// 从关闭切换到启用时重置令牌桶为"100ms 突发量"，避免携带旧的空桶。
    pub fn set(&self, active: bool, rate_per_sec: u64) {
        let rate = rate_per_sec.max(1);
        self.rate_per_sec.store(rate, Ordering::Relaxed);
        let was = self.active.swap(active, Ordering::Relaxed);
        if active && !was {
            let burst = (rate as i64 / 10).max(1) * 1000;
            let mut st = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            st.burst_milli = burst;
            st.tokens_milli = burst;
            st.last_ms = self.now_ms();
        }
    }

    /// 是否允许发送当前 NoteOn。未启用时恒为 `true`。
    pub fn allow(&self) -> bool {
        if !self.active.load(Ordering::Relaxed) {
            return true;
        }
        let now = self.now_ms();
        let rate = self.rate_per_sec.load(Ordering::Relaxed) as i64;
        let mut st = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let elapsed = now.saturating_sub(st.last_ms) as i64;
        if elapsed > 0 {
            st.tokens_milli = (st.tokens_milli + rate * elapsed).min(st.burst_milli);
            st.last_ms = now;
        }
        if st.tokens_milli >= 1000 {
            st.tokens_milli -= 1000;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_gate_never_blocks() {
        let gate = EmergencyGate::new();
        for _ in 0..10_000 {
            assert!(gate.allow());
        }
    }

    #[test]
    fn active_gate_throttles_to_burst_then_rate() {
        let gate = EmergencyGate::new();
        gate.set(true, 1_000_000); // 极高限速：突发内全部放行
        let mut allowed = 0;
        for _ in 0..100 {
            if gate.allow() {
                allowed += 1;
            }
        }
        assert!(allowed > 0);
        // 极低限速（1/s）且耗尽突发后，应拒绝
        gate.set(false, 1);
        gate.set(true, 1);
        let mut rejected = false;
        for _ in 0..10_000 {
            if !gate.allow() {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "低限速下应出现丢弃");
    }

    #[test]
    fn disabling_gate_restores_passthrough() {
        let gate = EmergencyGate::new();
        gate.set(true, 1);
        gate.set(false, 1);
        for _ in 0..1000 {
            assert!(gate.allow());
        }
    }
}
