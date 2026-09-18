//! 实时声部治理：静态软目标 + 重载保护（硬移除）+ 看门狗。
//!
//! 设计与教训（均来自 96kHz 重型黑乐谱实测）：
//! - 用户设**全局硬上限** `hard_max`（量程），运行目标固定为
//!   `v_soft = ratio × hard_max`（ratio 默认 `1 - 1/e ≈ 0.632`）。
//!   **目标不随负载漂移**：负载反馈会与抢占成本/管线调度互相激励，
//!   实测会把软目标砸到地板并引发自激卡顿；负载只用于诊断、看门狗与保命闸。
//! - 注入端由通道级**入场控制**兜底（见 `realtime_synth`）：超目标即推迟新
//!   NoteOn，从结构上封死"事件积压 → 全量注入 → 雪崩"的回路。
//! - 超目标时**硬移除**（从最老声部开始），单块移除量有界。
//! - L4 看门狗：持续过载 1s → 每键仅保留最新 N 组 + 重置基线（自愈）。

/// 负载平滑系数（约 200ms 收敛；仅用于诊断与看门狗，不参与目标调节）。
const EMA_ALPHA: f64 = 0.2;
/// L2：重度治理（供保命闸启用判断）。
const L2: f64 = 1.0;
/// L3：持续过载判定。
const L3: f64 = 1.5;
/// L4：看门狗触发所需的持续过载块数（≈1s @10ms 块）。
const WATCHDOG_BLOCKS: u64 = 100;
/// 看门狗每键保留的最新声部组数。
const WATCHDOG_KEEP_PER_KEY: usize = 16;
/// 软目标下限（防止小量程下目标为 0）。
const MIN_SOFT: f64 = 128.0;

/// 单块治理动作（由渲染管线执行）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorAction {
    /// 软抢占数量（保留字段；当前策略不使用）。
    pub steal: usize,
    /// 硬抢占数量（跳过淡出，立即移除）。
    pub hard_steal: usize,
    /// 看门狗自愈：每键保留的最新组数（`None` = 不触发）。
    pub watchdog_keep: Option<usize>,
    /// 软 NPS 闸是否启用（仅当外部允许且负载 > L2）。
    pub gate_active: bool,
    /// 软 NPS 闸速率（NoteOn/秒）。
    pub gate_rate: u64,
}

/// 声部治理器。
#[derive(Debug, Clone)]
pub struct Governor {
    hard_max: usize,
    ratio: f64,
    v_soft: f64,
    load_ema: f64,
    /// 当前降级等级 0..=4（诊断）。
    pub level: u8,
    overload_blocks: u64,
}

impl Governor {
    /// `hard_max`：全局硬上限（量程）；`ratio`：软目标比例（0.5~0.9）。
    pub fn new(hard_max: usize, ratio: f64) -> Self {
        let hard_max = hard_max.max(MIN_SOFT as usize);
        let ratio = ratio.clamp(0.3, 0.95);
        Self {
            hard_max,
            ratio,
            v_soft: ratio * hard_max as f64,
            load_ema: 0.0,
            level: 0,
            overload_blocks: 0,
        }
    }

    /// 当前软目标声部数（入场控制与诊断用）。
    pub fn v_soft(&self) -> f64 {
        self.v_soft
    }

    /// 平滑负载（诊断用）。
    pub fn load_ema(&self) -> f64 {
        self.load_ema
    }

    /// 每块调用一次：`load` = 本块渲染耗时 / 块时长，`total_voices` = 当前总声部数。
    ///
    /// `gate_enabled` 为 false 时永远不会启用软 NPS 闸（默认关闭）。
    pub fn update(&mut self, load: f64, total_voices: u64, gate_enabled: bool) -> GovernorAction {
        self.load_ema = EMA_ALPHA * load + (1.0 - EMA_ALPHA) * self.load_ema;

        if self.load_ema > L3 {
            self.overload_blocks += 1;
        } else {
            self.overload_blocks = 0;
        }

        self.level = if self.overload_blocks >= WATCHDOG_BLOCKS {
            4
        } else if self.load_ema > L3 {
            3
        } else if self.load_ema > L2 {
            2
        } else {
            0
        };

        // 目标固定，不随负载漂移（防自激）。
        self.v_soft = self.ratio * self.hard_max as f64;

        // 超目标即硬移除：单块最多移除 1/4 总声部（收敛快且工作量有界）。
        let deficit = (total_voices as f64 - self.v_soft).max(0.0);
        let hard_steal = if deficit > 0.0 {
            deficit.min((total_voices as f64 / 4.0).max(64.0)).ceil() as usize
        } else {
            0
        };

        GovernorAction {
            steal: 0,
            hard_steal,
            watchdog_keep: (self.level >= 4).then_some(WATCHDOG_KEEP_PER_KEY),
            gate_active: gate_enabled && self.load_ema > L2,
            gate_rate: (self.v_soft.max(MIN_SOFT) * 2.0) as u64,
        }
    }

    /// 看门狗触发后重置诊断状态（软目标本身固定，无需恢复）。
    pub fn reset_after_watchdog(&mut self) {
        self.load_ema = 0.0;
        self.level = 0;
        self.overload_blocks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_fixed_at_ratio_of_hard_max() {
        let mut g = Governor::new(10_000, 1.0 - 1.0 / std::f64::consts::E);
        assert!((g.v_soft() - 6320.0).abs() < 2.0);
        // 连续过载也不得漂移（防自激的关键回归）。
        for _ in 0..1000 {
            g.update(8.0, 50_000, false);
        }
        assert!((g.v_soft() - 6320.0).abs() < 2.0, "v_soft={}", g.v_soft());
    }

    #[test]
    fn idle_load_keeps_target() {
        let mut g = Governor::new(10_000, 0.632);
        for _ in 0..1000 {
            g.update(0.0, 0, false);
        }
        assert!((g.v_soft() - 6320.0).abs() < 1.0);
        assert_eq!(g.level, 0);
    }

    #[test]
    fn over_target_hard_steals_bounded() {
        let mut g = Governor::new(10_000, 0.632);
        let a = g.update(0.3, 9_000, false);
        // deficit=2680，单块上限 max(V/4,64)=2250
        assert_eq!(a.hard_steal, 2250);
        assert_eq!(a.steal, 0);
        // 目标以内不抢
        let b = g.update(0.3, 6_000, false);
        assert_eq!(b.hard_steal, 0);
    }

    #[test]
    fn watchdog_triggers_after_sustained_overload_and_resets() {
        let mut g = Governor::new(10_000, 0.632);
        let mut saw_watchdog = false;
        for _ in 0..WATCHDOG_BLOCKS + 40 {
            let a = g.update(1.8, 20_000, true);
            if a.watchdog_keep == Some(WATCHDOG_KEEP_PER_KEY) {
                saw_watchdog = true;
                g.reset_after_watchdog();
                break;
            }
        }
        assert!(saw_watchdog, "持续过载应触发看门狗");
        assert_eq!(g.level, 0);
    }

    #[test]
    fn gate_only_when_enabled_and_overloaded() {
        let mut g = Governor::new(10_000, 0.632);
        for _ in 0..20 {
            g.update(1.8, 20_000, false);
        }
        assert!(
            !g.update(1.8, 20_000, false).gate_active,
            "未开启时不得启用软 NPS 闸"
        );
        let mut g2 = Governor::new(10_000, 0.632);
        for _ in 0..20 {
            g2.update(1.8, 20_000, true);
        }
        let a2 = g2.update(1.8, 20_000, true);
        assert!(a2.gate_active);
        assert!(a2.gate_rate > 0);
    }
}
