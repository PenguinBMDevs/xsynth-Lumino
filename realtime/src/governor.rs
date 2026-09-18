//! 实时声部治理：负载闭环 + 降级阶梯 + 看门狗。
//!
//! 设计目标（与 lumino 侧约定）：
//! - 用户只设一个**全局硬上限** `hard_max`（量程），运行目标由负载闭环决定；
//! - 软目标 `v_soft` 初始为 `ratio * hard_max`（ratio 默认 `1 - 1/e ≈ 0.632`，
//!   留出约 37% 暂态余量），负载高则收缩、负载低则缓慢恢复；
//! - 绝不进入"过载后无休止正反馈"：L2 起硬移除、L3 保命闸、L4 看门狗自愈。

/// 负载平滑系数（约 200ms 收敛）。
const EMA_ALPHA: f64 = 0.2;
/// 软目标触发收缩的负载阈值。
const HI: f64 = 0.75;
/// 负载低于该值才允许恢复软目标。
const LO: f64 = 0.50;
/// L2：重度治理（硬移除）。
const L2: f64 = 1.0;
/// L3：保命（可选软 NPS 闸 + 加速收缩）。
const L3: f64 = 1.5;
/// L4：看门狗触发所需的持续过载块数（≈1s @10ms 块）。
const WATCHDOG_BLOCKS: u64 = 100;
/// 看门狗每键保留的最新声部组数。
const WATCHDOG_KEEP_PER_KEY: usize = 16;
/// 软目标下限（防止收缩到 0 导致无输出）。
const MIN_SOFT: f64 = 128.0;
/// 单块最大收缩比例（超出 HI 的部分按此系数缩放，防止尖峰导致塌缩）。
const SHRINK_PER_BLOCK: f64 = 0.08;

/// 单块治理动作（由渲染管线执行）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GovernorAction {
    /// 软抢占数量（分级选择 + 1ms 淡出 + 死期限）。
    pub steal: usize,
    /// 硬抢占数量（跳过淡出，用于 L2 及以上）。
    pub hard_steal: usize,
    /// 看门狗自愈：每键保留的最新组数（`None` = 不触发）。
    pub watchdog_keep: Option<usize>,
    /// 软 NPS 闸是否启用（仅当外部允许且负载 > L2）。
    pub gate_active: bool,
    /// 软 NPS 闸速率（NoteOn/秒）。
    pub gate_rate: u64,
}

/// 负载闭环治理器。
#[derive(Debug, Clone)]
pub struct Governor {
    hard_max: usize,
    ratio: f64,
    v_soft: f64,
    load_ema: f64,
    /// 当前降级等级 0..=4。
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

    /// 当前软目标声部数（诊断用）。
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
        } else if self.load_ema > HI {
            1
        } else {
            0
        };

        // 软目标调节：高负载快收缩，低负载慢恢复（攻快放慢 + 滞回）。
        //
        // 收缩限幅（防自激）：只有"声部确实构成压力"（V 超过软目标一半）
        // 才收缩，且每块最多收缩 `SHRINK_PER_BLOCK`。否则初始化/事件洪峰
        // 造成的瞬时高负载会把软目标砸到地板，进而引发"每块都在抢占 →
        // 抢占成本又推高负载"的正反馈。
        let voice_pressure = total_voices as f64 > self.v_soft * 0.5;
        if self.load_ema > HI && voice_pressure {
            let excess = (self.load_ema - HI).min(1.0);
            self.v_soft = (self.v_soft * (1.0 - SHRINK_PER_BLOCK * excess)).max(MIN_SOFT);
        } else if self.load_ema < LO {
            self.v_soft = (self.v_soft * 1.02).min(self.ratio * self.hard_max as f64);
        }

        let deficit = (total_voices as f64 - self.v_soft).max(0.0);
        let (steal, hard_steal) = if self.load_ema > L2 {
            // L2 起：硬移除，立即降低实际渲染成本。
            (0, (deficit * 0.5).ceil() as usize)
        } else {
            ((deficit * 0.5).ceil() as usize, 0)
        };

        GovernorAction {
            steal,
            hard_steal,
            watchdog_keep: (self.level >= 4).then_some(WATCHDOG_KEEP_PER_KEY),
            gate_active: gate_enabled && self.load_ema > L2,
            gate_rate: (self.v_soft.max(MIN_SOFT) * 2.0) as u64,
        }
    }

    /// 看门狗触发后重置内部状态（回到干净基线，避免带着过载历史决策）。
    pub fn reset_after_watchdog(&mut self) {
        self.v_soft = self.ratio * self.hard_max as f64;
        self.load_ema = 0.0;
        self.level = 0;
        self.overload_blocks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn healthy_load_converges_to_ratio_of_hard_max() {
        let mut g = Governor::new(10_000, 1.0 - 1.0 / std::f64::consts::E);
        for _ in 0..1000 {
            let a = g.update(0.3, 500, false);
            assert_eq!(a.steal, 0);
            assert_eq!(a.hard_steal, 0);
            assert_eq!(g.level, 0);
        }
        assert!((g.v_soft() - 6320.0).abs() < 100.0, "v_soft={}", g.v_soft());
    }

    #[test]
    fn overload_shrinks_soft_target_and_escalates() {
        let mut g = Governor::new(10_000, 0.632);
        // 持续 1.2 倍负载：进入 L2，并持续收缩
        for _ in 0..50 {
            g.update(1.2, 9000, false);
        }
        assert!(g.level >= 2, "level={}", g.level);
        assert!(g.v_soft() < 6320.0);
        let a = g.update(1.2, 9000, false);
        assert!(a.hard_steal > 0, "L2 应硬移除");
        assert_eq!(a.steal, 0);
    }

    #[test]
    fn watchdog_triggers_after_sustained_overload_and_resets() {
        let mut g = Governor::new(10_000, 0.632);
        let mut saw_watchdog = false;
        // EMA 收敛（约 10 块）后再持续过载 WATCHDOG_BLOCKS 块。
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
    fn transient_spike_without_voice_pressure_does_not_shrink_soft_target() {
        let mut g = Governor::new(10_000, 0.632);
        // 初始化/事件洪峰：V=0 或远小于软目标时，即使负载爆表也不收缩。
        for _ in 0..50 {
            g.update(8.0, 0, false);
        }
        assert!(
            (g.v_soft() - 6320.0).abs() < 1.0,
            "空闲期不得收缩软目标: {}",
            g.v_soft()
        );
    }

    #[test]
    fn sustained_overload_shrinks_gradually_not_collapses() {
        let mut g = Governor::new(10_000, 0.632);
        // 单块 5.0 尖峰：收缩不得超过 SHRINK_PER_BLOCK。
        g.update(5.0, 9000, false);
        let after_one = g.v_soft();
        assert!(
            after_one >= 6320.0 * (1.0 - SHRINK_PER_BLOCK - 1e-9),
            "单块收缩超过限幅: {}",
            after_one
        );
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
