//! Tracy 性能剖析插桩（`tracy` feature）。
//!
//! 未启用 `tracy` feature 时宏展开为空操作，零开销；
//! 启用后若 Tracy 客户端未启动（离线渲染等场景），`Client::running()` 返回 `None`，
//! 同样不发射任何数据，不会 panic。
//!
//! 粒度约定：zone 只落在「每渲染块 / 每通道 / 每管线阶段」级别，
//! 不在每键（128 键 × 16 通道 × 每块）级别发射——后者约 36 万 zone/s，
//! 会造成 trace 体积失控并干扰被测时序。

/// 在 `$body` 执行期间发射一个 Tracy zone（作用域结束时自动结束）。
///
/// 用法：`crate::profiling::tracy_zone!("name", { ... })`
#[cfg(feature = "tracy")]
macro_rules! tracy_zone {
    ($name:literal, $body:block) => {{
        let _tracy_zone = tracy_client::Client::running()
            .map(|client| client.span(tracy_client::span_location!($name), 0));
        $body
    }};
}

/// 未启用 `tracy` feature 时的空操作版本（不产生任何代码）。
#[cfg(not(feature = "tracy"))]
macro_rules! tracy_zone {
    ($name:literal, $body:block) => {{
        $body
    }};
}

pub(crate) use tracy_zone;

/// 发射一条 Tracy 数值曲线点（用于 A/B 的工作量归一化基准，如每块活跃声部数）。
///
/// 用法：`crate::profiling::tracy_plot!("ch_voices", count as f64)`
///
/// 为什么需要：跨 run 的 A/B 里「同一段音频的渲染内容」会因过载漂移而不可比
/// （播放位置与渲染进度脱钩），用每块声部数（工作量）做基准即可得到与内容无关的
/// ns/voice-block 口径，避免拿被负载差异污染的总量比值当结论。
#[cfg(feature = "tracy")]
macro_rules! tracy_plot {
    ($name:literal, $value:expr) => {{
        if let Some(client) = tracy_client::Client::running() {
            client.plot(tracy_client::plot_name!($name), $value as f64);
        }
    }};
}

/// 未启用 `tracy` feature 时的空操作版本（零开销）。
#[cfg(not(feature = "tracy"))]
macro_rules! tracy_plot {
    ($name:literal, $value:expr) => {{
        let _ = $value;
    }};
}

pub(crate) use tracy_plot;
