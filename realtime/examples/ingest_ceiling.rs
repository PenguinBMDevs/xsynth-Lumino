//! 事件接入上限实测台：生产者（播放线程侧）→ 每通道**无界队列** → 通道消费。
//!
//! 目的：把"接入上限 ≈ DRAIN_CAP × 2 × 块率 × 通道数"从纸上推算变成实测数字，
//! 并让"无界队列到底积了多少"可见（`event_queue_high_water`）。
//!
//! 关键设计：**不加载音色库**——NoteOn 不产生声部，负载恒 ~0，
//! 因此测得的纯粹是接入/消费吞吐，与 DSP 成本无关。
//! （代价：负载为 0 时不会进入紧急模式，故 `dropped` 预期为 0；
//!   带声部的洪峰与丢音观测需要真实音色库，见 `voice_load -- stream`。）
//!
//! 用法：
//! ```text
//! cargo run --release -p xsynth-realtime --example ingest_ceiling -- [事件总数] [通道数]
//! ```

use std::time::{Duration, Instant};

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelEvent},
    channel_group::SynthFormat,
};
use xsynth_realtime::{RealtimeSynth, SynthEvent, ThreadCount, XSynthRealtimeConfig};

const DEFAULT_TOTAL: u64 = 1_000_000;
const DEFAULT_CHANNELS: u32 = 16;

fn main() {
    let total: u64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOTAL);
    let channels: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CHANNELS);

    let cfg = XSynthRealtimeConfig {
        render_window_ms: 10.0,
        cushion_ms: 100.0,
        multithreading: ThreadCount::None,
        max_nps: 0,
        format: SynthFormat::Custom { channels },
        ..Default::default()
    };

    let synth = RealtimeSynth::open_with_default_output(cfg).expect("open realtime synth");
    let stats = synth.get_stats();
    let mut sender = synth.get_sender_ref().clone();

    // 静置一拍，确认管线已经在按实时节奏出块。
    std::thread::sleep(Duration::from_millis(200));
    let idle_depth = stats.event_queue_depth();

    println!(
        "ingest_ceiling: 事件={total} 通道={channels} 块=10ms DRAIN_CAP=256/admit（2 admit/块）"
    );
    println!("空闲时队列深度 = {idle_depth}（应为 0）");

    // 生产者：尽可能快地灌入（无界队列 → 永不阻塞，这正是要观测的风险）。
    let t0 = Instant::now();
    for i in 0..total {
        let ch = (i % channels as u64) as u32;
        sender.send_event(SynthEvent::Channel(
            ch,
            ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                key: 30 + (i % 60) as u8,
                vel: 1,
            }),
        ));
    }
    let inject = t0.elapsed();

    // 排空：轮询队列深度，直到它回落到接近 0（或超时）。
    let mut peak_seen = 0i64;
    let mut drained_at = None;
    while t0.elapsed() < Duration::from_secs(120) {
        let d = stats.event_queue_depth();
        peak_seen = peak_seen.max(d);
        if d <= 16 {
            drained_at = Some(t0.elapsed());
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let elapsed = drained_at.unwrap_or_else(|| t0.elapsed());
    let high_water = stats.event_queue_high_water();
    let dropped = stats.emergency_dropped_notes();

    println!(
        "生产者：{total} 事件 / {:.3}s = {:.0} 事件/秒（send 侧能力，非瓶颈）",
        inject.as_secs_f64(),
        total as f64 / inject.as_secs_f64().max(1e-9)
    );
    println!(
        "排空：{:.3}s（含注入期）→ 实测接入上限 ≈ {:.0} 事件/秒",
        elapsed.as_secs_f64(),
        total as f64 / elapsed.as_secs_f64().max(1e-9)
    );
    println!(
        "队列高水位 = {high_water} 事件（跨通道最大；轮询观测到的峰值 = {peak_seen}）"
    );
    let ev_size = std::mem::size_of::<ChannelEvent>();
    println!(
        "单事件 {} 字节 → 单通道队列高水位占用 ≈ {:.2} MB；若各区同时积压，总量 ≈ {:.2} MB",
        ev_size,
        high_water as f64 * ev_size as f64 / 1e6,
        high_water as f64 * ev_size as f64 * channels as f64 / 1e6
    );
    println!(
        "理论值：256 × 2 admit × 100 块/秒 × {channels} 通道 = {} 事件/秒",
        256 * 2 * 100 * channels
    );
    println!("紧急丢弃 = {dropped}（无音色库 → 负载≈0 → 预期 0）");

    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Audio(
        ChannelAudioEvent::AllNotesKilled,
    )));
    std::thread::sleep(Duration::from_millis(100));
}
