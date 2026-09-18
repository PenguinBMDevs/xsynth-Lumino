//! 实时路径声部成本实测（headless，不依赖 lumino）：
//! 用真实的 RealtimeSynth 管线把通道 0 铺到目标声部数，采样 `average_renderer_load`。
//!
//! 用法：
//!   cargo +nightly run --release -p xsynth-realtime --example voice_load -- <auto|none|manual:N> <target_voices>
//!
//! 输出 `per_voice_us`：每个声部每 10ms 块占用的墙钟微秒数（越小越快）。

use std::{env, sync::Arc, thread, time::Duration};

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent},
    soundfont::{SampleSoundfont, SoundfontBase},
};
use xsynth_realtime::{RealtimeSynth, SynthEvent, ThreadCount, XSynthRealtimeConfig};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "auto".to_string());
    let target: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4000);
    if mode == "stream" {
        run_stream(&args);
        return;
    }

    let mut cfg = XSynthRealtimeConfig::default();
    cfg.render_window_ms = 10.0;
    cfg.cushion_ms = 100.0;
    cfg.multithreading = match mode.as_str() {
        "none" => ThreadCount::None,
        m if m.starts_with("manual:") => {
            ThreadCount::Manual(m["manual:".len()..].parse::<usize>().unwrap_or(4))
        }
        _ => ThreadCount::Auto,
    };
    cfg.channel_init_options.fade_out_killing = false;
    cfg.channel_init_options.max_voices = None;
    // 关闭治理/限流，纯测渲染成本。
    cfg.global_max_voices = Some(200_000);
    cfg.voice_target_ratio = 1.0;
    cfg.max_nps = 0;

    let mut synth = RealtimeSynth::open_with_default_output(cfg).expect("open realtime synth");
    let params = synth.stream_params();
    eprintln!(
        "[realtime-bench] mode={} sr={} target={}",
        mode, params.sample_rate, target
    );

    let sf = SampleSoundfont::new(
        "D:/Soundfont/TWGMD Remake VI.sf2",
        params,
        Default::default(),
    )
    .expect("load soundfont");
    let mut sender = synth.get_sender_ref().clone();
    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetSoundfonts(vec![Arc::new(sf) as Arc<dyn SoundfontBase>]),
    )));
    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetLayerCount(None),
    )));
    thread::sleep(Duration::from_millis(800));

    let stats = synth.get_stats();

    // 铺声部：轮换键位，避免单键叠加（贴近真实分布）。
    let mut sent = 0u64;
    while stats.voice_count() < target && sent < target * 8 {
        let key = 30u8 + (sent % 60) as u8;
        sender.send_event(SynthEvent::Channel(
            0,
            ChannelEvent::Audio(ChannelAudioEvent::NoteOn { key, vel: 1 }),
        ));
        sent += 1;
        if sent % 64 == 0 {
            thread::sleep(Duration::from_millis(2));
        }
    }

    thread::sleep(Duration::from_millis(1500));
    let mut loads = Vec::new();
    for _ in 0..20 {
        loads.push(stats.buffer().average_renderer_load());
        thread::sleep(Duration::from_millis(50));
    }
    let voices = stats.voice_count();
    let avg = loads.iter().sum::<f64>() / loads.len() as f64;
    let max = loads.iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "[realtime-bench] mode={} voices={} load_avg={:.3} load_max={:.3} per_voice_us={:.2}",
        mode,
        voices,
        avg,
        max,
        avg * 10_000.0 / (voices.max(1) as f64)
    );

    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Audio(
        ChannelAudioEvent::AllNotesKilled,
    )));
    thread::sleep(Duration::from_millis(300));
}

/// 事件流模式：按采样时间实时喂入 events.bin（8 字节/事件），复现实时播放负载。
fn run_stream(args: &[String]) {
    use std::fs;

    let par = args.get(2).cloned().unwrap_or_else(|| "auto".to_string());
    let ev_path = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "events.bin".to_string());
    let fade = std::env::var_os("FADE").is_some();

    let mut cfg = XSynthRealtimeConfig::default();
    cfg.render_window_ms = 10.0;
    cfg.cushion_ms = 100.0;
    cfg.multithreading = if par == "none" {
        ThreadCount::None
    } else {
        ThreadCount::Auto
    };
    cfg.channel_init_options.fade_out_killing = fade;
    cfg.channel_init_options.max_voices = None;
    // 可选：CAP=N 复现"撞上限"（预算=软目标−当前声部数，超限抢旧不丢新）。
    let cap: Option<usize> = std::env::var("CAP").ok().and_then(|s| s.parse().ok());
    cfg.global_max_voices = Some(cap.unwrap_or(200_000));
    cfg.voice_target_ratio = 1.0;
    cfg.soft_nps_gate = std::env::var_os("GATE").is_some();
    cfg.max_nps = 0;

    let mut synth = RealtimeSynth::open_with_default_output(cfg).expect("open realtime synth");
    let params = synth.stream_params();
    eprintln!(
        "[stream-bench] par={} fade={} sr={} events={}",
        par, fade, params.sample_rate, ev_path
    );

    let sf = SampleSoundfont::new(
        "D:/Soundfont/TWGMD Remake VI.sf2",
        params,
        Default::default(),
    )
    .expect("load soundfont");
    let mut sender = synth.get_sender_ref().clone();
    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetSoundfonts(vec![Arc::new(sf) as Arc<dyn SoundfontBase>]),
    )));
    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetLayerCount(None),
    )));
    thread::sleep(Duration::from_millis(800));

    let data = fs::read(&ev_path).expect("read events");
    let events: Vec<(u64, u8, u8, u8, u8)> = data
        .chunks_exact(8)
        .map(|c| {
            (
                u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64,
                c[4],
                c[5],
                c[6],
                c[7],
            )
        })
        .collect();
    let stats = synth.get_stats();
    let start = std::time::Instant::now();
    let sr = params.sample_rate as f64;
    let mut idx = 0usize;
    let mut last_report = std::time::Instant::now();
    let mut peak_load: f64 = 0.0;
    let mut sum_load = 0.0;
    let mut n_samples = 0u32;

    while idx < events.len() {
        let elapsed_samples = start.elapsed().as_secs_f64() * sr;
        while idx < events.len() && (events[idx].0 as f64) <= elapsed_samples {
            let (_, kind, ch, key, vel) = events[idx];
            let ev = match kind {
                0 => ChannelAudioEvent::NoteOn { key, vel },
                1 => ChannelAudioEvent::NoteOff { key },
                _ => ChannelAudioEvent::ProgramChange(key),
            };
            sender.send_event(SynthEvent::Channel(ch as u32, ChannelEvent::Audio(ev)));
            idx += 1;
        }
        let load = stats.buffer().average_renderer_load();
        peak_load = peak_load.max(load);
        sum_load += load;
        n_samples += 1;
        if last_report.elapsed() >= Duration::from_secs(2) {
            eprintln!(
                "[stream-bench] t={:.1}s sent={} voices={} load={:.3}",
                start.elapsed().as_secs_f64(),
                idx,
                stats.voice_count(),
                load
            );
            last_report = std::time::Instant::now();
        }
        thread::sleep(Duration::from_micros(500));
    }
    println!(
        "[stream-bench] par={} fade={} sent={} load_avg={:.3} load_peak={:.3}",
        par,
        fade,
        idx,
        sum_load / (n_samples.max(1) as f64),
        peak_load
    );
}
