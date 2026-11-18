//! 离线 tracy bench：读取统一黑 MIDI，灌入 ChannelGroup 全速渲染（无 pacing / 无 WAV 落盘），
//! 在 tracy capture 下定位 core 渲染热路径。
//!
//! 用法（release + tracy，先启动 tracy-capture 监听 8086）：
//!   cargo run --release -p xsynth-core --features tracy \
//!     --example tracy_bench -- <midi> <sf2> [max_seconds] [layers] [parallel]
//!
//! - max_seconds: wall-clock 运行时上限（默认 25s），避免 4400 万音符跑不完
//! - layers: 每键声部上限（默认 2，模拟 Lumino app）；0 = 不限制
//! - parallel: 0 = 单线程批渲染（默认）；1 = 通道+key 双线程池（app 实时配置）；
//!   2 = 仅通道池（每通道内可批渲染）；3 = 仅 key 池
//!
//! 固定音频窗口（A/B 用）：环境变量 `XSYNTH_BENCH_START`（秒）+ `XSYNTH_BENCH_AUDIO`（秒）。
//! 只对窗口内的渲染计时；窗口之前照常渲染以保持声部状态。统一黑 MIDI 前 65s 近乎静音，
//! wall 截止会让不同跑次落在不同音频位置，负载不可比——并行 A/B 必须走固定窗口。

use std::{sync::Arc, time::Instant};

use midi_toolkit::{
    events::{Event, MIDIEventEnum},
    io::MIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time},
        unwrap_items, TimeCaster,
    },
};
use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ControlEvent},
    channel_group::{
        ChannelGroup, ChannelGroupConfig, ParallelismOptions, SynthEvent, SynthFormat, ThreadCount,
    },
    soundfont::{SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<String>>();
    let Some(midi_path) = args.get(1) else {
        println!("Usage: tracy_bench <midi> <sf2> [max_seconds] [layers] [parallel]");
        return Ok(());
    };
    let sf2_path = args
        .get(2)
        .ok_or("Usage: tracy_bench <midi> <sf2> [max_seconds] [layers] [parallel]")?;
    let max_seconds: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(25.0);
    let layers: Option<usize> = match args.get(4).and_then(|s| s.parse::<usize>().ok()) {
        Some(0) | None => Some(2),
        Some(n) => Some(n),
    };
    let parallel: u32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    let sample_rate = 48000u32;
    let params = AudioStreamParams::new(sample_rate, ChannelCount::Stereo);

    println!("Loading soundfont: {sf2_path}");
    let soundfonts: Vec<Arc<dyn SoundfontBase>> = vec![Arc::new(SampleSoundfont::new(
        sf2_path,
        params,
        Default::default(),
    )?)];

    // parallel: 0=单线程（可批渲染） 1=通道+key 双池（app 配置） 2=仅通道池 3=仅 key 池
    let parallelism = match parallel {
        1 => ParallelismOptions::AUTO_PER_KEY,
        2 => ParallelismOptions::AUTO_PER_CHANNEL,
        3 => ParallelismOptions {
            channel: ThreadCount::None,
            key: ThreadCount::Auto,
        },
        _ => ParallelismOptions {
            channel: ThreadCount::None,
            key: ThreadCount::None,
        },
    };
    let config = ChannelGroupConfig {
        channel_init_options: Default::default(),
        format: SynthFormat::Midi,
        audio_params: params,
        parallelism,
    };
    let mut synth = ChannelGroup::new(config);
    synth.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetSoundfonts(soundfonts),
    )));
    synth.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetLayerCount(layers),
    )));

    println!("Opening MIDI: {midi_path}");
    let midi = MIDIFile::open(midi_path, None).map_err(|e| format!("MIDI open failed: {e:?}"))?;
    let ppq = midi.ppq();
    let merged = pipe!(
        midi.iter_all_track_events_merged_batches()
        |>TimeCaster::<f64>::cast_event_delta()
        |>cancel_tempo_events(250000)
        |>scale_event_time(1.0 / ppq as f64)
        |>unwrap_items()
    );

    // 渲染量子 10ms（480 帧立体声 = 960 samples），与实时块同量级。
    let quantum_s = 0.01f64;
    let quantum_samples = (sample_rate as f64 * quantum_s) as usize * 2;
    let mut buffer = vec![0.0f32; quantum_samples];

    // 固定音频窗口（A/B 口径）：从 `XSYNTH_BENCH_START` 秒起渲染 `XSYNTH_BENCH_AUDIO` 秒，
    // 只对窗口计时。原因：统一黑 MIDI 前 65s 近乎静音（~3k notes），wall 截止会让
    // 不同跑次停在不同音频位置（实测 events 从 9k 到 1.1M 波动 100 倍），负载不可比。
    // 窗口之前的音频照常渲染（保持声部状态），但不计入耗时。
    let start_audio: f64 = std::env::var("XSYNTH_BENCH_START")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let fixed_audio: Option<f64> = std::env::var("XSYNTH_BENCH_AUDIO")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|s| *s > 0.0);

    let mut pending = 0.0f64;
    let t0 = Instant::now();
    let (mut blocks, mut events, mut audio_secs) = (0u64, 0u64, 0.0f64);
    let mut sim_time = 0.0f64;
    let mut window_start: Option<Instant> = None;
    let (mut window_audio, mut window_blocks, mut window_events) = (0.0f64, 0u64, 0u64);
    let mut truncated = false;
    // 可选输出校验（`XSYNTH_BENCH_CHECKSUM=1`）：窗口内 RMS/峰值，用于跨并行配置
    // 对比「有没有丢/重声部」。混音求和顺序不同 → 不做逐位比较。
    let checksum_enabled = std::env::var_os("XSYNTH_BENCH_CHECKSUM").is_some();
    let (mut energy, mut peak) = (0.0f64, 0.0f32);

    println!(
        "Rendering (max {max_seconds}s wall, layers={layers:?}, parallel={parallel}, \
         window={start_audio:.1}s+{fixed_audio:?})..."
    );
    'outer: for batch in merged {
        if t0.elapsed().as_secs_f64() >= max_seconds {
            truncated = true;
            break;
        }
        // 先消化到本批事件的时间间隔（拉取式渲染，与 offline renderer 一致）。
        pending += batch.delta.max(0.0);
        while pending >= quantum_s {
            if fixed_audio.is_some() && window_start.is_none() && sim_time >= start_audio {
                window_start = Some(Instant::now());
            }
            synth.read_samples(&mut buffer);
            if checksum_enabled && window_start.is_some() {
                for &s in &buffer {
                    energy += (s as f64) * (s as f64);
                    let a = s.abs();
                    if a > peak {
                        peak = a;
                    }
                }
            }
            pending -= quantum_s;
            sim_time += quantum_s;
            audio_secs += quantum_s;
            blocks += 1;
            if window_start.is_some() {
                window_audio += quantum_s;
                window_blocks += 1;
            }
            if let Some(dur) = fixed_audio {
                if sim_time >= start_audio + dur {
                    break 'outer;
                }
            }
            if t0.elapsed().as_secs_f64() >= max_seconds {
                truncated = true;
                break 'outer;
            }
        }

        for e in batch.iter_events() {
            let (channel, event) = match e.as_event() {
                Event::NoteOn(e) => (
                    e.channel as u32,
                    ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                        key: e.key,
                        vel: e.velocity,
                    }),
                ),
                Event::NoteOff(e) => (
                    e.channel as u32,
                    ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key: e.key }),
                ),
                Event::ControlChange(e) => (
                    e.channel as u32,
                    ChannelEvent::Audio(ChannelAudioEvent::Control(ControlEvent::Raw(
                        e.controller,
                        e.value,
                    ))),
                ),
                Event::PitchWheelChange(e) => (
                    e.channel as u32,
                    ChannelEvent::Audio(ChannelAudioEvent::Control(ControlEvent::PitchBendValue(
                        e.pitch as f32 / 8192.0,
                    ))),
                ),
                Event::ProgramChange(e) => (
                    e.channel as u32,
                    ChannelEvent::Audio(ChannelAudioEvent::ProgramChange(e.program)),
                ),
                _ => continue,
            };
            synth.send_event(SynthEvent::Channel(channel, event));
            events += 1;
            if window_start.is_some() {
                window_events += 1;
            }
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    let (measured_audio, measured_elapsed) = match window_start {
        Some(t) => (window_audio, t.elapsed().as_secs_f64()),
        None => (audio_secs, elapsed),
    };
    println!(
        "DONE total_wall={elapsed:.3}s audio={audio_secs:.2}s blocks={blocks} events={events} \
         window_audio={measured_audio:.2}s window_blocks={window_blocks} \
         window_events={window_events} window_wall={measured_elapsed:.3}s \
         realtime_x={:.2} voices={} truncated={truncated}",
        measured_audio / measured_elapsed,
        synth.voice_count()
    );
    if checksum_enabled {
        let samples = window_blocks as f64 * quantum_samples as f64;
        println!(
            "CHECKSUM rms={:.9} peak={:.9}",
            (energy / samples.max(1.0)).sqrt(),
            peak
        );
    }
    Ok(())
}
