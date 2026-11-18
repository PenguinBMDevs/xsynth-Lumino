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
    let mut pending = 0.0f64;
    let t0 = Instant::now();
    let (mut blocks, mut events, mut audio_secs) = (0u64, 0u64, 0.0f64);

    println!("Rendering (max {max_seconds}s wall, layers={layers:?}, parallel={parallel})...");
    for batch in merged {
        if t0.elapsed().as_secs_f64() >= max_seconds {
            break;
        }
        // 先消化到本批事件的时间间隔（拉取式渲染，与 offline renderer 一致）。
        pending += batch.delta.max(0.0);
        while pending >= quantum_s {
            synth.read_samples(&mut buffer);
            pending -= quantum_s;
            audio_secs += quantum_s;
            blocks += 1;
            if t0.elapsed().as_secs_f64() >= max_seconds {
                break;
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
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    println!(
        "DONE wall={elapsed:.3}s audio={audio_secs:.2}s blocks={blocks} events={events} \
         realtime_x={:.2} voices={}",
        audio_secs / elapsed,
        synth.voice_count()
    );
    Ok(())
}
