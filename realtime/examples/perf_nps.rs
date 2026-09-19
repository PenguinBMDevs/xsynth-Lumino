//! 实时渲染性能对照探针：对比 max_nps 限流开/关与 Lumino app 配置下的实时渲染负载。
//!
//! 用法：perf_nps <midi> <sf2> [stock|nps0|app|fade|appnofade]
//! - stock: 引擎默认（max_nps=10_000，10ms 窗口，fade=false）——与 midi.rs 相同
//! - nps0:  仅关闭 NPS 限流（max_nps=0），其余默认
//! - app:   Lumino app 旧配置（max_nps=0、24ms 窗口、fade=true、每键上限 2）
//! - fade:  仅开启 fade_out_killing（验证淡出滞留 O(n²) 回归）
//! - appnofade: 当前 Lumino app 配置（max_nps=0、24ms、fade=false、每键上限 2）
//!
//! 配合 `--features tracy` 构建，并用 tracy-capture 抓取，可获得音频全链路火焰图。

use std::{
    thread,
    time::{Duration, Instant},
};

use midi_toolkit::{
    events::Event,
    io::MIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time},
        unwrap_items, TimeCaster,
    },
};
use xsynth_core::{
    channel::{
        ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ChannelInitOptions, ControlEvent,
    },
    soundfont::{SampleSoundfont, SoundfontBase},
};
use xsynth_realtime::{RealtimeSynth, SynthEvent, XSynthRealtimeConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<String>>();
    let (Some(midi), Some(sfz)) = (args.get(1).cloned(), args.get(2).cloned()) else {
        println!("Usage: perf_nps <midi> <sf2> [stock|nps0|app]");
        return Ok(());
    };
    let mode = args.get(3).cloned().unwrap_or_else(|| "stock".to_string());

    let config = match mode.as_str() {
        "nps0" => XSynthRealtimeConfig {
            max_nps: 0,
            ..Default::default()
        },
        "app" => XSynthRealtimeConfig {
            max_nps: 0,
            render_window_ms: 24.0,
            channel_init_options: ChannelInitOptions {
                fade_out_killing: true,
            },
            ..Default::default()
        },
        "fade" => XSynthRealtimeConfig {
            max_nps: 0,
            channel_init_options: ChannelInitOptions {
                fade_out_killing: true,
            },
            ..Default::default()
        },
        "appnofade" => XSynthRealtimeConfig {
            max_nps: 0,
            render_window_ms: 24.0,
            ..Default::default()
        },
        _ => XSynthRealtimeConfig::default(),
    };

    let synth = RealtimeSynth::open_with_default_output(config)?;
    let mut sender = synth.get_sender_ref().clone();

    let params = synth.stream_params();
    println!("mode={mode}");

    let soundfonts: Vec<std::sync::Arc<dyn SoundfontBase>> = vec![std::sync::Arc::new(
        SampleSoundfont::new(sfz, params, Default::default())?,
    )];
    sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetSoundfonts(soundfonts),
    )));
    if mode == "app" || mode == "appnofade" {
        sender.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
            ChannelConfigEvent::SetLayerCount(Some(2)),
        )));
    }

    let stats = synth.get_stats();
    let start = Instant::now();
    thread::spawn(move || {
        let mut sec = 0u64;
        let (mut sum_v, mut max_v, mut sum_r, mut max_r, mut min_b, mut n) =
            (0u64, 0u64, 0f64, 0f64, i64::MAX, 0u64);
        let mut underruns = 0u64;
        loop {
            thread::sleep(Duration::from_millis(50));
            let v = stats.voice_count();
            let b = stats.buffer().last_samples_after_read();
            let r = stats.buffer().average_renderer_load();
            sum_v += v;
            max_v = max_v.max(v);
            sum_r += r;
            max_r = max_r.max(r);
            min_b = min_b.min(b);
            if b < 0 {
                underruns += 1;
            }
            n += 1;
            let elapsed = start.elapsed().as_secs();
            if elapsed > sec {
                sec = elapsed;
                println!(
                    "[t={sec:3}s] V avg={:6} max={:6} | R avg={:5.2} max={:5.2} | B min={:6} | underruns={}",
                    sum_v / n.max(1),
                    max_v,
                    sum_r / n as f64,
                    max_r,
                    min_b,
                    underruns
                );
                sum_v = 0;
                max_v = 0;
                sum_r = 0.0;
                max_r = 0.0;
                min_b = i64::MAX;
                n = 0;
                underruns = 0;
            }
        }
    });

    let midi = MIDIFile::open(&midi, None).map_err(|e| format!("MIDI open failed: {e:?}"))?;
    let ppq = midi.ppq();
    let merged = pipe!(
        midi.iter_all_events_merged_batches()
        |>TimeCaster::<f64>::cast_event_delta()
        |>cancel_tempo_events(250000)
        |>scale_event_time(1.0 / ppq as f64)
        |>unwrap_items()
    );

    let (snd, rcv) = crossbeam_channel::bounded(100);
    thread::spawn(move || {
        for batch in merged {
            snd.send(batch).unwrap();
        }
    });

    let now = Instant::now();
    let mut time = 0.0;
    for batch in rcv {
        if batch.delta != 0.0 {
            time += batch.delta;
            let diff = time - now.elapsed().as_secs_f64();
            if diff > 0.0 {
                spin_sleep::sleep(Duration::from_secs_f64(diff));
            }
        }

        for e in batch.iter_inner() {
            match e {
                Event::NoteOn(e) => {
                    sender.send_event(SynthEvent::Channel(
                        e.channel as u32,
                        ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                            key: e.key,
                            vel: e.velocity,
                        }),
                    ));
                }
                Event::NoteOff(e) => {
                    sender.send_event(SynthEvent::Channel(
                        e.channel as u32,
                        ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key: e.key }),
                    ));
                }
                Event::ControlChange(e) => {
                    sender.send_event(SynthEvent::Channel(
                        e.channel as u32,
                        ChannelEvent::Audio(ChannelAudioEvent::Control(ControlEvent::Raw(
                            e.controller,
                            e.value,
                        ))),
                    ));
                }
                Event::PitchWheelChange(e) => {
                    sender.send_event(SynthEvent::Channel(
                        e.channel as u32,
                        ChannelEvent::Audio(ChannelAudioEvent::Control(
                            ControlEvent::PitchBendValue(e.pitch as f32 / 8192.0),
                        )),
                    ));
                }
                Event::ProgramChange(e) => {
                    sender.send_event(SynthEvent::Channel(
                        e.channel as u32,
                        ChannelEvent::Audio(ChannelAudioEvent::ProgramChange(e.program)),
                    ));
                }
                _ => {}
            }
        }
    }

    thread::sleep(Duration::from_secs(10000));
    Ok(())
}
