//! XSynth 无设备 CPU 成本标定台。
//!
//! 读取 Python 预处理的事件流（8 字节/事件），按 10ms 块渲染并逐块计时：
//!   load_rt = 块渲染耗时 / 块音频时长
//! 同时记录块内活跃声部数与 NoteOn 数，供离线拟合
//! `load = c0 + c_v * voices + c_e * nps`。
//!
//! 用法:
//!   cargo run --release -p xsynth-core --example load_profile -- \
//!       <soundfont> <events.bin> <out.csv> [block_frames]

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;
use std::time::Instant;

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ChannelInitOptions},
    channel_group::{
        ChannelGroup, ChannelGroupConfig, ParallelismOptions, SynthEvent, SynthFormat, ThreadCount,
    },
    soundfont::{SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

const SR: u32 = 96_000;
const DEFAULT_BLOCK: usize = 960; // 10ms

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: load_profile <soundfont> <events.bin> <out.csv> [block_frames]");
        std::process::exit(2);
    }
    let sf2 = args[1].clone();
    let ev_path = args[2].clone();
    let out_path = args[3].clone();
    let block: usize = args
        .get(4)
        .map(|s| s.parse().expect("block_frames"))
        .unwrap_or(DEFAULT_BLOCK);

    let params = AudioStreamParams::new(SR, ChannelCount::Stereo);
    eprintln!("[bench] loading soundfont ...");
    let t_load = Instant::now();
    let sf = SampleSoundfont::new(sf2, params, Default::default())?;
    eprintln!(
        "[bench] soundfont loaded in {:.1}s",
        t_load.elapsed().as_secs_f64()
    );

    let config = ChannelGroupConfig {
        // 与实时路径一致：不淡出；每键不限（全局行为由上层治理，标定时不限）。
        channel_init_options: ChannelInitOptions {
            fade_out_killing: false,
            max_voices: None,
        },
        format: SynthFormat::Midi,
        audio_params: params,
        parallelism: ParallelismOptions {
            channel: if std::env::var_os("BENCH_SERIAL").is_some() {
                ThreadCount::None
            } else {
                ThreadCount::Auto
            },
            key: if std::env::var_os("BENCH_SERIAL").is_some() {
                ThreadCount::None
            } else {
                ThreadCount::Auto
            },
        },
    };
    let mut group = ChannelGroup::new(config);
    group.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetSoundfonts(vec![Arc::new(sf) as Arc<dyn SoundfontBase>]),
    )));
    group.send_event(SynthEvent::AllChannels(ChannelEvent::Config(
        ChannelConfigEvent::SetLayerCount(None),
    )));

    let mut data = Vec::new();
    BufReader::new(File::open(&ev_path)?).read_to_end(&mut data)?;
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
    let last_sample = events.last().map(|e| e.0).unwrap_or(0);
    eprintln!(
        "[bench] events: {} last_sample={}",
        events.len(),
        last_sample
    );

    let mut out = BufWriter::new(File::create(&out_path)?);
    writeln!(out, "block,sample,voices,on_count,load_rt")?;

    let mut buf = vec![0.0f32; block * 2];
    let mut idx = 0usize;
    let mut sample: u64 = 0;
    let mut block_no: u64 = 0;
    let block_dur = block as f64 / SR as f64;

    while idx < events.len() {
        let block_end = sample + block as u64;
        let t = Instant::now();
        let mut on_count: u32 = 0;
        while idx < events.len() && events[idx].0 < block_end {
            let (_, kind, ch, key, vel) = events[idx];
            let ev = match kind {
                0 => {
                    on_count += 1;
                    ChannelAudioEvent::NoteOn { key, vel }
                }
                1 => ChannelAudioEvent::NoteOff { key },
                2 => ChannelAudioEvent::ProgramChange(key),
                _ => {
                    idx += 1;
                    continue;
                }
            };
            group.send_event(SynthEvent::Channel(ch as u32, ChannelEvent::Audio(ev)));
            idx += 1;
        }
        group.read_samples(&mut buf);
        let elapsed = t.elapsed().as_secs_f64();
        writeln!(
            out,
            "{block_no},{sample},{},{on_count},{:.5}",
            group.voice_count(),
            elapsed / block_dur
        )?;
        sample = block_end;
        block_no += 1;
    }

    eprintln!("[bench] done: {block_no} blocks -> {out_path}");
    Ok(())
}
