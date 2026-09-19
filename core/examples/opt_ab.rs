//! 优化 A/B 标定台：确定性负载 + 输出签名，用于性能与行为的前后对比。
//!
//! 用法：`cargo run -p xsynth-core --release --example opt_ab -- <sf2> [seconds]`
//!
//! 负载：单通道立体声、480 帧块、固定随机种子的事件流（密度模拟黑 MIDI），
//! 输出：渲染耗时、实时倍率、RMS/峰值/FNV 哈希（行为对比）。
//! 同一 commit 上多次运行结果确定性（事件流与块长固定）。

use std::{sync::Arc, time::Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use xsynth_core::{
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, VoiceChannel},
    soundfont::{SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let Some(sfz) = args
        .get(1)
        .cloned()
        .or_else(|| std::env::var("XSYNTH_EXAMPLE_SFZ").ok())
    else {
        println!("Usage: opt_ab <sfz> [seconds]");
        return;
    };
    let seconds: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);

    let stream_params = AudioStreamParams::new(48000, ChannelCount::Stereo);
    println!("Loading soundfont: {sfz}");
    let soundfonts: Vec<Arc<dyn SoundfontBase>> = vec![Arc::new(
        SampleSoundfont::new(&sfz, stream_params, Default::default()).unwrap(),
    )];

    let mut channel = VoiceChannel::new(Default::default(), stream_params, None);
    channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
        soundfonts,
    )));
    // `LAYERS=<n>` 复现 app 实时后端的每键上限（lumino `xsynth_max_voices_per_key` 默认 4）；
    // 不设置 = 不限制（历史基线口径，保持既有哈希基准可比）。
    let layers: Option<usize> = std::env::var("LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n > 0);
    channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
        layers,
    )));

    let mut rng: StdRng = SeedableRng::from_seed([7u8; 32]);
    let mut buffer = vec![0.0f32; 960];
    let blocks = (seconds as u64 * 48_000 / 480) as usize;

    // 声部保持：每块 8 个 NoteOn，NoteOff 在 400~600ms 后（活动声部约数百）。
    let mut pending: Vec<(usize, u8)> = Vec::new();

    let t0 = Instant::now();
    let mut energy = 0.0f64;
    let mut peak = 0.0f32;
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for block in 0..blocks {
        for _ in 0..8 {
            let key = rng.gen_range(21u8..=108);
            let vel = rng.gen_range(30u8..=127);
            channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn { key, vel }));
            pending.push((block + 40 + rng.gen_range(0usize..20), key));
        }
        pending.retain(|&(at, key)| {
            if at <= block {
                channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key }));
                false
            } else {
                true
            }
        });
        channel.read_samples(&mut buffer);
        for &s in &buffer {
            energy += (s as f64) * (s as f64);
            let a = s.abs();
            if a > peak {
                peak = a;
            }
            hash ^= s.to_bits() as u64;
            hash = hash.wrapping_mul(0x1000_0000_01b3);
        }
    }
    let elapsed = t0.elapsed().as_secs_f64();
    let audio_secs = blocks as f64 * 480.0 / 48_000.0;
    let samples = (blocks * buffer.len()) as f64;
    let rms = (energy / samples).sqrt();
    println!(
        "blocks={blocks} audio={audio_secs:.1}s render={elapsed:.3}s realtime_x{:.2} voices={} rms={rms:.6} peak={peak:.6} hash={hash:016x}",
        audio_secs / elapsed,
        channel.get_channel_stats().voice_count(),
    );
}
