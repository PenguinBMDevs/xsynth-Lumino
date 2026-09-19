//! 长音标定台：**按住不放**的一个音符，逐段时间查看输出电平与声部存活。
//!
//! 用法：
//! - 单键：`cargo run -p xsynth-core --release --example held_note -- <sf2> [key] [vel] [seconds]`
//! - 扫描：`cargo run -p xsynth-core --release --example held_note -- <sf2> sweep [seconds]`
//! - 复现：`cargo run -p xsynth-core --release --example held_note -- <sf2> strike [key] [seconds]`
//!   （app 配置：每键上限 4 + `fade_out_killing=false`，同键反复触发 →
//!   统计**块边界采样级硬切**事件数；修复前每次超限抢占都会硬切一次）
//!
//! 与 `opt_ab` 的区别：`opt_ab` 在 400~600ms 后 NoteOff（对「按住不放」类缺陷不敏感），
//! 本工具**只发 NoteOn**，因此能回答：
//! - 长音在什么时刻被硬切（声部归零 / 输出恰为 0，且之前明显非零）？
//! - 切口处还剩多少电平（相对峰值的 dB）—— 决定听感上是「自然衰减完」还是「啪一下没了」。
//!
//! 生产路径：`VoiceChannel`（含批渲染路径），与 app 完全一致的渲染入口。
//!
//! `sweep` 模式对 21..=108 逐键做上述测量并输出「提前硬切」清单：判断某个音源里
//! 哪些音色的长音会被提前杀掉（切口电平仍高 = 可闻硬切）。

use std::sync::Arc;
use std::time::Instant;

use xsynth_core::{
    channel::{
        ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, ChannelInitOptions, VoiceChannel,
    },
    soundfont::{SampleSoundfont, SoundfontBase},
    AudioPipe, AudioStreamParams, ChannelCount,
};

const BLOCK_FRAMES: usize = 480;
const SAMPLE_RATE: u32 = 48_000;

fn build_channel(path: &str) -> VoiceChannel {
    let stream_params = AudioStreamParams::new(SAMPLE_RATE, ChannelCount::Stereo);
    let soundfonts: Vec<Arc<dyn SoundfontBase>> = vec![Arc::new(
        SampleSoundfont::new(path, stream_params, Default::default()).expect("加载音源失败"),
    )];
    let mut channel = VoiceChannel::new(Default::default(), stream_params, None);
    channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
        soundfonts,
    )));
    channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
        None,
    )));
    channel
}

struct HeldResult {
    /// 声部归零的块序号（`None` = 全程存活）。
    reaped_at: Option<usize>,
    /// 声部归零前最后 50ms 的 RMS 与全程峰值 RMS（dB 差 = 切口可闻度）。
    rms_before_cut: f64,
    peak_rms: f64,
    voices_end: usize,
}

/// 按住 `key` 渲染 `seconds`，返回声部存活情况。
fn measure_held(channel: &mut VoiceChannel, key: u8, vel: u8, seconds: u32) -> HeldResult {
    let blocks = (seconds as usize * SAMPLE_RATE as usize) / BLOCK_FRAMES;
    let mut buffer = vec![0.0f32; BLOCK_FRAMES * 2];
    channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn { key, vel }));

    // 50ms 滑动窗的电平历史（用于取"归零前"的电平）。
    let mut window_energy = 0.0f64;
    let mut recent_rms: Vec<f64> = Vec::new();
    let mut peak_rms = 0.0f64;
    let mut reaped_at = None;
    let mut voices_end = 0usize;

    for block in 0..blocks {
        buffer.fill(0.0);
        channel.read_samples(&mut buffer);
        let energy: f64 = buffer.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        window_energy += energy;

        if (block + 1) % 5 == 0 {
            let rms = (window_energy / (5 * BLOCK_FRAMES * 2) as f64).sqrt();
            recent_rms.push(rms);
            if rms > peak_rms {
                peak_rms = rms;
            }
            window_energy = 0.0;
        }

        voices_end = channel.get_channel_stats().voice_count() as usize;
        if voices_end == 0 && reaped_at.is_none() {
            reaped_at = Some(block);
        }
    }
    // 声部归零前最后一个 50ms 窗口的电平。
    let rms_before_cut = match reaped_at {
        Some(block) => {
            let cut_window = block / 5;
            recent_rms
                .get(cut_window.saturating_sub(2))
                .copied()
                .unwrap_or(0.0)
        }
        None => 0.0,
    };

    // 清场：硬杀 + 一段静音渲染，避免影响下一个键的测量。
    channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::AllNotesKilled));
    for _ in 0..24 {
        buffer.fill(0.0);
        channel.read_samples(&mut buffer);
    }

    HeldResult {
        reaped_at,
        rms_before_cut,
        peak_rms,
        voices_end,
    }
}

fn db(ratio: f64) -> f64 {
    if ratio <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * ratio.log10()
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        println!("Usage: held_note <sf2> [key|sweep] [vel|seconds] [seconds]");
        return;
    };
    let mode = args.get(2).cloned().unwrap_or_else(|| "60".to_string());

    if mode == "strike" {
        // 复现 app 实时后端配置：每键上限 4（`xsynth_max_voices_per_key` 默认值）
        // + `fade_out_killing = false`（api/xsynth.rs 显式关闭）。
        let key: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(60);
        let seconds: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);
        let strike_interval_ms = 50usize;

        println!("Loading soundfont: {path}");
        let stream_params = AudioStreamParams::new(SAMPLE_RATE, ChannelCount::Stereo);
        let soundfonts: Vec<Arc<dyn SoundfontBase>> = vec![Arc::new(
            SampleSoundfont::new(&path, stream_params, Default::default()).expect("加载音源失败"),
        )];
        let mut channel = VoiceChannel::new(
            ChannelInitOptions {
                fade_out_killing: false,
                max_voices: None,
            },
            stream_params,
            None,
        );
        channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetSoundfonts(
            soundfonts,
        )));
        channel.process_event(ChannelEvent::Config(ChannelConfigEvent::SetLayerCount(
            Some(4),
        )));

        // 第一个音符按住不放（长音），随后每 50ms 同键再触发。
        channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
            key,
            vel: 100,
        }));

        let blocks = (seconds as usize * SAMPLE_RATE as usize) / BLOCK_FRAMES;
        let strike_period = (strike_interval_ms * SAMPLE_RATE as usize / 1000) / BLOCK_FRAMES;
        let mut buffer = vec![0.0f32; BLOCK_FRAMES * 2];
        let mut prev_last = (0.0f32, 0.0f32);
        let mut boundary_steps: Vec<(f64, f32)> = Vec::new();

        for block in 0..blocks {
            if strike_period > 0 && block > 0 && block % strike_period == 0 {
                channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                    key,
                    vel: 100,
                }));
            }
            buffer.fill(0.0);
            channel.read_samples(&mut buffer);

            // 块边界跳变（同声道）：抢占发生在块间，硬切首先出现在这里。
            let step = (buffer[0] - prev_last.0)
                .abs()
                .max((buffer[1] - prev_last.1).abs());
            boundary_steps.push((
                block as f64 * BLOCK_FRAMES as f64 / SAMPLE_RATE as f64,
                step,
            ));
            let frames = buffer.len() / 2;
            prev_last = (buffer[(frames - 1) * 2], buffer[(frames - 1) * 2 + 1]);
        }

        let mut sorted: Vec<f32> = boundary_steps.iter().map(|(_, s)| *s).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("非 NaN"));
        let median = sorted[sorted.len() / 2];
        let max = *sorted.last().expect("非空");
        // 硬切判定：边界跳变超过中位数的 5 倍且绝对值 > 1e-3。
        let hard_cuts: Vec<&(f64, f32)> = boundary_steps
            .iter()
            .filter(|(_, s)| *s > median * 5.0 && *s > 1e-3)
            .collect();
        println!(
            "key={key} seconds={seconds} 同键每 {strike_interval_ms}ms 复击；块数={blocks}\n\
             边界跳变：median={median:.6} max={max:.6}\n\
             硬切事件（>5×median 且 >1e-3）：{} 次",
            hard_cuts.len()
        );
        for (t, s) in hard_cuts.iter().take(10) {
            println!("  t={t:.3}s step={s:.6}");
        }
        return;
    }

    if mode == "sweep" {
        let seconds: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
        println!("Loading soundfont: {path}");
        let t0 = Instant::now();
        let mut channel = build_channel(&path);
        println!(
            "加载耗时 {:.2}s；逐键按住 {seconds}s（vel=100），只列「声部被判结束」的键：",
            t0.elapsed().as_secs_f64()
        );
        println!("key  reaped_at(s)  rms_before_cut  peak_rms  cut_level(dB rel peak)");
        let mut cut_count = 0;
        for key in 21u8..=108 {
            let r = measure_held(&mut channel, key, 100, seconds);
            if let Some(block) = r.reaped_at {
                cut_count += 1;
                let t = block as f64 * BLOCK_FRAMES as f64 / SAMPLE_RATE as f64;
                let level = db(r.rms_before_cut / r.peak_rms);
                println!(
                    "{key:>3}  {t:>10.3}  {:>13.8}  {:>8.6}  {level:>8.1}",
                    r.rms_before_cut, r.peak_rms
                );
            }
        }
        println!(
            "共 {} 个键在 {seconds}s 内声部被判结束（其余键全程存活）",
            cut_count
        );
        return;
    }

    let key: u8 = mode.parse().unwrap_or(60);
    let vel: u8 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);
    let seconds: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8);

    println!("Loading soundfont: {path}");
    let mut channel = build_channel(&path);
    let r = measure_held(&mut channel, key, vel, seconds);
    println!("key={key} vel={vel} seconds={seconds}");
    match r.reaped_at {
        Some(block) => {
            let t = block as f64 * BLOCK_FRAMES as f64 / SAMPLE_RATE as f64;
            println!(
                "声部在第 {block} 块（t={t:.3}s）被判结束 —— 切口电平 {:.8}（相对峰值 {:.1} dB），\
                 峰值 RMS {:.6}",
                r.rms_before_cut,
                db(r.rms_before_cut / r.peak_rms.max(f64::MIN_POSITIVE)),
                r.peak_rms
            );
        }
        None => println!(
            "全程存活（{seconds}s 内声部未归零）；峰值 RMS {:.6}",
            r.peak_rms
        ),
    }
    let _ = r.voices_end;
}
