//! 长音审计：统计 SF2 里「按住的 Sustain 循环」区域，并量化会被硬切的时间点。
//!
//! 用法：`cargo run -p xsynth-soundfonts --release --example loop_audit -- <sf2> [sample_rate]`
//!
//! # 背景（回归存档）
//!
//! `SampleReaderLoopSustain::is_past_end` 曾在 `2d8a15a` 的「SF2 语义优化」里被改写成
//! 符号翻转的代数形式（`pos - last - offset >= len` → `pos >= len - (last + offset)`）。
//! `last` 是未回绕的读取位置，未释放时随 `pos` 一起增长，于是阈值单调下降：
//! **按住不放的 sustain 长音会在播放到 `sample_end / 2` 时被判越界**，被
//! `VoiceBuffer::remove_ended_voices` 直接移除（硬切，无 release 尾巴）。
//!
//! 本工具回答「这个缺陷覆盖了给定音源的哪些预设/键区、会在第几秒被切」——
//! 修复前后都可用（修复后本工具仍有价值：它是「哪些音色依赖 sustain 循环」的清单）。
//!
//! 注：`sample_end` 已由加载器转换到输出采样率（`convert_sample_index`），
//! 因此硬切时间 = `sample_end / 2 / sample_rate`。

use std::collections::BTreeSet;

use xsynth_soundfonts::{
    sf2::{load_soundfont, Sf2Preset},
    LoopMode,
};

const DEFAULT_SAMPLE_RATE: u32 = 48_000;

struct PresetRow {
    bank: u16,
    preset: u16,
    regions: usize,
    keys: String,
    cut_mean: f64,
    cut_min: f64,
    cut_max: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(path) = args.get(1).cloned() else {
        println!("Usage: loop_audit <sf2> [sample_rate]");
        return;
    };
    let sample_rate: u32 = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE_RATE);

    println!("Parsing: {path} (sample_rate={sample_rate})");
    let presets = match load_soundfont(&path, sample_rate) {
        Ok(presets) => presets,
        Err(e) => {
            println!("解析失败: {e}");
            return;
        }
    };

    let (mut total, mut sustain, mut continuous, mut no_loop, mut one_shot) = (0usize, 0, 0, 0, 0);
    let mut rows = Vec::<PresetRow>::new();
    let mut all_cuts = Vec::<f64>::new();
    // exclusive class 统计：`Some(0)` 是危险信号（SF2 里 0 = 无独占类，
    // 加载器若记为 Some(0)，则每次 NoteOn 都会杀掉同类所有声部 = 长音必死）。
    let mut excl_none = 0usize;
    let mut excl_zero = 0usize;
    let mut excl_nonzero = 0usize;
    let mut excl_values = BTreeSet::<u8>::new();

    for preset in &presets {
        let preset = preset as &Sf2Preset;
        let mut cuts = Vec::new();
        let mut keys = BTreeSet::new();
        for region in &preset.regions {
            total += 1;
            match region.exclusive_class {
                None => excl_none += 1,
                Some(0) => {
                    excl_zero += 1;
                    excl_values.insert(0);
                }
                Some(v) => {
                    excl_nonzero += 1;
                    excl_values.insert(v);
                }
            }
            match region.loop_mode {
                LoopMode::LoopSustain => {
                    sustain += 1;
                    cuts.push(region.sample_end as f64 / sample_rate as f64);
                    keys.extend(region.keyrange.clone());
                }
                LoopMode::LoopContinuous => continuous += 1,
                LoopMode::NoLoop => no_loop += 1,
                LoopMode::OneShot => one_shot += 1,
            }
        }
        if cuts.is_empty() {
            continue;
        }
        all_cuts.extend(cuts.iter().copied());
        rows.push(PresetRow {
            bank: preset.bank,
            preset: preset.preset,
            regions: cuts.len(),
            keys: format!(
                "{}-{}",
                keys.iter().next().copied().unwrap_or(0),
                keys.iter().next_back().copied().unwrap_or(0)
            ),
            cut_mean: mean(&cuts),
            cut_min: cuts.iter().copied().fold(f64::MAX, f64::min),
            cut_max: cuts.iter().copied().fold(f64::MIN, f64::max),
        });
    }

    if total == 0 {
        println!("没有解析出任何区域");
        return;
    }

    println!(
        "\nexclusive_class: None={excl_none} Some(0)={excl_zero} Some(非0)={excl_nonzero} \
         取值集合={:?}",
        excl_values
    );
    if excl_zero > 0 {
        println!(
            "⚠️  警告：{excl_zero} 个区域带 `exclusive_class = Some(0)`。SF2 规范中 0 = 无独占类，\
             但引擎会在**每次 NoteOn** 时杀掉同类（class 0）的全部声部 → 长音会被后续音符直接杀掉。"
        );
    }

    println!(
        "\n区域总数 {total}: LoopSustain={sustain} ({:.1}%)  LoopContinuous={continuous}  \
         NoLoop={no_loop}  OneShot={one_shot}",
        sustain as f64 * 100.0 / total as f64
    );
    if !all_cuts.is_empty() {
        let min = all_cuts.iter().copied().fold(f64::MAX, f64::min);
        let max = all_cuts.iter().copied().fold(f64::MIN, f64::max);
        println!(
            "受影响的 Sustain 区域 {}/{} ({:.1}%)：按住不放时将被硬切于 {:.2}s ~ {:.2}s \
             （均值 {:.2}s）",
            all_cuts.len(),
            total,
            all_cuts.len() as f64 * 100.0 / total as f64,
            min / 2.0,
            max / 2.0,
            mean(&all_cuts) / 2.0
        );
    }

    rows.sort_by_key(|row| std::cmp::Reverse(row.regions));
    println!(
        "\n┌────────┬──────────────┬──────────┬───────────────────────┐\n\
         │ bank/program │ Sustain 区域 │  键区    │ 按住硬切时间(均值/最短/最长) │\n\
         ├────────┼──────────────┼──────────┼───────────────────────┤"
    );
    for row in rows.iter().take(25) {
        println!(
            "│  {:>3}/{:>3}   │     {:>4}     │ {:>8} │ {:>6.2}s / {:>5.2}s / {:>5.2}s │",
            row.bank, row.preset, row.regions, row.keys, row.cut_mean, row.cut_min, row.cut_max
        );
    }
    println!("└────────┴──────────────┴──────────┴───────────────────────┘");
    if rows.len() > 25 {
        println!(
            "（共 {} 个预设含 Sustain 区域，仅显示前 25 个）",
            rows.len()
        );
    }
}
