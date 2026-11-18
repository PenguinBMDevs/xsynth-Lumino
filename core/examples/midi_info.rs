//! 诊断工具：统计 MIDI 的音符密度随播放时间分布，用于给并行 bench 选固定区间。
//!
//! 用法：`cargo run --release -p xsynth-core --example midi_info -- <midi> [bucket_secs] [max_secs]`

use midi_toolkit::{
    events::{Event, MIDIEventEnum},
    io::MIDIFile,
    pipe,
    sequence::{
        event::{cancel_tempo_events, scale_event_time},
        unwrap_items, TimeCaster,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<String>>();
    let midi_path = args.get(1).cloned().expect("midi path required");
    let bucket: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5.0);
    let max_secs: f64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(180.0);

    let midi = MIDIFile::open(&midi_path, None).map_err(|e| format!("{e:?}"))?;
    let ppq = midi.ppq();
    let merged = pipe!(
        midi.iter_all_track_events_merged_batches()
        |>TimeCaster::<f64>::cast_event_delta()
        |>cancel_tempo_events(250000)
        |>scale_event_time(1.0 / ppq as f64)
        |>unwrap_items()
    );

    let mut time = 0.0f64;
    let mut buckets: Vec<(u64, u64)> = Vec::new(); // (notes, all events)
    let mut total_notes = 0u64;
    let mut total_events = 0u64;

    for batch in merged {
        time += batch.delta.max(0.0);
        if time >= max_secs {
            break;
        }
        let idx = (time / bucket) as usize;
        if idx >= buckets.len() {
            buckets.resize(idx + 1, (0, 0));
        }
        for e in batch.iter_events() {
            total_events += 1;
            if let Event::NoteOn(n) = e.as_event() {
                if n.velocity > 0 {
                    total_notes += 1;
                    buckets[idx].0 += 1;
                }
            }
            buckets[idx].1 += 1;
        }
    }

    println!("ppq={ppq} max_secs={max_secs} bucket={bucket}s");
    println!("total_notes={total_notes} total_events={total_events}");
    for (i, (notes, events)) in buckets.iter().enumerate() {
        if *notes == 0 && *events == 0 {
            continue;
        }
        println!(
            "t={:>6.1}s..{:>6.1}s notes={:>9} events={:>9}",
            i as f64 * bucket,
            (i + 1) as f64 * bucket,
            notes,
            events
        );
    }
    Ok(())
}
