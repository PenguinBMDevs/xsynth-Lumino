//! 渲染线程 pacing 自旋开销测量台（`BufferedRenderer` 稳态 CPU 成本）。
//!
//! 复刻 Lumino 的实际参数：48kHz 立体声、10ms 渲染块、100ms 缓冲目标，
//! 消费者按实时节奏（10ms/块）读取，等价于常驻音频回调。
//! 渲染管道是**静音且零成本**的，因此测得的进程 CPU 基本全部来自
//! `BufferedRenderer` 的 pacing 逻辑（`spin_sleep` 的「粗睡 + 自旋」尾部），
//! 与 DSP 负载无关——即「不播放任何音符也要烧掉多少核」。
//!
//! 用法：
//! ```text
//! cargo run --release -p xsynth-realtime --example pacing_cpu -- [运行秒数]
//! ```
//!
//! 输出 `cores = 进程CPU时间 / 墙钟时间`：0.17 表示常驻烧掉 17% 单核。

use std::thread;
use std::time::{Duration, Instant};

use xsynth_core::buffered_renderer::BufferedRenderer;
use xsynth_core::{AudioStreamParams, ChannelCount, FunctionAudioPipe};

const SR: u32 = 48_000;
const BLOCK_MS: f64 = 10.0;
const CUSHION_MS: f64 = 100.0;
const DEFAULT_RUN_SECS: f64 = 3.0;

fn frames(ms: f64) -> usize {
    (SR as f64 * ms / 1000.0) as usize
}

/// 进程累计 CPU 时间（秒）。非 Windows 返回 `None`（只测墙钟与节奏）。
#[cfg(windows)]
fn process_cpu_secs() -> Option<f64> {
    use std::ffi::c_void;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn GetProcessTimes(
            process: *mut c_void,
            creation: *mut u64,
            exit: *mut u64,
            kernel: *mut u64,
            user: *mut u64,
        ) -> i32;
    }

    let (mut creation, mut exit, mut kernel, mut user) = (0u64, 0u64, 0u64, 0u64);
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return None;
    }
    // FILETIME 单位为 100ns
    Some((kernel + user) as f64 * 1e-7)
}

#[cfg(not(windows))]
fn process_cpu_secs() -> Option<f64> {
    None
}

/// 按实时节奏消费（等价音频回调）；用 `native_sleep` 保证不把自旋算进被测项。
/// 返回（墙钟秒数，启动阶段欠载样本数，稳态阶段欠载样本数）。
fn drain_realtime<F: FnMut(&mut [f32])>(
    secs: f64,
    sink: &mut [f32],
    expect: f32,
    warmup_blocks: usize,
    mut read: F,
) -> (f64, u64, u64) {
    let period = Duration::from_secs_f64(BLOCK_MS / 1000.0);
    let t0 = Instant::now();
    let mut next = t0;
    let mut block = 0usize;
    let (mut warmup_missing, mut steady_missing) = (0u64, 0u64);
    while t0.elapsed().as_secs_f64() < secs {
        next += period;
        read(sink);
        let missing = sink.iter().filter(|s| **s != expect).count() as u64;
        if block < warmup_blocks {
            warmup_missing += missing;
        } else {
            steady_missing += missing;
        }
        block += 1;
        let now = Instant::now();
        if next > now {
            spin_sleep::native_sleep(next - now);
        } else {
            next = now;
        }
    }
    (t0.elapsed().as_secs_f64(), warmup_missing, steady_missing)
}

fn report(label: &str, wall: f64, cpu: Option<f64>) {
    match cpu {
        Some(cpu) => println!(
            "{label:<28} wall={wall:6.3}s cpu={cpu:6.3}s  cores={:5.3}",
            cpu / wall
        ),
        None => println!("{label:<28} wall={wall:6.3}s cpu=n/a（非 Windows）"),
    }
}

fn main() {
    let secs: f64 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_RUN_SECS);

    let params = AudioStreamParams::new(SR, ChannelCount::Stereo);
    let block = frames(BLOCK_MS);
    let cushion = frames(CUSHION_MS);
    let mut sink = vec![0.0f32; block * 2];
    /// 管道输出的常量电平：任何不等于该值的样本都说明音频回调**没拿到数据**
    /// （`BufferedRenderer::read` 会把缺口填零）——即真实的欠载/爆音。
    const PIPE_LEVEL: f32 = 0.25;
    /// 启动预热块数（缓冲从空开始填充，前若干块必然静音，不计入欠载）。
    const WARMUP_BLOCKS: usize = 20;

    println!(
        "pacing_cpu: {}Hz 立体声 / 块 {BLOCK_MS}ms ({block} 帧) / 缓冲目标 {CUSHION_MS}ms / 运行 {secs}s",
        SR
    );

    // 基线：只有消费者（无渲染线程），用于扣除测量本身的噪声。
    let cpu0 = process_cpu_secs();
    let (wall0, _, _) = drain_realtime(secs, &mut sink, PIPE_LEVEL, WARMUP_BLOCKS, |_| {});
    let cpu0 = process_cpu_secs().zip(cpu0).map(|(a, b)| a - b);
    report("baseline(仅消费者)", wall0, cpu0);

    // 被测：真实 BufferedRenderer（零成本静音管道 → 测得的全是 pacing 成本）。
    let mut renderer = BufferedRenderer::new(
        FunctionAudioPipe::new(params, |out: &mut [f32]| out.fill(PIPE_LEVEL)),
        params,
        block,
        cushion,
    )
    .expect("BufferedRenderer::new");

    let cpu1 = process_cpu_secs();
    let (wall1, warmup_missing, steady_missing) =
        drain_realtime(secs, &mut sink, PIPE_LEVEL, WARMUP_BLOCKS, |s| {
            renderer.read(s)
        });
    let cpu1 = process_cpu_secs().zip(cpu1).map(|(a, b)| a - b);
    report("BufferedRenderer 渲染线程", wall1, cpu1);

    if let (Some(a), Some(b)) = (cpu1, cpu0) {
        let net = a - b;
        println!(
            "净增：{net:.3}s CPU / {wall1:.3}s 墙钟 = 常驻 {:.3} 核（{:.1}% 单核）",
            net / wall1,
            net / wall1 * 100.0
        );
    }
    println!(
        "欠载检查：启动期(前 {WARMUP_BLOCKS} 块)缺样 {warmup_missing} / 稳态缺样 {steady_missing}（必须为 0）"
    );
    if steady_missing > 0 {
        println!("❌ 稳态出现欠载 → 该改动不可接受");
    } else {
        println!("✅ 稳态零欠载");
    }

    drop(renderer);
    thread::sleep(Duration::from_millis(50));
}
