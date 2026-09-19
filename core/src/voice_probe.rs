//! 性能探针（`voice_probe` feature 门控）。
//!
//! 目的：把 `channel_keys` 内的 voice 渲染拆到阶段级（pitch / 时间推进 / 采样抓取 /
//! 包络 / 逐声部滤波 / 混音循环），用于回答“渲染到底慢在哪一段”。
//!
//! 运行方式：编译期启用 `voice_probe` feature，运行期设置
//! `XSYNTH_VOICE_PROBE=1`（采样步长 `XSYNTH_VOICE_PROBE_STRIDE`，默认 64）。
//! 每 `stride` 个新建 voice 采样 1 个（包装为 [`ProbeVoice`]），
//! 由被采样 voice 的 `render_to` 每约 2 秒向 stderr 打一行聚合统计。
//!
//! 计数设计：所有累加先写线程本地（TLS `Cell`），`flush_if_due` 时合并到全局原子。
//! 共享原子 `fetch_add` 在 16 个通道线程并发下会造成严重的缓存行争用，
//! 实测会把被测量本身放大数倍（探针自扰动），因此热路径**绝不**直接碰全局原子。
//!
//! 计时用 TSC（`_rdtsc`，~6ns）而非 `Instant::now()`（本机 ~41ns/次），
//! 避免时钟开销淹没 100~300ns 的阶段测量。
//!
//! 未启用 feature 时本模块全部为空操作（`should_probe` 恒为 `false`），
//! 生产构建零成本。

use crate::voice::{
    ReleaseType, Voice, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator,
};

#[cfg(feature = "voice_probe")]
mod imp {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::OnceLock;
    use std::time::Instant;

    /// 线程本地累加器（热路径只写这里；每 2s 合并进全局）。
    struct Local {
        voice_calls: Cell<u64>,
        voice_ns: Cell<u64>,
        voice_cpu_ns: Cell<u64>,
        voice_frames: Cell<u64>,
        chain_ns: Cell<u64>,
        env_ns: Cell<u64>,
        cut_ns: Cell<u64>,
        smpl_calls: Cell<u64>,
        pitch_ns: Cell<u64>,
        time_ns: Cell<u64>,
        grab_ns: Cell<u64>,
        samples: Cell<u64>,
        probed: Cell<u64>,
        filtered: Cell<u64>,
    }

    impl Local {
        const fn new() -> Self {
            Local {
                voice_calls: Cell::new(0),
                voice_ns: Cell::new(0),
                voice_cpu_ns: Cell::new(0),
                voice_frames: Cell::new(0),
                chain_ns: Cell::new(0),
                env_ns: Cell::new(0),
                cut_ns: Cell::new(0),
                smpl_calls: Cell::new(0),
                pitch_ns: Cell::new(0),
                time_ns: Cell::new(0),
                grab_ns: Cell::new(0),
                samples: Cell::new(0),
                probed: Cell::new(0),
                filtered: Cell::new(0),
            }
        }
    }

    thread_local! {
        static LOCAL: Local = const { Local::new() };
    }

    #[inline(always)]
    fn add(cell: &Cell<u64>, v: u64) {
        cell.set(cell.get().wrapping_add(v));
    }

    /// voice 采样序号（每 spawn 一个自增，命中步长的被采样）。
    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// 全局聚合计数（仅由 `flush_if_due` 低频合并/读取）。
    static G_VOICE_CALLS: AtomicU64 = AtomicU64::new(0);
    static G_VOICE_NS: AtomicU64 = AtomicU64::new(0);
    static G_VOICE_CPU_NS: AtomicU64 = AtomicU64::new(0);
    static G_VOICE_FRAMES: AtomicU64 = AtomicU64::new(0);
    static G_CHAIN_NS: AtomicU64 = AtomicU64::new(0);
    static G_ENV_NS: AtomicU64 = AtomicU64::new(0);
    static G_CUT_NS: AtomicU64 = AtomicU64::new(0);
    static G_SMPL_CALLS: AtomicU64 = AtomicU64::new(0);
    static G_PITCH_NS: AtomicU64 = AtomicU64::new(0);
    static G_TIME_NS: AtomicU64 = AtomicU64::new(0);
    static G_GRAB_NS: AtomicU64 = AtomicU64::new(0);
    static G_SAMPLES: AtomicU64 = AtomicU64::new(0);
    static G_PROBED_VOICES: AtomicU64 = AtomicU64::new(0);
    static G_FILTERED_VOICES: AtomicU64 = AtomicU64::new(0);

    static LAST_FLUSH_MS: AtomicU64 = AtomicU64::new(0);

    fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(|| std::env::var_os("XSYNTH_VOICE_PROBE").is_some())
    }

    fn stride() -> u64 {
        static S: OnceLock<u64> = OnceLock::new();
        *S.get_or_init(|| {
            std::env::var("XSYNTH_VOICE_PROBE_STRIDE")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(64)
        })
    }

    /// 是否对该 voice 采样（在 voice 创建时调用一次）。
    #[inline(always)]
    pub fn should_probe() -> bool {
        enabled() && SEQ.fetch_add(1, Ordering::Relaxed).is_multiple_of(stride())
    }

    /// rdtsc 计数刻度（x86_64；其他平台 `None`，探针自动静默）。
    #[inline(always)]
    pub fn tick() -> Option<u64> {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: rdtsc 在 x86_64 上恒可用（TSC 由 CPU 保证存在）。
            Some(unsafe { core::arch::x86_64::_rdtsc() })
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            None
        }
    }

    /// tick → ns 换算因子（首次调用忙等 1ms 标定）。
    fn ns_per_tick() -> f64 {
        static F: OnceLock<f64> = OnceLock::new();
        *F.get_or_init(|| {
            let t0 = Instant::now();
            let c0 = tick().unwrap_or(0);
            let mut c1 = c0;
            while t0.elapsed().as_micros() < 1000 {
                if let Some(c) = tick() {
                    c1 = c;
                }
            }
            let ns = t0.elapsed().as_nanos().max(1) as f64;
            (ns / (c1.saturating_sub(c0).max(1)) as f64).max(0.0001)
        })
    }

    /// 采样器阶段标记：`p0` 进入、`p1` pitch 结束、`p2` 时间推进结束、结束即抓取完成。
    pub struct SamplerMarks {
        p0: u64,
        p1: Option<u64>,
        p2: Option<u64>,
    }

    /// 开始计时（仅当该 voice 被采样且探针开启）。非采样路径只做一次 bool 判断。
    #[inline(always)]
    pub fn sampler_begin(probe: bool) -> Option<SamplerMarks> {
        if probe {
            tick().map(|t| SamplerMarks {
                p0: t,
                p1: None,
                p2: None,
            })
        } else {
            None
        }
    }

    /// 阶段打点：`stage == 1` 为 pitch 结束，`stage == 2` 为时间推进结束。
    #[inline(always)]
    pub fn sampler_mark(marks: &mut Option<SamplerMarks>, stage: u8) {
        if let (Some(m), Some(now)) = (marks.as_mut(), tick()) {
            if stage == 1 {
                m.p1 = Some(now);
            } else {
                m.p2 = Some(now);
            }
        }
    }

    /// 结束计时并累计（`width` = 本次抓取覆盖的样本数）。
    #[inline(always)]
    pub fn sampler_end(marks: &mut Option<SamplerMarks>, width: usize) {
        let Some(m) = marks else { return };
        let Some(end) = tick() else { return };
        let f = ns_per_tick();
        LOCAL.with(|l| {
            if let (Some(p1), Some(p2)) = (m.p1, m.p2) {
                add(&l.pitch_ns, ((p1 - m.p0) as f64 * f) as u64);
                add(&l.time_ns, ((p2 - p1) as f64 * f) as u64);
                add(&l.grab_ns, ((end - p2) as f64 * f) as u64);
            }
            add(&l.smpl_calls, 1);
            add(&l.samples, width as u64);
        });
    }

    /// 记录一次整 voice 渲染耗时（墙钟 ns + 线程 CPU ns + 覆盖立体声帧数）。
    #[inline(always)]
    pub fn record_voice_render(wall_ns: u64, cpu_ns: u64, frames: u64) {
        LOCAL.with(|l| {
            add(&l.voice_calls, 1);
            add(&l.voice_ns, wall_ns);
            add(&l.voice_cpu_ns, cpu_ns);
            add(&l.voice_frames, frames);
        });
    }

    /// 记录一次 voice 内链式生成器的累计 TSC 刻度。
    #[inline(always)]
    pub fn record_chain_ticks(ticks: u64) {
        let ns = (ticks as f64 * ns_per_tick()) as u64;
        LOCAL.with(|l| add(&l.chain_ns, ns));
    }

    /// 记录一次包络 `next_sample` 耗时。
    #[inline(always)]
    pub fn record_env_ticks(ticks: u64) {
        let ns = (ticks as f64 * ns_per_tick()) as u64;
        LOCAL.with(|l| add(&l.env_ns, ns));
    }

    /// 记录一次逐声部 biquad 滤波耗时。
    #[inline(always)]
    pub fn record_cut_ticks(ticks: u64) {
        let ns = (ticks as f64 * ns_per_tick()) as u64;
        LOCAL.with(|l| add(&l.cut_ns, ns));
    }

    /// 记录一个探针 voice 及其是否带逐声部滤波（在 voice 创建时调用一次）。
    #[inline(always)]
    pub fn record_voice_kind(filtered: bool) {
        LOCAL.with(|l| {
            add(&l.probed, 1);
            if filtered {
                add(&l.filtered, 1);
            }
        });
    }

    /// 当前线程 CPU 周期（Windows；其他平台返回 `None`）。
    #[cfg(windows)]
    pub fn cpu_cycles_now() -> Option<u64> {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentThread() -> *mut core::ffi::c_void;
        }
        #[link(name = "ntdll")]
        unsafe extern "system" {
            fn QueryThreadCycleTime(thread: *mut core::ffi::c_void, cycles: *mut u64) -> u8;
        }
        unsafe {
            let mut cycles = 0u64;
            let ok = QueryThreadCycleTime(GetCurrentThread(), &mut cycles);
            (ok != 0).then_some(cycles)
        }
    }

    #[cfg(not(windows))]
    pub fn cpu_cycles_now() -> Option<u64> {
        None
    }

    /// 周期 → 纳秒换算比（首次调用忙等 1ms 标定一次）。
    pub fn cpu_cycles_per_ns() -> f64 {
        static RATIO: OnceLock<f64> = OnceLock::new();
        *RATIO.get_or_init(|| {
            let c0 = cpu_cycles_now().unwrap_or(0);
            let t0 = Instant::now();
            while t0.elapsed().as_micros() < 1000 {
                std::hint::spin_loop();
            }
            let c1 = cpu_cycles_now().unwrap_or(0);
            (c1.saturating_sub(c0) as f64 / t0.elapsed().as_nanos().max(1) as f64).max(0.001)
        })
    }

    /// 每约 2 秒由被采样 voice 触发一次聚合输出。
    ///
    /// 每次调用先把本线程 TLS 计数合并进全局（低频、无争用），
    /// 达到 2s 间隔时 swap 全局计数并打印一行统计。
    pub fn flush_if_due() {
        LOCAL.with(|l| {
            G_VOICE_CALLS.fetch_add(l.voice_calls.replace(0), Ordering::Relaxed);
            G_VOICE_NS.fetch_add(l.voice_ns.replace(0), Ordering::Relaxed);
            G_VOICE_CPU_NS.fetch_add(l.voice_cpu_ns.replace(0), Ordering::Relaxed);
            G_VOICE_FRAMES.fetch_add(l.voice_frames.replace(0), Ordering::Relaxed);
            G_CHAIN_NS.fetch_add(l.chain_ns.replace(0), Ordering::Relaxed);
            G_ENV_NS.fetch_add(l.env_ns.replace(0), Ordering::Relaxed);
            G_CUT_NS.fetch_add(l.cut_ns.replace(0), Ordering::Relaxed);
            G_SMPL_CALLS.fetch_add(l.smpl_calls.replace(0), Ordering::Relaxed);
            G_PITCH_NS.fetch_add(l.pitch_ns.replace(0), Ordering::Relaxed);
            G_TIME_NS.fetch_add(l.time_ns.replace(0), Ordering::Relaxed);
            G_GRAB_NS.fetch_add(l.grab_ns.replace(0), Ordering::Relaxed);
            G_SAMPLES.fetch_add(l.samples.replace(0), Ordering::Relaxed);
            G_PROBED_VOICES.fetch_add(l.probed.replace(0), Ordering::Relaxed);
            G_FILTERED_VOICES.fetch_add(l.filtered.replace(0), Ordering::Relaxed);
        });

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let last = LAST_FLUSH_MS.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last) < 2000 {
            return;
        }
        if LAST_FLUSH_MS
            .compare_exchange(last, now_ms, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }

        let voice_calls = G_VOICE_CALLS.swap(0, Ordering::Relaxed);
        let voice_ns = G_VOICE_NS.swap(0, Ordering::Relaxed);
        let voice_cpu_ns = G_VOICE_CPU_NS.swap(0, Ordering::Relaxed);
        let voice_frames = G_VOICE_FRAMES.swap(0, Ordering::Relaxed);
        let chain_ns = G_CHAIN_NS.swap(0, Ordering::Relaxed);
        let env_ns = G_ENV_NS.swap(0, Ordering::Relaxed);
        let cut_ns = G_CUT_NS.swap(0, Ordering::Relaxed);
        let probed = G_PROBED_VOICES.swap(0, Ordering::Relaxed);
        let filtered = G_FILTERED_VOICES.swap(0, Ordering::Relaxed);
        let smpl_calls = G_SMPL_CALLS.swap(0, Ordering::Relaxed);
        let pitch_ns = G_PITCH_NS.swap(0, Ordering::Relaxed);
        let time_ns = G_TIME_NS.swap(0, Ordering::Relaxed);
        let grab_ns = G_GRAB_NS.swap(0, Ordering::Relaxed);
        let samples = G_SAMPLES.swap(0, Ordering::Relaxed);

        let secs = 2.0f64;
        let per = |ns: u64, n: u64| {
            if n == 0 {
                0.0
            } else {
                ns as f64 / n as f64
            }
        };
        let sampler_total = pitch_ns + time_ns + grab_ns;
        let mix_ns = voice_ns.saturating_sub(chain_ns);
        let cpu_frac = if voice_ns == 0 {
            0.0
        } else {
            voice_cpu_ns as f64 / voice_ns as f64
        };
        eprintln!(
            "[VP] voice/s={:.0} frames/voice={:.0} avg_voice wall={:.0}ns cpu={:.0}ns cpu/wall={:.2} | chain={:.0}ns mix={:.0}ns | smpl/s={:.0} ns/call: pitch={:.0} time={:.0} grab={:.0} total={:.0} env={:.0} cut={:.0} | ns/sample={:.2} | ksample/s={:.0} | filtered={:.0}%",
            voice_calls as f64 / secs,
            per(voice_frames, voice_calls),
            per(voice_ns, voice_calls),
            per(voice_cpu_ns, voice_calls),
            cpu_frac,
            per(chain_ns, voice_calls),
            per(mix_ns, voice_calls),
            smpl_calls as f64 / secs,
            per(pitch_ns, smpl_calls),
            per(time_ns, smpl_calls),
            per(grab_ns, smpl_calls),
            per(sampler_total, smpl_calls),
            per(env_ns, smpl_calls),
            per(cut_ns, smpl_calls),
            per(sampler_total, samples),
            samples as f64 / secs / 1000.0,
            if probed == 0 {
                0.0
            } else {
                filtered as f64 / probed as f64 * 100.0
            },
        );
    }
}

#[cfg(not(feature = "voice_probe"))]
mod imp {
    /// 未启用 feature：恒不采样。
    #[inline(always)]
    pub fn should_probe() -> bool {
        false
    }

    #[inline(always)]
    pub fn sampler_begin(_probe: bool) -> Option<()> {
        None
    }

    #[inline(always)]
    pub fn sampler_mark(_marks: &mut Option<()>, _stage: u8) {}

    #[inline(always)]
    pub fn sampler_end(_marks: &mut Option<()>, _width: usize) {}

    #[inline(always)]
    pub fn record_voice_render(_wall_ns: u64, _cpu_ns: u64, _frames: u64) {}

    #[inline(always)]
    pub fn record_chain_ticks(_ticks: u64) {}

    #[inline(always)]
    pub fn record_env_ticks(_ticks: u64) {}

    #[inline(always)]
    pub fn record_cut_ticks(_ticks: u64) {}

    #[inline(always)]
    pub fn record_voice_kind(_filtered: bool) {}

    #[inline(always)]
    pub fn tick() -> Option<u64> {
        None
    }

    #[inline(always)]
    pub fn cpu_cycles_now() -> Option<u64> {
        None
    }

    #[inline(always)]
    pub fn cpu_cycles_per_ns() -> f64 {
        1.0
    }

    #[inline(always)]
    pub fn flush_if_due() {}
}

pub use imp::*;

/// 被采样 voice 的包装器：转发全部行为，仅在 `render_to` 上做整段计时。
///
/// 只有 `should_probe()` 命中的 voice 会被包装（约 1/stride），
/// 因此计时开销不会进入绝大多数 voice 的渲染路径。
pub struct ProbeVoice {
    inner: Box<dyn Voice>,
}

impl ProbeVoice {
    pub fn new(inner: Box<dyn Voice>) -> Self {
        ProbeVoice { inner }
    }
}

impl VoiceGeneratorBase for ProbeVoice {
    #[inline(always)]
    fn ended(&self) -> bool {
        self.inner.ended()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.inner.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.inner.process_controls(control);
    }
}

impl VoiceSampleGenerator for ProbeVoice {
    fn render_to(&mut self, buffer: &mut [f32]) {
        #[cfg(feature = "voice_probe")]
        {
            let start = std::time::Instant::now();
            let cpu0 = cpu_cycles_now();
            self.inner.render_to(buffer);
            let wall_ns = start.elapsed().as_nanos() as u64;
            let cpu_ns = match (cpu0, cpu_cycles_now()) {
                (Some(c0), Some(c1)) => {
                    ((c1.saturating_sub(c0)) as f64 / cpu_cycles_per_ns()) as u64
                }
                _ => 0,
            };
            record_voice_render(wall_ns, cpu_ns, (buffer.len() / 2) as u64);
            flush_if_due();
        }
        #[cfg(not(feature = "voice_probe"))]
        {
            self.inner.render_to(buffer);
        }
    }
}

impl Voice for ProbeVoice {
    #[inline(always)]
    fn is_releasing(&self) -> bool {
        self.inner.is_releasing()
    }

    #[inline(always)]
    fn is_killed(&self) -> bool {
        self.inner.is_killed()
    }

    #[inline(always)]
    fn velocity(&self) -> u8 {
        self.inner.velocity()
    }

    #[inline(always)]
    fn exclusive_class(&self) -> Option<u8> {
        self.inner.exclusive_class()
    }
}
