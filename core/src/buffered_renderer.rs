use std::{
    collections::VecDeque,
    io,
    sync::{
        atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering},
        Arc, RwLock,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crossbeam_channel::{unbounded, Receiver};

use crate::AudioStreamParams;

use super::AudioPipe;

/// 缓冲已满时的单次等待粒度上限（秒）。
///
/// 缓冲充足时渲染线程没有任何截止期限需要保精度，用 `native_sleep`（纯内核等待、
/// 不自旋）粗睡即可：唤醒频率仍 ≥ 每秒 30 次，远高于缓冲耗尽所需的分辨率。
/// 历史实现用 `spin_sleep::sleep(delay / 10)`（0.9ms）——`spin_sleep` 的语义是
/// 「粗睡 + 尾部自旋到 deadline」，而 0.9ms 恰好小于其 Windows 精度阈值（700µs），
/// 于是每次都退化为自旋，白白烧掉一整个核的若干个百分点。
const CUSHION_WAIT_MAX_SECS: f64 = 0.03;

/// 缓冲已满时的单次等待时长（秒）。
///
/// **安全不变量**：`buffered_secs >= block_secs` 时，`buffered_secs - 返回值 >= block_secs`
/// 恒成立（`wait <= (buffered - block) / 2`）。按实时消费速度，睡满后缓冲里至少还剩
/// 一个整块，因此醒来即可继续渲染，**结构上不可能造成欠载**。
fn cushion_wait_secs(buffered_secs: f64, block_secs: f64, max_wait_secs: f64) -> f64 {
    ((buffered_secs - block_secs) * 0.5)
        .min(max_wait_secs)
        .max(0.0)
}

/// 目标缓冲余量（交织 f32 计数）。
///
/// `cushion_samples` 与 `render_size` 的单位都是「帧」，而渲染线程的缓冲计数器
/// `samples` 按**交织 f32** 累计（帧 × 声道数）。两者量纲必须在这里对齐，
/// 否则配置的缓冲目标会按声道数缩水（立体声下 100ms 只剩 50ms），
/// 单次渲染尖峰即可把缓冲抽干造成 underrun。
fn cushion_target_interleaved(
    cushion_frames: usize,
    render_size_frames: usize,
    channel_count: usize,
) -> i64 {
    (cushion_frames.max(render_size_frames) * channel_count) as i64
}

/// Holds the statistics for an instance of BufferedRenderer.
#[derive(Debug, Clone)]
struct BufferedRendererStats {    samples: Arc<AtomicI64>,

    last_samples_after_read: Arc<AtomicI64>,

    last_request_samples: Arc<AtomicI64>,

    render_time: Arc<RwLock<VecDeque<f64>>>,

    render_size: Arc<AtomicUsize>,
}

/// Reads the statistics of an instance of BufferedRenderer in a usable way.
pub struct BufferedRendererStatsReader {
    stats: BufferedRendererStats,
}

impl BufferedRendererStatsReader {
    /// The number of samples currently buffered.
    /// Can be negative if the reader is waiting for more samples.
    pub fn samples(&self) -> i64 {
        self.stats.samples.load(Ordering::Relaxed)
    }

    /// The number of samples that were in the buffer after the last read.
    pub fn last_samples_after_read(&self) -> i64 {
        self.stats.last_samples_after_read.load(Ordering::Relaxed)
    }

    /// The last number of samples last requested by the read command.
    pub fn last_request_samples(&self) -> i64 {
        self.stats.last_request_samples.load(Ordering::Relaxed)
    }

    /// The number of samples to render each iteration.
    pub fn render_size(&self) -> usize {
        self.stats.render_size.load(Ordering::Relaxed)
    }

    /// The average render time percentages (0 to 1)
    /// of how long the render thread spent rendering, from the max allowed time.
    pub fn average_renderer_load(&self) -> f64 {
        let queue = self.stats.render_time.read().unwrap();
        let total = queue.len();
        if total == 0 {
            0.0
        } else {
            queue.iter().sum::<f64>() / total as f64
        }
    }

    /// The last render time percentage (0 to 1)
    /// of how long the render thread spent rendering, from the max allowed time.
    pub fn last_renderer_load(&self) -> f64 {
        let queue = self.stats.render_time.read().unwrap();
        *queue.front().unwrap_or(&0.0)
    }
}

/// The helper struct for deferred sample rendering.
/// Helps avoid stutter when the render time is exceding the max time allowed by the audio driver.
///
/// Instead, it renders in a separate thread with much smaller sample sizes, causing a minimal impact on latency
/// while allowing more time to render per sample.
///
/// Designed to be used in realtime playback only.
///
/// **线程模型（RT 安全契约）**：本结构体是**控制侧**句柄，只被非实时线程使用
/// （构造、读取统计、改渲染块大小、关机）。播放消费端在 [`BufferedRendererReader`] 里，
/// 由音频回调独占持有 —— 音频线程上**不存在任何互斥量**。
pub struct BufferedRenderer {
    stats: BufferedRendererStats,

    /// 仅用于派生 [`BufferedRendererReader`]（`Receiver` 的克隆是同一通道的另一个句柄）。
    receive: Receiver<Vec<f32>>,

    /// Whether the render thread should be killed.
    killed: Arc<AtomicBool>,

    /// The thread handle to wait for at the end.
    thread_handle: Option<JoinHandle<()>>,

    stream_params: AudioStreamParams,
}

/// 播放消费端：独占读取队列与余量缓冲，供音频回调持有。
///
/// 它被 `move` 进 cpal 的回调闭包（`FnMut`），因此回调可以直接可变访问，
/// **不需要锁、也不需要 `Arc`** —— 这正是"音频回调不阻塞/不持锁"的收口。
///
/// 每次构建音频流时用 [`BufferedRenderer::reader`] 派生一个新的 reader；
/// 重启流（设备热插拔）时旧 reader 随旧流一起析构，新流拿到新 reader。
pub struct BufferedRendererReader {
    /// The receiver for samples (the render thread has the sender).
    receive: Receiver<Vec<f32>>,

    /// Remainder of samples from the last received samples vec.
    remainder: Vec<f32>,

    stats: BufferedRendererStats,

    stream_params: AudioStreamParams,
}

impl BufferedRendererReader {
    /// Reads samples from the remainder and the output queue into the destination array.
    pub fn read(&mut self, dest: &mut [f32]) {
        dest.fill(0.0);

        let mut i: usize = 0;
        let len = dest.len().min(self.remainder.len());

        self.stats
            .last_request_samples
            .store(dest.len() as i64, Ordering::SeqCst);

        // Read from current remainder
        for r in self.remainder.drain(0..len) {
            dest[i] = r;
            i += 1;
        }

        // Read from output queue, leave the remainder if there is any.
        // Never block the audio callback: if the queue is temporarily empty
        // (render thread preempted / slow block), leave the rest as silence
        // and let the next callback continue from the queue. Blocking here
        // would stall the OS audio thread and cause glitches.
        while self.remainder.is_empty() {
            let mut buf = match self.receive.try_recv() {
                Ok(buf) => buf,
                Err(_) => break,
            };

            let len = buf.len().min(dest.len() - i);
            for r in buf.drain(0..len) {
                dest[i] = r;
                i += 1;
            }

            self.remainder = buf;
        }

        // Only subtract what was actually consumed: on an underrun the
        // remaining destination is silence, not queued samples, so charging
        // the full request would make `samples` drift permanently negative
        // and corrupt the render thread's cushion check.
        let samples = self.stats.samples.fetch_sub(i as i64, Ordering::SeqCst);
        self.stats
            .last_samples_after_read
            .store(samples, Ordering::Relaxed);
    }
}

impl BufferedRenderer {
    /// Creates a new instance of BufferedRenderer.
    ///
    /// - `render`: An object implementing the AudioPipe struct for BufferedRenderer to
    ///   read samples from
    /// - `stream_params`: Parameters of the output audio
    /// - `render_size`: The number of frames to render each iteration
    /// - `cushion_samples`: Target number of rendered-but-unconsumed **frames**
    ///   to keep buffered (clamped to at least one `render_size`). The render
    ///   thread keeps rendering until this cushion is reached, then paces at
    ///   ~90% of realtime.
    pub fn new<F: 'static + AudioPipe + Send>(
        mut render: F,
        stream_params: AudioStreamParams,
        render_size: usize,
        cushion_samples: usize,
    ) -> Result<Self, io::Error> {
        let (tx, rx) = unbounded();

        let samples = Arc::new(AtomicI64::new(0));
        let last_request_samples = Arc::new(AtomicI64::new(0));
        let render_size = Arc::new(AtomicUsize::new(render_size));

        let last_samples_after_read = Arc::new(AtomicI64::new(0));

        let render_time = Arc::new(RwLock::new(VecDeque::new()));

        let killed = Arc::new(AtomicBool::new(false));

        let thread_handle = {
            let samples = samples.clone();
            let render_size = render_size.clone();
            let render_time = render_time.clone();
            let killed = killed.clone();
            thread::Builder::new()
                .name("xsynth_buffered_rendering".to_string())
                .spawn(move || loop {
                    let size = render_size.load(Ordering::SeqCst);

                    // The expected render time per iteration. It is slightly smaller (*90/100) than
                    // the real time so the render thread can catch up if it's behind.
                    let delay =
                        Duration::from_secs(1) * size as u32 / stream_params.sample_rate * 90 / 100;

                    // Keep the configured cushion buffered. The render thread
                    // is CPU-heavy (dense black-MIDI blocks take tens of ms)
                    // and can be preempted; a shallow cushion would let the
                    // audio callback run dry while a block is still rendering,
                    // which stutters even at low average render loads. The
                    // cushion is decoupled from the block size so small blocks
                    // (tight event timing) can still have a deep buffer.
                    //
                    // 量纲对齐：`samples` 为交织 f32 计数，目标同样换算成交织计数
                    // （见 `cushion_target_interleaved`）。
                    let cushion_target = cushion_target_interleaved(
                        cushion_samples,
                        size,
                        stream_params.channels.count() as usize,
                    );
                    // 缓冲充足时的等待：`native_sleep` 只做内核等待、**不自旋**。
                    // 这里没有任何截止期限需要保精度（音频由缓冲余量兜底），
                    // 自旋纯属浪费；等待粒度同时受「剩余余量」约束以保证不欠载。
                    let channels = stream_params.channels.count() as usize;
                    let interleaved_per_sec = stream_params.sample_rate as f64 * channels as f64;
                    let block_secs = size as f64 / stream_params.sample_rate as f64;
                    loop {
                        let buffered = samples.load(Ordering::SeqCst);
                        if buffered <= cushion_target {
                            break;
                        }

                        // 安全上界见 `cushion_wait_secs`：醒来时缓冲里至少还剩一个整块。
                        // 故本改动只减少唤醒次数，不改变任何一块音频的内容，也不会欠载。
                        let buffered_secs = buffered.max(0) as f64 / interleaved_per_sec;
                        let wait_secs = cushion_wait_secs(
                            buffered_secs,
                            block_secs,
                            CUSHION_WAIT_MAX_SECS.min(delay.as_secs_f64()),
                        );
                        if wait_secs > 0.0 {
                            spin_sleep::native_sleep(Duration::from_secs_f64(wait_secs));
                        } else {
                            thread::yield_now();
                        }

                        if killed.load(Ordering::Acquire) {
                            return;
                        }
                    }

                    let start = Instant::now();
                    let end = start + delay;

                    // Create the vec and write the samples
                    let mut vec =
                        vec![Default::default(); size * stream_params.channels.count() as usize];
                    crate::profiling::tracy_zone!("buffered_render_chunk", {
                        render.read_samples(&mut vec);
                    });

                    // Send the samples, break if the pipe is broken
                    samples.fetch_add(vec.len() as i64, Ordering::SeqCst);
                    match tx.send(vec) {
                        Ok(_) => {}
                        Err(_) => break,
                    };

                    // Write the elapsed render time percentage to the render_time queue
                    {
                        let mut queue = render_time.write().unwrap();
                        let elaspsed = start.elapsed().as_secs_f64();
                        let total = delay.as_secs_f64();
                        queue.push_front(elaspsed / total);
                        if queue.len() > 100 {
                            queue.pop_back();
                        }
                    }

                    // Sleep until the next iteration.
                    // 与缓冲等待同理：`native_sleep` 的抖动（平均 ~150µs、最坏 ~730µs）
                    // 远小于缓冲余量，不需要 `spin_sleep` 的尾部自旋换来的亚毫秒精度。
                    let now = Instant::now();
                    if end > now {
                        spin_sleep::native_sleep(end - now);
                    }
                })?
        };

        Ok(Self {
            stats: BufferedRendererStats {
                samples,
                last_request_samples,
                render_time,
                render_size,
                last_samples_after_read,
            },
            receive: rx,
            stream_params,
            thread_handle: Some(thread_handle),
            killed,
        })
    }

    /// 派生一个播放消费端（供音频回调独占持有）。
    ///
    /// `Receiver` 的克隆是同一通道的另一个句柄；正常情况同一时刻只有一个活跃的
    /// reader（重启流时旧流先析构），与历史"共享 `BufferedRenderer`"的语义一致。
    pub fn reader(&self) -> BufferedRendererReader {
        BufferedRendererReader {
            receive: self.receive.clone(),
            remainder: Vec::new(),
            stats: self.stats.clone(),
            stream_params: self.stream_params,
        }
    }

    /// Returns the stream parameters of the audio output.
    pub fn stream_params(&self) -> AudioStreamParams {
        self.stream_params
    }

    /// Sets the number of samples that should be rendered each iteration.
    ///
    /// 纯原子写：非实时线程调用不会阻塞音频线程（**无锁**）。
    pub fn set_render_size(&self, size: usize) {
        self.stats.render_size.store(size, Ordering::SeqCst);
    }

    /// Returns a statistics reader.
    /// See the `BufferedRendererStatsReader` documentation for more information.
    ///
    /// 只克隆 `Arc`：非实时线程调用不会阻塞音频线程（**无锁**）。
    pub fn get_buffer_stats(&self) -> BufferedRendererStatsReader {
        BufferedRendererStatsReader {
            stats: self.stats.clone(),
        }
    }
}

impl Drop for BufferedRenderer {
    fn drop(&mut self) {
        self.killed.store(true, Ordering::Release);
        if let Some(handle) = self.thread_handle.take() {
            if handle.join().is_err() {
                eprintln!("xsynth-core: buffered renderer thread panicked during shutdown");
            }
        }
    }
}

impl AudioPipe for BufferedRendererReader {
    fn stream_params(&self) -> &'_ AudioStreamParams {
        &self.stream_params
    }

    fn read_samples_unchecked(&mut self, to: &mut [f32]) {
        self.read(to)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            atomic::{AtomicI64, AtomicUsize, Ordering},
            Arc, RwLock,
        },
        time::Duration,
    };

    use super::{
        BufferedRenderer, BufferedRendererReader, BufferedRendererStats,
        BufferedRendererStatsReader, CUSHION_WAIT_MAX_SECS, cushion_target_interleaved,
        cushion_wait_secs,
    };
    use crate::audio_pipe::AudioPipe;
    use crate::audio_stream::{AudioStreamParams, ChannelCount};
    use crate::FunctionAudioPipe;
    use crossbeam_channel::{Receiver, unbounded};

    fn test_stats() -> BufferedRendererStats {
        BufferedRendererStats {
            samples: Arc::new(AtomicI64::new(0)),
            last_samples_after_read: Arc::new(AtomicI64::new(0)),
            last_request_samples: Arc::new(AtomicI64::new(0)),
            render_time: Arc::new(RwLock::new(VecDeque::new())),
            render_size: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn test_reader(
        receive: Receiver<Vec<f32>>,
        stats: BufferedRendererStats,
    ) -> BufferedRendererReader {
        BufferedRendererReader {
            receive,
            remainder: Vec::new(),
            stats,
            stream_params: AudioStreamParams::new(48_000, ChannelCount::Stereo),
        }
    }

    #[test]
    fn average_renderer_load_is_zero_when_no_samples_have_been_rendered() {
        let reader = BufferedRendererStatsReader {
            stats: BufferedRendererStats {
                samples: Arc::new(AtomicI64::new(0)),
                last_samples_after_read: Arc::new(AtomicI64::new(0)),
                last_request_samples: Arc::new(AtomicI64::new(0)),
                render_time: Arc::new(RwLock::new(VecDeque::new())),
                render_size: Arc::new(AtomicUsize::new(0)),
            },
        };

        assert_eq!(reader.average_renderer_load(), 0.0);
        assert_eq!(reader.last_renderer_load(), 0.0);
    }

    #[test]
    fn reader_zero_fills_underrun_and_counts_only_consumed_samples() {
        // 播放侧语义契约（从 `BufferedRenderer::read` 迁移而来，必须逐位一致）：
        // 队列不足时余下部分是**静音**，且 `samples` 只扣实际消费量——
        // 若按请求量扣，欠载会让计数永久偏负并破坏渲染线程的缓冲判断。
        let (tx, rx) = unbounded();
        let stats = test_stats();
        stats.samples.store(4, Ordering::SeqCst);
        let mut reader = test_reader(rx, stats.clone());
        tx.send(vec![1.0, 2.0, 3.0, 4.0]).expect("send block");

        let mut dest = [9.0f32; 8];
        reader.read(&mut dest);

        assert_eq!(dest, [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(stats.last_request_samples.load(Ordering::SeqCst), 8);
        // 契约钉住：`last_samples_after_read` 存的是**本次读取前**的缓冲计数
        // （`fetch_sub` 的返回值），不是读后的剩余量。历史语义如此，
        // `perf_nps` 的 `b < 0`（读前已空 = 欠载）判定依赖它，故不改，只固化。
        assert_eq!(stats.last_samples_after_read.load(Ordering::Relaxed), 4);
        assert_eq!(
            stats.samples.load(Ordering::SeqCst),
            0,
            "只应扣掉实际消费的 4 个样本"
        );
    }

    #[test]
    fn reader_keeps_remainder_for_next_read() {
        // 一个块大于一次请求时，多余样本必须留到下一次读取（不得丢弃、不得重复）。
        let (tx, rx) = unbounded();
        let stats = test_stats();
        stats.samples.store(8, Ordering::SeqCst);
        let mut reader = test_reader(rx, stats.clone());
        tx.send(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("send block");

        let mut first = [0.0f32; 4];
        reader.read(&mut first);
        assert_eq!(first, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(stats.samples.load(Ordering::SeqCst), 4);

        // 第二次读取完全由 remainder 供数，无需新块。
        let mut second = [0.0f32; 4];
        reader.read(&mut second);
        assert_eq!(second, [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(stats.samples.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn reader_reports_stream_params_as_audio_pipe() {
        let (_tx, rx) = unbounded();
        let reader = test_reader(rx, test_stats());
        assert_eq!(reader.stream_params().sample_rate, 48_000);
    }

    #[test]
    fn control_side_is_lock_free_while_reader_is_active() {
        // 本改动核心契约：控制侧（统计读取 / 改渲染块大小）与播放侧（reader）
        // 之间**不再有共享锁**——非实时线程可自由调用，结构上不可能阻塞音频线程。
        // 这里做一次真实的跨线程并发调用（若仍存在共享锁，此测试就是那个锁的争用点）。
        let params = AudioStreamParams::new(48_000, ChannelCount::Stereo);
        let renderer = Arc::new(
            BufferedRenderer::new(
                FunctionAudioPipe::new(params, |out: &mut [f32]| out.fill(0.25)),
                params,
                480,
                4_800,
            )
            .expect("BufferedRenderer::new"),
        );
        let mut reader = renderer.reader();

        let control = Arc::clone(&renderer);
        let handle = std::thread::spawn(move || {
            for _ in 0..2_000 {
                let _ = control.get_buffer_stats().samples();
                control.set_render_size(480);
            }
        });

        let mut dest = vec![0.0f32; 960];
        for _ in 0..120 {
            reader.read(&mut dest);
            std::thread::sleep(Duration::from_millis(1));
        }
        handle.join().expect("控制侧线程不得 panic");
        drop(reader);
        drop(renderer);
    }

    #[test]
    fn cushion_target_matches_interleaved_counter_units() {
        // 立体声、100ms 目标（4800 帧）、10ms 块（480 帧）→ 交织计数 9600
        assert_eq!(cushion_target_interleaved(4800, 480, 2), 9600);
        // 目标小于块长时以块长为下限（再换算成交织计数）
        assert_eq!(cushion_target_interleaved(100, 480, 2), 960);
        // 单声道：帧 == 交织计数
        assert_eq!(cushion_target_interleaved(4800, 480, 1), 4800);
    }

    #[test]
    fn cushion_wait_never_drains_below_one_block() {
        // 这是「渲染线程粗睡不自旋」改动的安全契约：醒来时缓冲里必须还有一个整块。
        // 覆盖真实配置组合：块 5/10/20ms，缓冲余量从「刚好一块」到「远端充足」。
        for block_secs in [0.005f64, 0.01, 0.02] {
            for buffered_secs in [
                block_secs,
                block_secs * 1.001,
                block_secs * 1.5,
                block_secs * 3.0,
                0.1,
                0.5,
                3.0,
            ] {
                let wait = cushion_wait_secs(
                    buffered_secs,
                    block_secs,
                    CUSHION_WAIT_MAX_SECS.min(block_secs * 0.9),
                );
                assert!(wait >= 0.0, "等待时长不得为负：{wait}");
                assert!(
                    wait <= CUSHION_WAIT_MAX_SECS,
                    "等待时长必须受上限约束：{wait}"
                );
                assert!(
                    buffered_secs - wait >= block_secs,
                    "睡眠后不足一个整块（buffered={buffered_secs} block={block_secs} wait={wait}）"
                );
            }
        }
        // 缓冲不足一个块（异常/启动态）：不得产生等待，避免与消费者抢时间。
        assert_eq!(cushion_wait_secs(0.004, 0.01, CUSHION_WAIT_MAX_SECS), 0.0);
        assert_eq!(cushion_wait_secs(0.0, 0.01, CUSHION_WAIT_MAX_SECS), 0.0);
        // 负计数（欠载瞬间 `samples` 可能为负）同样不得等待。
        assert_eq!(cushion_wait_secs(-0.02, 0.01, CUSHION_WAIT_MAX_SECS), 0.0);
    }
}
