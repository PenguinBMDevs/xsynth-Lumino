use std::{
    collections::VecDeque,
    io,
    sync::{
        atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self},
    time::Instant,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    BuildStreamError, DefaultStreamConfigError, Device, PauseStreamError, PlayStreamError,
    SizedSample, Stream, SupportedStreamConfig,
};
use crossbeam_channel::{bounded, unbounded};
use thiserror::Error;

use xsynth_core::{
    buffered_renderer::{BufferedRenderer, BufferedRendererStatsReader},
    channel::{ChannelAudioEvent, ChannelConfigEvent, ChannelEvent, VoiceChannel},
    channel_group::SynthFormat,
    effects::VolumeLimiter,
    helpers::{prepapre_cache_vec, sum_simd},
    AudioPipe, AudioStreamParams, FunctionAudioPipe,
};

use crate::{
    util::ReadWriteAtomicU64, EmergencyGate, Governor, RealtimeEventSender, SynthEvent,
    ThreadCount, XSynthRealtimeConfig, DEFAULT_HARD_MAX_VOICES,
};

#[derive(Debug, Error)]
pub enum RealtimeSynthError {
    #[error("failed to find output device")]
    NoOutputDevice,

    #[error("failed to get default output config: {0}")]
    DefaultOutputConfig(#[from] DefaultStreamConfigError),

    #[error("failed to build thread pool: {0}")]
    ThreadPoolBuild(#[from] rayon::ThreadPoolBuildError),

    #[error("failed to spawn realtime channel thread: {0}")]
    ChannelThreadSpawn(#[source] io::Error),

    #[error("failed to spawn realtime stream thread: {0}")]
    StreamThreadSpawn(#[source] io::Error),

    #[error("realtime stream thread terminated during startup")]
    StreamThreadInit,

    #[error("failed to create realtime event sender: {0}")]
    EventSenderInit(#[source] io::Error),

    #[error("failed to spawn buffered renderer thread: {0}")]
    BufferedRendererThreadSpawn(#[source] io::Error),

    #[error("failed to create audio stream: {0}")]
    BuildStream(#[from] BuildStreamError),

    #[error("failed to start audio stream: {0}")]
    PlayStream(#[from] PlayStreamError),

    #[error("unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),
}

/// 音频流重定向（restart）错误。
///
/// Lumino vendored 扩展：音频设备被拔出/更换后，将输出流重定向到
/// 系统默认输出设备（合成管线保持不变）时可能出现的错误。
#[derive(Debug, Error)]
pub enum StreamRestartError {
    /// 找不到默认输出设备
    #[error("failed to find default output device")]
    NoDefaultDevice,

    /// 获取默认输出配置失败
    #[error("failed to get default output config: {0}")]
    DefaultOutputConfig(#[from] DefaultStreamConfigError),

    /// 新设备参数与合成管线参数不一致（采样率/声道数/采样格式），
    /// 直接重定向会导致数据语义错乱，必须重建整个合成管线。
    #[error("output device config changed, pipeline must be rebuilt: {0}")]
    ConfigChanged(String),

    /// 构建输出流失败
    #[error("failed to build output stream: {0}")]
    Build(#[from] BuildStreamError),

    /// 启动输出流失败
    #[error("failed to play output stream: {0}")]
    Play(#[from] PlayStreamError),

    /// 不支持的采样格式
    #[error("unsupported sample format: {0:?}")]
    UnsupportedSampleFormat(cpal::SampleFormat),

    /// stream owner 线程已退出（合成器已关闭）
    #[error("stream owner thread is not running")]
    StreamThreadDown,
}

/// Holds the statistics for an instance of RealtimeSynth.
#[derive(Debug, Clone)]
struct RealtimeSynthStats {
    voice_count: Arc<AtomicU64>,
}

impl RealtimeSynthStats {
    pub fn new() -> RealtimeSynthStats {
        RealtimeSynthStats {
            voice_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Reads the statistics of an instance of RealtimeSynth in a usable way.
pub struct RealtimeSynthStatsReader {
    buffered_stats: BufferedRendererStatsReader,
    stats: RealtimeSynthStats,
}

impl RealtimeSynthStatsReader {
    pub(self) fn new(
        stats: RealtimeSynthStats,
        buffered_stats: BufferedRendererStatsReader,
    ) -> RealtimeSynthStatsReader {
        RealtimeSynthStatsReader {
            stats,
            buffered_stats,
        }
    }

    /// Returns the active voice count of all the MIDI channels.
    pub fn voice_count(&self) -> u64 {
        self.stats.voice_count.load(Ordering::Relaxed)
    }

    /// Returns the statistics of the buffered renderer used.
    ///
    /// See the BufferedRendererStatsReader documentation for more information.
    pub fn buffer(&self) -> &BufferedRendererStatsReader {
        &self.buffered_stats
    }
}

struct RealtimeSynthThreadSharedData {
    buffered_renderer: Arc<Mutex<BufferedRenderer>>,

    stream_control: crossbeam_channel::Sender<StreamCommand>,

    event_senders: RealtimeEventSender,
}

/// 通道线程命令。
///
/// - `Render`：渲染一块并送回管线（现行为）；
/// - `Steal`：软抢占（分级选择 + 短淡出 + 死期限）；
/// - `HardSteal`：硬抢占（L2 重度治理：跳过淡出立即移除）；
/// - `WatchdogReset`：L4 看门狗自愈（每键仅保留最新 N 组）。
enum ChannelCommand {
    Render(Vec<f32>),
    Steal(usize),
    HardSteal(usize),
    WatchdogReset(usize),
}

struct PreparedRealtimeChannels {
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    senders: Vec<crossbeam_channel::Sender<ChannelEvent>>,
    command_senders: Vec<crossbeam_channel::Sender<ChannelCommand>>,
    /// 全局 NoteOn 入场预算池（软目标 - 当前声部数，管道每块刷新）。
    /// 全局共享：单通道文件可独享整个预算，多通道文件按需竞争。
    admission_budget: Arc<AtomicI64>,
    join_handles: Vec<thread::JoinHandle<()>>,
    output_receiver: crossbeam_channel::Receiver<Vec<f32>>,
}

/// 每通道音频域混音状态：增益（线性，1.0 = 0 dB）与声像（-1..1，0 = 居中）。
///
/// 由 UI 线程写入、各通道独立音频线程读取，使用 `AtomicU32` 无锁访问
/// （以位模式存储 `f32`，避免对 `AtomicF32` 可用性的依赖）。
/// 索引 = MIDI 通道号（0..channel_count）。
///
/// 设为 `pub` 以便上层（如 lumino `XSynth` 后端）在 `RealtimeSynth` 之外
/// 通过共享句柄设置每通道增益/声像。
pub struct ChannelMix {
    pub gain: AtomicU32,
    pub pan: AtomicU32,
    /// 每通道实时响度峰值（振幅，0..≈1；超过 1 表示接近/超过削波）。
    ///
    /// 由对应通道音频线程在每个渲染块测量并写入（读取上一块峰值做衰减，
    /// 再与当前块峰值取大，形成回落型 VU），UI 线程经共享句柄无锁读取。
    /// 索引 = MIDI 通道号（0..channel_count）。
    pub peak: AtomicU32,
}

/// 每渲染块峰值衰减系数：用于让电平表平滑回落（约数百毫秒级）。
const PEAK_DECAY: f32 = 0.90;

/// 计算缓冲中样本绝对值的最大值（峰值振幅）。
fn max_abs(buf: &[f32]) -> f32 {
    let mut p = 0.0f32;
    for &s in buf {
        let a = s.abs();
        if a > p {
            p = a;
        }
    }
    p
}

/// 对立体声交织缓冲 `[L, R, L, R, …]` 施加音频域增益与等功率声像。
///
/// - `gain`：线性增益（1.0 = 0 dB），负数按 0 处理。
/// - `pan`：声像，-1=全左、0=居中、1=全右（等功率分配）。
fn apply_channel_mix(buf: &mut [f32], gain: f32, pan: f32) {
    let gain = gain.max(0.0);
    let pan = pan.clamp(-1.0, 1.0);
    let left = ((1.0 - pan) * 0.5f32).sqrt() * gain;
    let right = ((1.0 + pan) * 0.5f32).sqrt() * gain;
    for i in (0..buf.len()).step_by(2) {
        buf[i] *= left;
        if i + 1 < buf.len() {
            buf[i + 1] *= right;
        }
    }
}

/// A realtime MIDI synthesizer using an audio device for output.
pub struct RealtimeSynth {
    data: Option<RealtimeSynthThreadSharedData>,
    stream_owner: Option<thread::JoinHandle<()>>,
    join_handles: Vec<thread::JoinHandle<()>>,

    stats: RealtimeSynthStats,

    stream_params: AudioStreamParams,

    /// 每通道音频域混音状态（增益/声像），索引 = MIDI 通道号。
    channel_mix: Arc<Vec<ChannelMix>>,

    /// 主输出实时响度峰值（共享句柄，渲染管线写入、UI 经 `clone_master_peak` 读取）。
    master_peak: Arc<AtomicU32>,

    /// 自愈重定向失败通知（err_fn 触发 RestartSelf 失败时由 stream owner 线程发送）
    recovery_rx: crossbeam_channel::Receiver<StreamRestartError>,
}

enum StreamCommand {
    Pause(crossbeam_channel::Sender<Result<(), PauseStreamError>>),
    Resume(crossbeam_channel::Sender<Result<(), PlayStreamError>>),
    /// 重定向音频流到系统默认输出设备（同步等待结果）。
    /// Lumino vendored 扩展。
    Restart(crossbeam_channel::Sender<Result<(), StreamRestartError>>),
    /// 音频流自愈：err_fn 检测到设备不可用时触发（无需回复，
    /// 失败通过 recovery 通道通知上层）。Lumino vendored 扩展。
    RestartSelf,
    Shutdown,
}

impl RealtimeSynth {
    /// Initializes a new realtime synthesizer using the default config and
    /// the default audio output.
    pub fn open_with_all_defaults() -> Result<Self, RealtimeSynthError> {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .ok_or(RealtimeSynthError::NoOutputDevice)?;
        if let Ok(name) = device.name() {
            println!("Output device: {name}");
        }

        let stream_config = device.default_output_config()?;

        RealtimeSynth::open(Default::default(), &device, stream_config)
    }

    /// Initializes as new realtime synthesizer using a given config and
    /// the default audio output.
    ///
    /// See the `XSynthRealtimeConfig` documentation for the available options.
    pub fn open_with_default_output(
        config: XSynthRealtimeConfig,
    ) -> Result<Self, RealtimeSynthError> {
        let host = cpal::default_host();

        let device = host
            .default_output_device()
            .ok_or(RealtimeSynthError::NoOutputDevice)?;
        if let Ok(name) = device.name() {
            println!("Output device: {name}");
        }

        let stream_config = device.default_output_config()?;

        RealtimeSynth::open(config, &device, stream_config)
    }

    /// Initializes a new realtime synthesizer using a given config and a
    /// specified audio output device.
    ///
    /// See the `XSynthRealtimeConfig` documentation for the available options.
    /// See the `cpal` crate documentation for the `device` and `stream_config` parameters.
    pub fn open(
        config: XSynthRealtimeConfig,
        device: &Device,
        stream_config: SupportedStreamConfig,
    ) -> Result<Self, RealtimeSynthError> {
        let sample_rate = stream_config.sample_rate().0;
        let stream_params = AudioStreamParams::new(sample_rate, stream_config.channels().into());
        let channel_pool = build_channel_pool(config.multithreading)?;
        let channel_count = channel_count(config.format);

        // 每通道音频域混音状态（增益/声像），索引 = MIDI 通道号。
        // 同一 Arc 同时由 RealtimeSynth（UI 写）与各通道线程（读）共享。
        let channel_mix = Arc::new(
            (0..channel_count)
                .map(|_| ChannelMix {
                    gain: AtomicU32::new(1.0f32.to_bits()),
                    pan: AtomicU32::new(0.0f32.to_bits()),
                    peak: AtomicU32::new(0.0f32.to_bits()),
                })
                .collect::<Vec<_>>(),
        );

        // 主输出实时响度峰值（与通道峰值同语义），由渲染管线汇总后写入。
        let master_peak = Arc::new(AtomicU32::new(0.0f32.to_bits()));

        let PreparedRealtimeChannels {
            channel_stats,
            senders,
            command_senders,
            admission_budget,
            join_handles,
            output_receiver,
        } = prepare_channels(
            channel_count,
            config.channel_init_options,
            stream_params,
            channel_pool,
            config.format,
            channel_mix.clone(),
        )?;

        let stats = RealtimeSynthStats::new();
        // 硬上限（量程）：`None`/`0` 为自动模式，使用默认 10000。
        let hard_max_voices = match config.global_max_voices {
            Some(n) if n > 0 => n,
            _ => DEFAULT_HARD_MAX_VOICES,
        };
        let gate = EmergencyGate::new();
        let render = build_render_pipe(
            stream_params,
            channel_count,
            command_senders,
            output_receiver,
            channel_stats,
            admission_budget,
            &stats,
            master_peak.clone(),
            hard_max_voices,
            config.voice_target_ratio,
            config.soft_nps_gate,
            gate.clone(),
        );
        let render_size = calculate_render_size(sample_rate, config.render_window_ms).max(1);
        let cushion_samples =
            calculate_render_size(sample_rate, config.cushion_ms).max(render_size);
        let buffered = Arc::new(Mutex::new(
            BufferedRenderer::new(render, stream_params, render_size, cushion_samples)
                .map_err(RealtimeSynthError::BufferedRendererThreadSpawn)?,
        ));
        let (stream_control, stream_owner, recovery_rx) =
            spawn_stream_thread(device.clone(), stream_config, buffered.clone())?;

        let max_nps = Arc::new(ReadWriteAtomicU64::new(config.max_nps));

        Ok(Self {
            data: Some(RealtimeSynthThreadSharedData {
                buffered_renderer: buffered,

                event_senders: RealtimeEventSender::new(
                    senders,
                    max_nps,
                    config.ignore_range,
                    gate,
                )
                .map_err(RealtimeSynthError::EventSenderInit)?,
                stream_control,
            }),
            stream_owner: Some(stream_owner),
            join_handles,

            stats,
            stream_params,
            channel_mix,
            master_peak,
            recovery_rx,
        })
    }

    /// Sends a SynthEvent to the realtime synthesizer.
    ///
    /// See the `SynthEvent` documentation for more information.
    pub fn send_event(&mut self, event: SynthEvent) {
        let data = self.data.as_mut().unwrap();
        data.event_senders.send_event(event);
    }

    /// Sends a u32 event to the realtime synthesizer.
    pub fn send_event_u32(&mut self, event: u32) {
        let data = self.data.as_mut().unwrap();
        data.event_senders.send_event_u32(event);
    }

    /// Returns a reference to the event sender of the realtime synthesizer.
    /// This can be used to clone the sender so it can be passed in threads.
    ///
    /// See the `RealtimeEventSender` documentation for more information
    /// on how to use.
    pub fn get_sender_ref(&self) -> &RealtimeEventSender {
        let data = self.data.as_ref().unwrap();
        &data.event_senders
    }

    /// Returns a mutable reference the event sender of the realtime synthesizer.
    /// This can be used to modify its parameters (eg. ignore range).
    /// Please note that each clone will store its own distinct parameters.
    ///
    /// See the `RealtimeEventSender` documentation for more information
    /// on how to use.
    pub fn get_sender_mut(&mut self) -> &mut RealtimeEventSender {
        let data = self.data.as_mut().unwrap();
        &mut data.event_senders
    }

    /// Returns the statistics reader of the realtime synthesizer.
    ///
    /// See the `RealtimeSynthStatsReader` documentation for more information
    /// on how to use.
    pub fn get_stats(&self) -> RealtimeSynthStatsReader {
        let data = self.data.as_ref().unwrap();
        let buffered_stats = data.buffered_renderer.lock().unwrap().get_buffer_stats();

        RealtimeSynthStatsReader::new(self.stats.clone(), buffered_stats)
    }

    /// Returns the stream parameters of the audio output device.
    pub fn stream_params(&self) -> AudioStreamParams {
        self.stream_params
    }

    /// 设置某 MIDI 通道的音频域增益（线性，1.0 = 0 dB）。
    ///
    /// 由 UI 线程调用；对应通道的音频线程在下一渲染块读取并平滑应用。
    /// 越界通道静默忽略。
    pub fn set_channel_gain(&self, channel: u8, gain: f32) {
        if let Some(m) = self.channel_mix.get(channel as usize) {
            m.gain.store(gain.max(0.0).to_bits(), Ordering::Relaxed);
        }
    }

    /// 设置某 MIDI 通道的音频域声像（-1..1，0 = 居中）。
    pub fn set_channel_pan(&self, channel: u8, pan: f32) {
        if let Some(m) = self.channel_mix.get(channel as usize) {
            m.pan
                .store(pan.clamp(-1.0, 1.0).to_bits(), Ordering::Relaxed);
        }
    }

    /// 获取混音参数共享句柄（重建稳定的 `Arc<Vec<ChannelMix>>` 克隆引用）。
    ///
    /// 上层（如 lumino `XSynth` 后端）借此在 `RealtimeSynth` 之外设置每通道增益/声像，
    /// 数据流与 `sender_shared` 一致：句柄本身（外层 `Arc`）稳定，
    /// 重建时替换为新的内层 `Vec<ChannelMix>`，已创建的连接自动跟随。
    pub fn clone_channel_mix(&self) -> Arc<Vec<ChannelMix>> {
        Arc::clone(&self.channel_mix)
    }

    /// 获取主输出实时响度峰值的共享句柄（重建稳定的 `Arc<AtomicU32>` 克隆引用）。
    ///
    /// 上层（如 lumino `XSynth` 后端）借此在 `RealtimeSynth` 之外读取主输出电平，
    /// 与 `clone_channel_mix` 同生命周期语义：句柄外层 `Arc` 稳定，重建时跟随新管线。
    pub fn clone_master_peak(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.master_peak)
    }

    /// Pauses the playback of the audio output device.
    pub fn pause(&mut self) -> Result<(), PauseStreamError> {
        let data = self.data.as_ref().unwrap();
        let (sender, receiver) = bounded(1);
        if data
            .stream_control
            .send(StreamCommand::Pause(sender))
            .is_err()
        {
            return Err(PauseStreamError::DeviceNotAvailable);
        }
        receiver
            .recv()
            .unwrap_or(Err(PauseStreamError::DeviceNotAvailable))
    }

    /// Resumes the playback of the audio output device.
    pub fn resume(&mut self) -> Result<(), PlayStreamError> {
        let data = self.data.as_ref().unwrap();
        let (sender, receiver) = bounded(1);
        if data
            .stream_control
            .send(StreamCommand::Resume(sender))
            .is_err()
        {
            return Err(PlayStreamError::DeviceNotAvailable);
        }
        receiver
            .recv()
            .unwrap_or(Err(PlayStreamError::DeviceNotAvailable))
    }

    /// Changes the length of the buffer reader.
    pub fn set_buffer(&self, render_window_ms: f64) {
        let data = self.data.as_ref().unwrap();
        let sample_rate = self.stream_params.sample_rate;
        let size = calculate_render_size(sample_rate, render_window_ms);
        data.buffered_renderer.lock().unwrap().set_render_size(size);
    }

    /// 将音频流重定向到系统默认输出设备（合成管线保持不变）。
    ///
    /// Lumino vendored 扩展：音频设备被拔出/更换后调用。
    /// 仅当新设备的采样率/声道数/采样格式与当前一致时可直接重定向；
    /// 否则返回 [`StreamRestartError::ConfigChanged`]，调用方应重建整个合成管线。
    pub fn restart_stream(&self) -> Result<(), StreamRestartError> {
        let data = self.data.as_ref().unwrap();
        let (sender, receiver) = bounded(1);
        if data
            .stream_control
            .send(StreamCommand::Restart(sender))
            .is_err()
        {
            return Err(StreamRestartError::StreamThreadDown);
        }
        receiver
            .recv()
            .unwrap_or(Err(StreamRestartError::StreamThreadDown))
    }

    /// 检查自愈重定向（err_fn 触发）是否失败。
    ///
    /// Lumino vendored 扩展：返回 `Some` 表示音频流自愈失败
    /// （通常因新设备参数与管线不一致），需要上层介入重建管线。
    pub fn poll_recovery_error(&self) -> Option<StreamRestartError> {
        self.recovery_rx.try_recv().ok()
    }
}

fn build_channel_pool(
    thread_count: ThreadCount,
) -> Result<Option<Arc<rayon::ThreadPool>>, RealtimeSynthError> {
    Ok(match thread_count {
        ThreadCount::None => None,
        ThreadCount::Auto => Some(Arc::new(rayon::ThreadPoolBuilder::new().build()?)),
        ThreadCount::Manual(threads) => Some(Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()?,
        )),
    })
}

fn channel_count(format: SynthFormat) -> u32 {
    match format {
        SynthFormat::Midi => 16,
        SynthFormat::Custom { channels } => channels,
    }
}

fn prepare_channels(
    channel_count: u32,
    init_options: xsynth_core::channel::ChannelInitOptions,
    stream_params: AudioStreamParams,
    channel_pool: Option<Arc<rayon::ThreadPool>>,
    format: SynthFormat,
    channel_mix: Arc<Vec<ChannelMix>>,
) -> Result<PreparedRealtimeChannels, RealtimeSynthError> {
    let (output_sender, output_receiver) = bounded::<Vec<f32>>(channel_count as usize);

    let mut channel_stats = Vec::new();
    let mut senders = Vec::new();
    let mut command_senders = Vec::new();
    let mut join_handles = Vec::new();
    // 全局入场预算池：初始 0（首块渲染前由管道刷新为真实值）。
    let admission_budget = Arc::new(AtomicI64::new(0));

    for i in 0..channel_count {
        let channel = VoiceChannel::new(init_options, stream_params, channel_pool.clone());
        channel_stats.push(channel.get_channel_stats());

        let (event_sender, event_receiver) = unbounded();
        senders.push(event_sender);

        let (command_sender, command_receiver) = bounded::<ChannelCommand>(1);
        command_senders.push(command_sender);

        let output_sender = output_sender.clone();
        let join_handle = spawn_channel_thread(
            channel,
            i as u8,
            channel_mix.clone(),
            event_receiver,
            command_receiver,
            output_sender,
            admission_budget.clone(),
        )?;
        join_handles.push(join_handle);
    }

    if format == SynthFormat::Midi {
        let _ = senders[9].send(ChannelEvent::Config(ChannelConfigEvent::SetPercussionMode(
            true,
        )));
    }

    Ok(PreparedRealtimeChannels {
        channel_stats,
        senders,
        command_senders,
        admission_budget,
        join_handles,
        output_receiver,
    })
}

fn spawn_channel_thread(
    mut channel: VoiceChannel,
    channel_index: u8,
    mix: Arc<Vec<ChannelMix>>,
    event_receiver: crossbeam_channel::Receiver<ChannelEvent>,
    command_receiver: crossbeam_channel::Receiver<ChannelCommand>,
    output_sender: crossbeam_channel::Sender<Vec<f32>>,
    admission_budget: Arc<AtomicI64>,
) -> Result<thread::JoinHandle<()>, RealtimeSynthError> {
    thread::Builder::new()
        .name("xsynth_channel_handler".to_string())
        .spawn(move || {
            // 当前增益/声像：向 UI 设定的目标平滑逼近，避免拖动产生 zipper noise。
            let mut cur_gain = 1.0f32;
            let mut cur_pan = 0.0f32;

            // 声部上限语义（连续播放优先，抢旧不丢新）：
            // - **全局预算池**（软目标 − 当前总声部数，管道每块刷新）：预算内
            //   NoteOn 直接发声；预算耗尽时先**抢占一个旧声部**（T1 释放中最轻 →
            //   T2 最轻，1ms 短淡出）再发声——新音符永远不丢、不推迟、不补发，
            //   听感连续；超上限的代价由"最不重要的旧声部被硬切"承担。
            //   这正是"顶到上限有硬切、但音乐不断"的语义（曾经错误的
            //   defer/drop 实现会把音乐切成碎片并让音频落后进度条）。
            // - **NoteOff 按"发声计数"正确配对**：键上仍有在响音符时直通。
            // - 其他事件（CC/PB/Program/Config）永远直通。
            // 每键"已下发且尚未收到 NoteOff"的音符数（FIFO 配对基准）。
            let mut sounding = [0u32; 128];
            const DRAIN_CAP: usize = 4096;
            // 预算取用：`fetch_sub` 返回旧值，>0 表示取到额度；取不到则回补（净零）。
            let try_acquire = |budget: &AtomicI64| -> bool {
                if budget.fetch_sub(1, Ordering::Relaxed) > 0 {
                    true
                } else {
                    budget.fetch_add(1, Ordering::Relaxed);
                    false
                }
            };
            let admit = |channel: &mut VoiceChannel, sounding: &mut [u32; 128]| {
                let mut drained = 0;
                while drained < DRAIN_CAP {
                    let Ok(event) = event_receiver.try_recv() else {
                        break;
                    };
                    drained += 1;
                    match event {
                        ChannelEvent::Audio(ChannelAudioEvent::NoteOn { key, vel }) => {
                            if !try_acquire(&admission_budget) {
                                // 预算耗尽：抢一个最不重要的旧声部再发声，
                                // 保证"每个新音符都响"，避免断续与补发旧音。
                                channel.steal_voices(1);
                            }
                            channel.process_event(ChannelEvent::Audio(ChannelAudioEvent::NoteOn {
                                key,
                                vel,
                            }));
                            sounding[key as usize] += 1;
                        }
                        ChannelEvent::Audio(ChannelAudioEvent::NoteOff { key }) => {
                            let k = key as usize;
                            if k < sounding.len() && sounding[k] > 0 {
                                // 该键仍有在响音符：NoteOff 必须下发，释放对应声部。
                                channel.process_event(ChannelEvent::Audio(
                                    ChannelAudioEvent::NoteOff { key },
                                ));
                                sounding[k] -= 1;
                            } else {
                                channel.process_event(ChannelEvent::Audio(
                                    ChannelAudioEvent::NoteOff { key },
                                ));
                            }
                        }
                        ChannelEvent::Audio(ChannelAudioEvent::AllNotesOff) => {
                            sounding.fill(0);
                            channel
                                .process_event(ChannelEvent::Audio(ChannelAudioEvent::AllNotesOff));
                        }
                        ChannelEvent::Audio(ChannelAudioEvent::AllNotesKilled) => {
                            sounding.fill(0);
                            channel.process_event(ChannelEvent::Audio(
                                ChannelAudioEvent::AllNotesKilled,
                            ));
                        }
                        ChannelEvent::Audio(ChannelAudioEvent::SystemReset) => {
                            sounding.fill(0);
                            channel
                                .process_event(ChannelEvent::Audio(ChannelAudioEvent::SystemReset));
                        }
                        other => channel.process_event(other),
                    }
                }
            };

            loop {
                admit(&mut channel, &mut sounding);
                let command = match command_receiver.recv() {
                    Ok(command) => command,
                    Err(_) => break,
                };
                admit(&mut channel, &mut sounding);

                let mut vec = match command {
                    // 全局治理：只抢占声部，不渲染、不回送音频块。
                    ChannelCommand::Steal(count) => {
                        channel.steal_voices(count);
                        continue;
                    }
                    // L2 重度治理：硬移除（跳过淡出）。
                    ChannelCommand::HardSteal(count) => {
                        channel.steal_voices_hard(count);
                        continue;
                    }
                    // L4 看门狗：每键仅保留最新 N 组，立即释放其余。
                    ChannelCommand::WatchdogReset(keep) => {
                        channel.trim_to_newest(keep);
                        continue;
                    }
                    ChannelCommand::Render(vec) => vec,
                };

                channel.read_samples(&mut vec);
                // 音频域混音：立体声交织缓冲施加增益 + 等功率声像。
                let tgt_gain =
                    f32::from_bits(mix[channel_index as usize].gain.load(Ordering::Relaxed))
                        .max(0.0);
                let tgt_pan =
                    f32::from_bits(mix[channel_index as usize].pan.load(Ordering::Relaxed))
                        .clamp(-1.0, 1.0);
                cur_gain += (tgt_gain - cur_gain) * 0.25;
                cur_pan += (tgt_pan - cur_pan) * 0.25;
                apply_channel_mix(&mut vec, cur_gain, cur_pan);
                // 实时响度峰值：读取上一块峰值做衰减，与当前块峰值取大（回落型 VU）。
                let peak_slot = &mix[channel_index as usize].peak;
                let prev = f32::from_bits(peak_slot.load(Ordering::Relaxed));
                let new_peak = prev * PEAK_DECAY;
                let block_peak = max_abs(&vec);
                let peak = if block_peak > new_peak {
                    block_peak
                } else {
                    new_peak
                };
                peak_slot.store(peak.to_bits(), Ordering::Relaxed);
                if output_sender.send(vec).is_err() {
                    break;
                }
            }
        })
        .map_err(RealtimeSynthError::ChannelThreadSpawn)
}

#[allow(clippy::too_many_arguments)]
fn build_render_pipe(
    stream_params: AudioStreamParams,
    channel_count: u32,
    command_senders: Vec<crossbeam_channel::Sender<ChannelCommand>>,
    output_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    admission_budget: Arc<AtomicI64>,
    stats: &RealtimeSynthStats,
    master_peak: Arc<AtomicU32>,
    hard_max_voices: usize,
    voice_target_ratio: f64,
    soft_nps_gate: bool,
    gate: Arc<EmergencyGate>,
) -> FunctionAudioPipe<impl FnMut(&mut [f32]) + Send> {
    let mut vec_cache: VecDeque<Vec<f32>> = VecDeque::new();
    for _ in 0..channel_count {
        vec_cache.push_front(Vec::new());
    }

    let total_voice_count = stats.voice_count.clone();
    let sample_rate = stream_params.sample_rate as f64;
    let output_channels = (stream_params.channels.count() as f64).max(1.0);

    // 声部治理器：运行目标固定 = ratio × 硬上限（不随负载漂移，防自激）。
    let mut governor = Governor::new(hard_max_voices, voice_target_ratio);

    FunctionAudioPipe::new(stream_params, move |out| {
        let block_start = Instant::now();

        for sender in &command_senders {
            let mut buf = vec_cache.pop_front().unwrap();
            prepapre_cache_vec(&mut buf, out.len(), 0.0);
            sender.send(ChannelCommand::Render(buf)).unwrap();
        }

        for _ in 0..channel_count {
            let buf = output_receiver.recv().unwrap();
            sum_simd(&buf, out);
            vec_cache.push_front(buf);
        }

        // 主输出实时响度峰值：汇总后的 `out` 即最终混音（限幅前），
        // 同通道峰值做衰减 + 取大。
        let prev = f32::from_bits(master_peak.load(Ordering::Relaxed));
        let new_peak = prev * PEAK_DECAY;
        let block_peak = max_abs(out);
        let peak = if block_peak > new_peak {
            block_peak
        } else {
            new_peak
        };
        master_peak.store(peak.to_bits(), Ordering::Relaxed);

        let total_voices: u64 = channel_stats.iter().map(|c| c.voice_count()).sum();
        total_voice_count.store(total_voices, Ordering::SeqCst);

        // 负载 = 本块渲染耗时 / 块时长（与采样率无关的可比量）。
        let block_secs = out.len() as f64 / (sample_rate * output_channels);
        let load = if block_secs > 0.0 {
            block_start.elapsed().as_secs_f64() / block_secs
        } else {
            0.0
        };

        let action = governor.update(load, total_voices, soft_nps_gate);
        gate.set(action.gate_active, action.gate_rate);

        // 刷新全局入场预算池："软目标 - 当前总声部数"（由通道侧原子取用），
        // 下一块各通道据此接纳 NoteOn；单通道文件可独享整个预算。
        let available_voice_budget = (governor.v_soft() - total_voices as f64).max(0.0) as i64;
        admission_budget.store(available_voice_budget, Ordering::Relaxed);

        // 跨通道全局治理：从"声部最多的通道"按比例分摊抢占。
        // `Steal`/`HardSteal` 命令不产生音频块，通道处理完即继续等待
        // 下一块渲染命令，因此不会破坏本块的通道同步。
        //
        // 超目标即硬移除（数量已由治理器限制在总声部数的 1/4 以内），
        // 这里只做一个防御性上限，避免异常值造成块内长任务。
        let want = action.steal.max(action.hard_steal).min(4096);
        if want > 0 {
            let mut deficit = want as u64;
            let mut counts: Vec<u64> = channel_stats.iter().map(|c| c.voice_count()).collect();
            let mut order: Vec<usize> = (0..channel_count as usize).collect();
            order.sort_by_key(|&i| std::cmp::Reverse(counts[i]));
            for i in order {
                if deficit == 0 {
                    break;
                }
                let take = counts[i].min(deficit);
                if take == 0 {
                    continue;
                }
                let command = if action.hard_steal > 0 {
                    ChannelCommand::HardSteal(take as usize)
                } else {
                    ChannelCommand::Steal(take as usize)
                };
                if command_senders[i].send(command).is_ok() {
                    counts[i] -= take;
                    deficit -= take;
                }
            }
        }

        // L4 看门狗：每键仅保留最新 K 组 + 重置治理基线，避免带着过载
        // 历史继续决策（自愈，防"顶破缓冲后永久损坏"）。
        if let Some(keep) = action.watchdog_keep {
            for sender in &command_senders {
                sender.send(ChannelCommand::WatchdogReset(keep)).ok();
            }
            gate.set(false, action.gate_rate);
            governor.reset_after_watchdog();
        }

        // 治理诊断（受 XSYNTH_GOV_DEBUG 控制）：异常时 1s 一次，正常时 5s 一次，
        // 始终输出 V/soft/def，便于观察撞墙与积压情况。
        if std::env::var_os("XSYNTH_GOV_DEBUG").is_some() {
            static LAST_LOG: AtomicU64 = AtomicU64::new(0);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let interval = if governor.level > 0 { 1 } else { 5 };
            if now.saturating_sub(LAST_LOG.load(Ordering::Relaxed)) >= interval {
                LAST_LOG.store(now, Ordering::Relaxed);
                // 诊断：声部最多的 3 个通道 + 各通道最大积压 NoteOn（定位病态分布）。
                let mut counts: Vec<(usize, u64)> = channel_stats
                    .iter()
                    .enumerate()
                    .map(|(i, c)| (i, c.voice_count()))
                    .collect();
                counts.sort_by_key(|&(_, v)| std::cmp::Reverse(v));
                let top = counts
                    .iter()
                    .take(3)
                    .map(|(i, v)| format!("c{i}={v}"))
                    .collect::<Vec<_>>()
                    .join(" ");
                eprintln!(
                    "[GOV] L{} load={:.3} V={} soft={:.0} steal={} hard={} gate={} top=[{}]",
                    governor.level,
                    governor.load_ema(),
                    total_voices,
                    governor.v_soft(),
                    action.steal,
                    action.hard_steal,
                    action.gate_active,
                    top,
                );
            }
        }
    })
}

fn build_output_stream(
    device: &Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
    restart_notify: crossbeam_channel::Sender<StreamCommand>,
) -> Result<Stream, RealtimeSynthError> {
    match stream_config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_output_stream_for::<f32>(device, stream_config, buffered, restart_notify)
        }
        cpal::SampleFormat::I16 => {
            build_output_stream_for::<i16>(device, stream_config, buffered, restart_notify)
        }
        cpal::SampleFormat::U16 => {
            build_output_stream_for::<u16>(device, stream_config, buffered, restart_notify)
        }
        _ => Err(RealtimeSynthError::UnsupportedSampleFormat(
            stream_config.sample_format(),
        )),
    }
}

fn build_output_stream_for<T: SizedSample + ConvertSample>(
    device: &Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
    restart_notify: crossbeam_channel::Sender<StreamCommand>,
) -> Result<Stream, RealtimeSynthError> {
    let err_fn = move |err| {
        eprintln!("an error occurred on stream: {err}");
        // Lumino vendored 扩展：设备被拔出/更换（DeviceNotAvailable）时，
        // 通知 stream owner 线程将流重定向到系统默认输出设备。
        // err_fn 在 cpal 音频线程上执行：仅发送非阻塞消息后返回，
        // 线程随后退出（cpal 错误后 Break），owner 线程 drop 旧流不会死锁。
        if matches!(err, cpal::StreamError::DeviceNotAvailable) {
            let _ = restart_notify.send(StreamCommand::RestartSelf);
        }
    };
    let mut output_vec = Vec::new();
    let mut limiter = VolumeLimiter::new(stream_config.channels());

    Ok(device.build_output_stream(
        &stream_config.into(),
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            output_vec.resize(data.len(), 0.0);
            buffered.lock().unwrap().read(&mut output_vec);
            for (i, s) in limiter.limit_iter(output_vec.drain(0..)).enumerate() {
                data[i] = ConvertSample::from_f32(s);
            }
        },
        err_fn,
        None,
    )?)
}

fn spawn_stream_thread(
    device: Device,
    stream_config: SupportedStreamConfig,
    buffered: Arc<Mutex<BufferedRenderer>>,
) -> Result<
    (
        crossbeam_channel::Sender<StreamCommand>,
        thread::JoinHandle<()>,
        crossbeam_channel::Receiver<StreamRestartError>,
    ),
    RealtimeSynthError,
> {
    let (command_sender, command_receiver) = unbounded();
    let (ready_sender, ready_receiver) = bounded(1);
    let (recovery_sender, recovery_receiver) = unbounded();
    // 供流构建与 restart 使用的命令发送器（闭包 move 克隆，原始 sender 返回给调用方）
    let command_sender_for_stream = command_sender.clone();
    let join_handle = thread::Builder::new()
        .name("xsynth_stream_owner".to_string())
        .spawn(move || {
            let mut stream = match build_output_stream(
                &device,
                stream_config.clone(),
                buffered.clone(),
                command_sender_for_stream.clone(),
            ) {
                Ok(stream) => stream,
                Err(err) => {
                    let _ = ready_sender.send(Err(err));
                    return;
                }
            };
            if let Err(err) = stream.play() {
                let _ = ready_sender.send(Err(err.into()));
                return;
            }
            if ready_sender.send(Ok(())).is_err() {
                return;
            }

            while let Ok(command) = command_receiver.recv() {
                match command {
                    StreamCommand::Pause(reply) => {
                        let _ = reply.send(stream.pause());
                    }
                    StreamCommand::Resume(reply) => {
                        let _ = reply.send(stream.play());
                    }
                    StreamCommand::Restart(reply) => {
                        let result = restart_stream(
                            &mut stream,
                            &stream_config,
                            &buffered,
                            &command_sender_for_stream,
                        );
                        let _ = reply.send(result);
                    }
                    StreamCommand::RestartSelf => {
                        if let Err(err) = restart_stream(
                            &mut stream,
                            &stream_config,
                            &buffered,
                            &command_sender_for_stream,
                        ) {
                            // 自愈失败（通常是设备参数变化）：通知上层重建管线
                            eprintln!("xsynth-realtime: audio stream restart failed: {err}");
                            let _ = recovery_sender.send(err);
                        }
                    }
                    StreamCommand::Shutdown => break,
                }
            }
        })
        .map_err(RealtimeSynthError::StreamThreadSpawn)?;

    match ready_receiver.recv() {
        Ok(Ok(())) => Ok((command_sender, join_handle, recovery_receiver)),
        Ok(Err(err)) => {
            let _ = join_handle.join();
            Err(err)
        }
        Err(_) => {
            let _ = join_handle.join();
            Err(RealtimeSynthError::StreamThreadInit)
        }
    }
}

/// 将音频流重定向到系统默认输出设备。
///
/// Lumino vendored 扩展：在 stream owner 线程内执行。
/// 先校验新设备参数（采样率/声道数/采样格式）与当前管线参数一致，
/// 再 drop 旧流并构建新流；合成管线（BufferedRenderer）保持不变。
/// 校验失败时旧流保持原样（不中断），由上层决定是否重建管线。
fn restart_stream(
    stream: &mut Stream,
    old_config: &SupportedStreamConfig,
    buffered: &Arc<Mutex<BufferedRenderer>>,
    restart_notify: &crossbeam_channel::Sender<StreamCommand>,
) -> Result<(), StreamRestartError> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or(StreamRestartError::NoDefaultDevice)?;

    let new_config = device
        .default_output_config()
        .map_err(StreamRestartError::DefaultOutputConfig)?;

    // 管线参数一致性校验：任一变化都会导致渲染数据语义错乱（变调/声道错乱），
    // 必须由上层重建整个合成管线。旧流保持原样，等待上层决定。
    if new_config.sample_rate() != old_config.sample_rate()
        || new_config.channels() != old_config.channels()
        || new_config.sample_format() != old_config.sample_format()
    {
        return Err(StreamRestartError::ConfigChanged(format!(
            "sample_rate {}->{} Hz, channels {}->{}, format {:?}->{:?}",
            old_config.sample_rate().0,
            new_config.sample_rate().0,
            old_config.channels(),
            new_config.channels(),
            old_config.sample_format(),
            new_config.sample_format(),
        )));
    }

    // 构建新流（新设备）。若失败，旧流不受影响。
    let new_stream = match build_output_stream(
        &device,
        new_config,
        buffered.clone(),
        restart_notify.clone(),
    ) {
        Ok(stream) => stream,
        Err(RealtimeSynthError::BuildStream(err)) => return Err(StreamRestartError::Build(err)),
        Err(RealtimeSynthError::UnsupportedSampleFormat(fmt)) => {
            return Err(StreamRestartError::UnsupportedSampleFormat(fmt))
        }
        Err(other) => {
            return Err(StreamRestartError::ConfigChanged(format!(
                "failed to build output stream: {other}"
            )))
        }
    };
    new_stream.play().map_err(StreamRestartError::Play)?;

    // 替换持有的流：旧流在此 drop（等待其内部线程退出）。
    // cpal 错误后音频线程已退出（Break），join 立即返回，无死锁。
    *stream = new_stream;
    eprintln!("xsynth-realtime: audio stream redirected to default output device");

    Ok(())
}

impl Drop for RealtimeSynth {
    fn drop(&mut self) {
        let data = self.data.take().unwrap();
        let _ = data.stream_control.send(StreamCommand::Shutdown);
        drop(data);
        if let Some(handle) = self.stream_owner.take() {
            if handle.join().is_err() {
                eprintln!("xsynth-realtime: stream owner thread panicked during shutdown");
            }
        }
        for handle in self.join_handles.drain(..) {
            if handle.join().is_err() {
                eprintln!("xsynth-realtime: channel handler thread panicked during shutdown");
            }
        }
    }
}

trait ConvertSample: SizedSample {
    fn from_f32(s: f32) -> Self;
}

impl ConvertSample for f32 {
    fn from_f32(s: f32) -> Self {
        s
    }
}

impl ConvertSample for i16 {
    fn from_f32(s: f32) -> Self {
        (s * i16::MAX as f32) as i16
    }
}

impl ConvertSample for u16 {
    fn from_f32(s: f32) -> Self {
        ((s * u16::MAX as f32) as i32 + i16::MIN as i32) as u16
    }
}

fn calculate_render_size(sample_rate: u32, buffer_ms: f64) -> usize {
    (sample_rate as f64 * buffer_ms / 1000.0) as usize
}

#[cfg(test)]
mod tests {
    use super::{apply_channel_mix, RealtimeSynth};

    #[test]
    fn realtime_synth_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RealtimeSynth>();
    }

    #[test]
    fn apply_channel_mix_gain_and_pan() {
        // 立体声交织 [L,R,L,R]；等功率声像：居中为 sqrt(0.5)（-3dB）
        let mut buf = vec![1.0, 1.0, 1.0, 1.0];
        apply_channel_mix(&mut buf, 1.0, 0.0);
        let center = 0.5f32.sqrt();
        assert!(
            (buf[0] - center).abs() < 1e-5 && (buf[1] - center).abs() < 1e-5,
            "center pan must be equal-power: {buf:?}"
        );

        // 全左 → 右声道归零，左声道保持
        let mut buf = vec![1.0, 1.0, 1.0, 1.0];
        apply_channel_mix(&mut buf, 1.0, -1.0);
        assert!((buf[0] - 1.0).abs() < 1e-5, "全左时左声道应保持: {buf:?}");
        assert!(buf[1].abs() < 1e-5, "全左时右声道应归零: {buf:?}");

        // 全右 + 增益 2 → 左归零，右声道 = 2
        let mut buf = vec![1.0, 1.0, 1.0, 1.0];
        apply_channel_mix(&mut buf, 2.0, 1.0);
        assert!((buf[1] - 2.0).abs() < 1e-5, "全右时右声道应为增益: {buf:?}");
        assert!(buf[0].abs() < 1e-5, "全右时左声道应归零: {buf:?}");

        // 增益 0 → 全静音
        let mut buf = vec![1.0, 1.0, 1.0, 1.0];
        apply_channel_mix(&mut buf, 0.0, 0.5);
        assert!(
            buf.iter().all(|&v| v.abs() < 1e-5),
            "增益0应全静音: {buf:?}"
        );
    }
}
