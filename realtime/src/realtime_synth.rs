use std::{
    collections::VecDeque,
    io,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc, Mutex,
    },
    thread::{self},
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
    channel::{ChannelConfigEvent, ChannelEvent, VoiceChannel},
    channel_group::SynthFormat,
    effects::VolumeLimiter,
    helpers::{prepapre_cache_vec, sum_simd},
    AudioPipe, AudioStreamParams, FunctionAudioPipe,
};

use crate::{
    util::ReadWriteAtomicU64, RealtimeEventSender, SynthEvent, ThreadCount, XSynthRealtimeConfig,
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

struct PreparedRealtimeChannels {
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    senders: Vec<crossbeam_channel::Sender<ChannelEvent>>,
    command_senders: Vec<crossbeam_channel::Sender<Vec<f32>>>,
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
        let render = build_render_pipe(
            stream_params,
            channel_count,
            command_senders,
            output_receiver,
            channel_stats,
            &stats,
            master_peak.clone(),
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

                event_senders: RealtimeEventSender::new(senders, max_nps, config.ignore_range)
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

    for i in 0..channel_count {
        let channel = VoiceChannel::new(init_options, stream_params, channel_pool.clone());
        channel_stats.push(channel.get_channel_stats());

        let (event_sender, event_receiver) = unbounded();
        senders.push(event_sender);

        let (command_sender, command_receiver) = bounded::<Vec<f32>>(1);
        command_senders.push(command_sender);

        let output_sender = output_sender.clone();
        let join_handle = spawn_channel_thread(
            channel,
            i as u8,
            channel_mix.clone(),
            event_receiver,
            command_receiver,
            output_sender,
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
        join_handles,
        output_receiver,
    })
}

fn spawn_channel_thread(
    mut channel: VoiceChannel,
    channel_index: u8,
    mix: Arc<Vec<ChannelMix>>,
    event_receiver: crossbeam_channel::Receiver<ChannelEvent>,
    command_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    output_sender: crossbeam_channel::Sender<Vec<f32>>,
) -> Result<thread::JoinHandle<()>, RealtimeSynthError> {
    thread::Builder::new()
        .name("xsynth_channel_handler".to_string())
        .spawn(move || {
            // 当前增益/声像：向 UI 设定的目标平滑逼近，避免拖动产生 zipper noise。
            let mut cur_gain = 1.0f32;
            let mut cur_pan = 0.0f32;
            loop {
                channel.push_events_iter(event_receiver.try_iter());
                let mut vec = match command_receiver.recv() {
                    Ok(vec) => vec,
                    Err(_) => break,
                };
                channel.push_events_iter(event_receiver.try_iter());
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

fn build_render_pipe(
    stream_params: AudioStreamParams,
    channel_count: u32,
    command_senders: Vec<crossbeam_channel::Sender<Vec<f32>>>,
    output_receiver: crossbeam_channel::Receiver<Vec<f32>>,
    channel_stats: Vec<xsynth_core::channel::VoiceChannelStatsReader>,
    stats: &RealtimeSynthStats,
    master_peak: Arc<AtomicU32>,
) -> FunctionAudioPipe<impl FnMut(&mut [f32]) + Send> {
    let mut vec_cache: VecDeque<Vec<f32>> = VecDeque::new();
    for _ in 0..channel_count {
        vec_cache.push_front(Vec::new());
    }

    let total_voice_count = stats.voice_count.clone();

    FunctionAudioPipe::new(stream_params, move |out| {
        for sender in &command_senders {
            let mut buf = vec_cache.pop_front().unwrap();
            prepapre_cache_vec(&mut buf, out.len(), 0.0);
            sender.send(buf).unwrap();
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

        let total_voices = channel_stats.iter().map(|c| c.voice_count()).sum();
        total_voice_count.store(total_voices, Ordering::SeqCst);
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
        // 立体声交织 [L,R,L,R]；增益 1、声像居中 → 不变
        let mut buf = vec![1.0, 1.0, 1.0, 1.0];
        apply_channel_mix(&mut buf, 1.0, 0.0);
        assert!(
            (buf[0] - 1.0).abs() < 1e-5 && (buf[1] - 1.0).abs() < 1e-5,
            "居中增益1应不变: {buf:?}"
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
