//! B1：跨 voice SoA 批处理渲染（lane = voice）。
//!
//! # 架构
//!
//! `StereoBatchVoice` 在 spawn 期由 `StereoSampledVoiceSpawner` 直接构造：lane 状态
//! 就是该 voice 的**权威状态**，不再保留原链式生成器。渲染时 `VoiceChannel` 把连续的
//! 可批 voice 按运行时 SIMD 宽度（本机 AVX2 = 8）分块交给 [`render_batch_chunk`]：
//!
//! - 标量部分：逐 lane 推进采样（时间推进 + 回绕 + 抓取）、包络（7 阶段状态机）、增益；
//! - 向量部分：跨 voice 并行完成 biquad（DF1）。时间维串行依赖链 → 跨 voice 并行，
//!   这是 B0 决策门测出的结构性收益来源（biquad 阶段 4.8-6.4×）；
//! - 混音：`horizontal_add` 求和后一次 `+=` 进通道缓冲（与 B0 原型语义一致）。
//!
//! # 逐位等价口径（B1 决策门）
//!
//! - **单 lane 信号链逐位一致**：采样/包络/增益/biquad 的每一步运算顺序与标量实现
//!   完全相同，并用差分测试逐位验证（见本文件 tests）：
//!   `BatchLane::render_scalar` ≡ 真实链式生成器；内核单 lane ≡ `render_scalar`。
//! - 包络复刻真实实现的「8 帧组」机制：组内是否走 SIMD 曲线公式还是逐帧 scalar 公式，
//!   取决于组起点（`t + WIDTH - 1 >= end`），与真实实现完全一致，因此 concave/convex
//!   曲线在阶段末尾的取值切换点也逐位相同。
//! - **混音求和顺序不同**：批内 8 lane 经 `horizontal_add` 树形求和，与逐 voice 顺序
//!   累加在 f32 下不满足结合律 → 与 `LUMINO_BATCH=0` 存在 ulp 级差异（端到端由
//!   1e-6 容差 + RMS/峰值对照覆盖）；只有「单 lane 独占一个 batch」时逐位一致。
//!
//! # 开关
//!
//! `LUMINO_BATCH=1`（默认关）。关闭时 spawner 走原链路，渲染路径完全不变。

use std::sync::{Arc, OnceLock};

use simdeez::prelude::*;
use simdeez::simd_runtime_generate;
use xsynth_soundfonts::LoopMode;

use crate::effects::BiQuadFilter;
use crate::soundfont::Interpolator;
use crate::voice::{
    BufferSampler, BufferSamplers, EnvelopeControlData, EnvelopeParameters, EnvelopePart,
    EnvelopeStage, ReleaseType, Voice, VoiceControlData, VoiceGeneratorBase, VoiceSampleGenerator,
};

/// 运行时所有 SIMD 宽度（Scalar=1 … AVX512=16）的上界。
const MAX_LANES: usize = 16;
/// 启用批渲染所需的最小 SIMD 宽度（更低宽度下跨 voice 并行收益不足）。
const MIN_BATCH_WIDTH: usize = 8;

/// 批处理开关：`LUMINO_BATCH` 存在且不为 `0`/`false` 时启用（默认关）。
pub(crate) fn batch_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var_os("LUMINO_BATCH").is_some_and(|v| {
            let v = v.to_string_lossy();
            v != "0" && !v.eq_ignore_ascii_case("false")
        })
    })
}

/// 运行时 SIMD 宽度 = 一个 batch chunk 的 lane 数（本机 AVX2 = 8）。
pub(crate) fn batch_chunk_width() -> usize {
    static WIDTH: OnceLock<usize> = OnceLock::new();
    *WIDTH.get_or_init(|| {
        simd_runtime_generate!(
            fn width() -> usize {
                S::Vf32::WIDTH
            }
        );
        width()
    })
}

/// 批渲染是否可用：开关打开 + SIMD 宽度足够。
pub(crate) fn batching_supported() -> bool {
    batch_enabled() && batch_chunk_width() >= MIN_BATCH_WIDTH
}

/// spawn 期传入的静态参数（由 `StereoSampledVoiceSpawner` 填充）。
pub(crate) struct BatchLaneInit {
    pub speed_mult: f32,
    pub gain_l: f32,
    pub gain_r: f32,
    pub samples_l: Arc<[f32]>,
    pub samples_r: Arc<[f32]>,
    pub loop_mode: LoopMode,
    pub loop_offset: usize,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_stop: Option<usize>,
    pub interpolator: Interpolator,
    pub filter: Option<BiQuadFilter>,
    pub envelope: EnvelopeParameters,
    pub sample_rate: f32,
    pub group_len: u8,
    pub velocity: u8,
    pub exclusive_class: Option<u8>,
}

/// 一侧采样读取器状态（镜像 `SampleReader*` 的动态字段）。
struct SideReader {
    buffer: BufferSamplers,
    offset: usize,
    /// `SampleReaderLoopSustain` 的 `last`（最后一次未释放时的索引，含 offset）。
    last: usize,
    /// `loop_params.stop` 或缓冲长度（未设置 stop 时）。
    length: Option<usize>,
}

/// 包络曲线（标量版 `StageData`）。
#[derive(Clone, Copy)]
enum LaneCurve {
    /// `start + length * factor`（`SIMDLerper`）
    Lerp { start: f32, length: f32 },
    /// `length * (1 - factor)^8 + end`（`SIMDLerperConcave`）
    Concave { length: f32, end: f32 },
    /// `length * factor^8 + start`（`SIMDLerperConvex`）
    Convex { start: f32, length: f32 },
    /// `EnvelopePart::Hold(value)`
    Constant(f32),
}

/// 逐 lane 包络状态机（镜像 `SIMDVoiceEnvelope`）。
///
/// 关键：真实实现以「8 帧组」为单位推进（`next_sample` 一次产生 `WIDTH` 个值），
/// 组内若跨越阶段末尾则退回逐帧 scalar 公式（`manually_build_simd_sample`）。
/// 两种路径的 concave/convex 曲线公式**写法不同**（`powi(8)` vs 三次平方），
/// 因此必须复刻组的划分方式才能逐位一致。
struct LaneEnvelope {
    original: EnvelopeParameters,
    params: EnvelopeParameters,
    sample_rate: f32,
    allow_release: bool,
    killed: bool,
    stage: EnvelopeStage,
    data: LaneCurve,
    /// 当前阶段内已产生的帧数（真实实现的 `StageTime::simd_array_start`，lane 0）。
    t: u32,
    /// 当前阶段时长（`Constant` 阶段为 0）。
    end: u32,
    /// 组大小（= 运行时 SIMD 宽度）。
    group_len: u8,
    /// 当前组剩余帧数（`group_len` … 1）。
    group_left: u8,
    /// 当前组是否已进入逐帧 scalar 公式路径。
    group_manual: bool,
    /// 组缓冲：一次填满 `group_len` 帧的包络值（镜像真实实现一次 `next_sample`
    /// 产生一整组值；控制消息/Release 只可能在组边界到达，与真实粒度一致）。
    fifo: [f32; MAX_LANES],
    /// 组缓冲中已消费的帧数（`group_len` 表示需要重填）。
    fifo_pos: u8,
}

impl LaneEnvelope {
    fn new(
        original: EnvelopeParameters,
        params: EnvelopeParameters,
        allow_release: bool,
        sample_rate: f32,
        group_len: u8,
    ) -> LaneEnvelope {
        let mut env = LaneEnvelope {
            original,
            params,
            sample_rate,
            allow_release,
            killed: false,
            stage: EnvelopeStage::Delay,
            data: LaneCurve::Constant(0.0),
            t: 0,
            end: 0,
            group_len,
            group_left: group_len,
            group_manual: false,
            fifo: [0.0; MAX_LANES],
            fifo_pos: group_len,
        };
        let start = params.start_amplitude();
        env.set_stage(EnvelopeStage::Delay, start);
        env
    }

    /// 镜像 `EnvelopeParameters::get_stage_data`：`duration == 0` 递归到下一阶段。
    fn set_stage(&mut self, mut stage: EnvelopeStage, mut amp: f32) {
        loop {
            match self.params.parts[stage.as_usize()] {
                EnvelopePart::Lerp { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Lerp {
                        start: amp,
                        length: target - amp,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::LerpConcave { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Concave {
                        length: amp - target,
                        end: target,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::LerpConvex { target, duration } => {
                    if duration == 0 {
                        amp = target;
                        stage = stage.next_stage();
                        continue;
                    }
                    self.data = LaneCurve::Convex {
                        start: amp,
                        length: target - amp,
                    };
                    self.t = 0;
                    self.end = duration;
                }
                EnvelopePart::Hold(value) => {
                    self.data = LaneCurve::Constant(value);
                    self.t = 0;
                    self.end = 0;
                }
            }
            self.stage = stage;
            return;
        }
    }

    #[inline(always)]
    fn factor(&self) -> f32 {
        self.t as f32 / self.end as f32
    }

    /// 逐帧 scalar 公式（真实路径 `get_value_at_current_time` / 手动组路径）。
    #[inline(always)]
    fn value_scalar(&self) -> f32 {
        let factor = self.factor();
        match self.data {
            LaneCurve::Lerp { start, length } => start + length * factor,
            LaneCurve::Concave { length, end } => {
                let mult = (1.0 - factor).powi(8);
                length * mult + end
            }
            LaneCurve::Convex { start, length } => {
                let mult = factor.powi(8);
                length * mult + start
            }
            LaneCurve::Constant(value) => value,
        }
    }

    /// 组内 SIMD 公式（真实路径 `lerp_simd`：显式三次平方，非 `powi`）。
    #[inline(always)]
    fn value_simd(&self) -> f32 {
        let factor = self.factor();
        match self.data {
            LaneCurve::Lerp { start, length } => start + length * factor,
            LaneCurve::Concave { length, end } => {
                let r1 = 1.0 - factor;
                let r2 = r1 * r1;
                let r3 = r2 * r2;
                let mult = r3 * r3;
                length * mult + end
            }
            LaneCurve::Convex { start, length } => {
                let r1 = factor * factor;
                let r2 = r1 * r1;
                let mult = r2 * r2;
                length * mult + start
            }
            LaneCurve::Constant(value) => value,
        }
    }

    #[inline(always)]
    fn advance_group(&mut self) {
        if self.group_left <= 1 {
            self.group_left = self.group_len;
        } else {
            self.group_left -= 1;
        }
    }

    fn switch_stage(&mut self) {
        let amp = self.value_scalar();
        self.set_stage(self.stage.next_stage(), amp);
    }

    /// 镜像 `SIMDVoiceEnvelope::update_stage`（控制消息改包络时保留当前阶段与当前幅度）。
    fn update_stage(&mut self) {
        let amp = self.value_scalar();
        self.set_stage(self.stage, amp);
    }

    /// 逐帧 scalar 步进（镜像 `next_sample_inner` 的手动路径与 `Constant` 分支）。
    /// `self.t` 语义：**下一个待产生帧**的阶段时间（组边界上与真实实现完全一致）。
    #[inline(always)]
    fn frame_scalar(&mut self) -> f32 {
        loop {
            match self.data {
                LaneCurve::Constant(value) => {
                    // `StageData::Constant` 分支不推进阶段时间。
                    self.advance_group();
                    return value;
                }
                _ => {
                    if self.t >= self.end {
                        // 组起点已越过阶段末尾：切换阶段，不消耗帧。
                        self.switch_stage();
                        continue;
                    }
                    if self.group_left == self.group_len {
                        // 组入口：按当前阶段判定本组是否退回逐帧 scalar 公式。
                        self.group_manual = self.t + (self.group_len as u32 - 1) >= self.end;
                    }
                    let value = if self.group_manual {
                        self.value_scalar()
                    } else {
                        self.value_simd()
                    };
                    self.t += 1;
                    if self.group_manual && self.t >= self.end {
                        self.switch_stage();
                    }
                    self.advance_group();
                    return value;
                }
            }
        }
    }

    /// 组填充（慢路径）：逐帧 scalar，复用已验证的逐帧状态机，可跨阶段切换。
    fn fill_slow(&mut self) {
        for i in 0..self.group_len as usize {
            self.fifo[i] = self.frame_scalar();
        }
        self.fifo_pos = 0;
    }

    /// 组填充（快路径）：整组落在同一阶段内（或 `Constant`）→ 一次向量计算。
    ///
    /// 与 `fill_slow` 逐位等价：Lerp 公式两端相同；Concave/Convex 使用真实实现
    /// `lerp_simd` 的显式三次平方公式（手动路径才用 `powi`），且调用条件是
    /// 「本组不越阶段末尾」，正好对应真实实现的 case A。
    #[inline(always)]
    fn fill_fast<S: Simd>(&mut self) {
        if let LaneCurve::Constant(value) = self.data {
            for i in 0..self.group_len as usize {
                self.fifo[i] = value;
            }
            self.group_left = self.group_len;
            self.fifo_pos = 0;
            return;
        }

        simd_invoke!(S, {
            let n = S::Vf32::WIDTH;
            let mut idx = S::Vf32::zeroes();
            for i in 0..n {
                idx[i] = i as f32;
            }
            // 等价于真实 `StageTime::new` + `progress_simd_array`。
            let factor = (S::Vf32::set1(self.t as f32) + idx) / S::Vf32::set1(self.end as f32);
            let values = match self.data {
                LaneCurve::Lerp { start, length } => {
                    S::Vf32::set1(start) + S::Vf32::set1(length) * factor
                }
                LaneCurve::Concave { length, end } => {
                    let r1 = S::Vf32::set1(1.0) - factor;
                    let r2 = r1 * r1;
                    let r3 = r2 * r2;
                    let mult = r3 * r3;
                    S::Vf32::set1(length) * mult + S::Vf32::set1(end)
                }
                LaneCurve::Convex { start, length } => {
                    let r1 = factor * factor;
                    let r2 = r1 * r1;
                    let mult = r2 * r2;
                    S::Vf32::set1(length) * mult + S::Vf32::set1(start)
                }
                LaneCurve::Constant(value) => S::Vf32::set1(value),
            };
            let mut arr = [0.0f32; MAX_LANES];
            values.copy_to_slice(&mut arr);
            self.fifo = arr;
        });

        self.t += self.group_len as u32;
        self.group_left = self.group_len;
        self.group_manual = false;
        self.fifo_pos = 0;
    }

    /// 组边界重填（内核路径：能整组向量化时走快路径）。
    #[inline(always)]
    fn refill_simd<S: Simd>(&mut self) {
        let n = self.group_len as u32;
        let fast = match self.data {
            LaneCurve::Constant(_) => true,
            _ => self.t + (n - 1) < self.end,
        };
        if fast {
            self.fill_fast::<S>();
        } else {
            self.fill_slow();
        }
    }

    /// 内核入口：消费组缓冲中的下一帧（镜像 `next_sample` 的 8 帧粒度）。
    #[inline(always)]
    fn next_frame_simd<S: Simd>(&mut self) -> f32 {
        if self.fifo_pos >= self.group_len {
            self.refill_simd::<S>();
        }
        let value = self.fifo[self.fifo_pos as usize];
        self.fifo_pos += 1;
        value
    }

    /// 标量入口（非批回退路径与差分测试参考实现）：逐帧 scalar 组填充。
    #[inline(always)]
    fn next_frame(&mut self) -> f32 {
        if self.fifo_pos >= self.group_len {
            self.fill_slow();
        }
        let value = self.fifo[self.fifo_pos as usize];
        self.fifo_pos += 1;
        value
    }

    fn signal_release(&mut self, rel_type: ReleaseType) {
        if rel_type == ReleaseType::Kill {
            self.params.modify_stage_data(
                5,
                EnvelopePart::lerp(0.0, (0.001 * self.sample_rate) as u32),
            );
            self.update_stage();
            self.killed = true;
        }
        if self.allow_release || self.killed {
            let amp = self.value_scalar();
            self.set_stage(EnvelopeStage::Release, amp);
        }
    }

    fn modify_envelope(&mut self, envelope: EnvelopeControlData) {
        if !self.killed {
            self.params = self
                .original
                .with_envelope_control(envelope, self.sample_rate);
            self.update_stage();
        }
    }

    #[inline(always)]
    fn ended(&self) -> bool {
        self.stage == EnvelopeStage::Finished
    }
}

/// 循环模式热标记（避免热循环内做 `LoopMode` 枚举匹配）。
const HOT_MODE_NOLOOP: u8 = 0;
const HOT_MODE_LOOP: u8 = 1;
const HOT_MODE_SUSTAIN: u8 = 2;

/// 回绕：镜像 `SampleReaderLoop`/`SampleReaderLoopSustain` 的 `pos > end` 分支。
#[inline(always)]
fn wrap_pos(pos: usize, start: usize, end: usize) -> usize {
    let span = end - start;
    let d = pos - end - 1;
    start + if d >= span { d % span } else { d }
}

/// 每 lane 的采样热状态。
///
/// 每 chunk 从 `BatchLane` 裁剪一次（指针/长度/回绕边界/模式标记全部展平到连续
/// 内存），热循环内不再穿过 `&mut BatchLane` → `BufferSamplers` 枚举 → `Arc` 间接
/// 的多级寻址。加载/回绕均保持与真实 reader 逐位相同的语义。
///
/// # Safety
/// `ptr_l/ptr_r` 指向 lane 持有的 `Arc<[f32]>` 数据；chunk 渲染期间 lane 不被结构性
/// 修改（`Arc` 不会被释放或替换），因此指针在 chunk 内有效。
#[derive(Clone, Copy)]
struct HotLane {
    ptr_l: *const f32,
    ptr_r: *const f32,
    len_l: usize,
    len_r: usize,
    off_l: usize,
    off_r: usize,
    end_l: usize,
    end_r: usize,
    start: usize,
    last_l: usize,
    last_r: usize,
    time: f64,
    speed: f32,
    gain_l: f32,
    gain_r: f32,
    mode: u8,
    interp: u8,
    /// `LoopSustain` 且已释放（位置线性外推，可能越界 → 必须走有检查加载）。
    released: bool,
    /// 循环模式下回绕后位置必然在缓冲内（含线性插值的 `+1` 余量）→ 无检查加载。
    in_range: bool,
}

impl HotLane {
    const EMPTY: HotLane = HotLane {
        ptr_l: std::ptr::null(),
        ptr_r: std::ptr::null(),
        len_l: 0,
        len_r: 0,
        off_l: 0,
        off_r: 0,
        end_l: 0,
        end_r: 0,
        start: 0,
        last_l: 0,
        last_r: 0,
        time: 0.0,
        speed: 0.0,
        gain_l: 0.0,
        gain_r: 0.0,
        mode: 0,
        interp: 0,
        released: false,
        in_range: false,
    };

    /// 镜像 `F32BufferSampler::get`：越界返回 0.0（`in_range` 时证明不可能越界）。
    #[inline(always)]
    fn load(&self, pos: usize, ptr: *const f32, len: usize) -> f32 {
        if self.in_range || pos < len {
            // SAFETY: `in_range` 由构造期证明（回绕后位置 + 线性余量 ≤ 缓冲长度），
            // 否则显式做了 `pos < len` 检查；`ptr` 由 lane 持有的 Arc 保证有效。
            unsafe { *ptr.add(pos) }
        } else {
            0.0
        }
    }

    /// 消融用：无分支采样（恒定 in_range、nearest、无回绕检查）——仅用于定位
    /// 计时瓶颈，语义不完整，不参与正确性路径。
    #[inline(always)]
    fn sample_step_flat(&mut self) -> (f32, f32) {
        let t = self.time;
        self.time = t + self.speed as f64;
        let index = (t as i32) as usize & 1023;
        unsafe {
            (
                *self.ptr_l.add(index + self.off_l),
                *self.ptr_r.add(index + self.off_r),
            )
        }
    }

    /// 推进一帧采样（时间 → 索引 → 回绕 → 抓取/插值），返回滤波前的左右样本。
    #[inline(always)]
    fn sample_step(&mut self) -> (f32, f32) {
        let t = self.time;
        self.time = t + self.speed as f64;
        let index = t as i32;
        // `reader.get(pos)` 的入参（未回绕，含 offset）。
        let base_l = index as usize + self.off_l;
        let base_r = index as usize + self.off_r;

        let sustain = self.mode == HOT_MODE_SUSTAIN;
        if sustain && !self.released {
            // 未释放：`SampleReaderLoopSustain::get` 每次都更新 `last`（未回绕值）；
            // 线性插值连续读 index 与 index+1 → `last` 最终为 index+1。
            self.last_l = if self.interp == 1 { base_l + 1 } else { base_l };
            self.last_r = if self.interp == 1 { base_r + 1 } else { base_r };
        }

        let (pos_l, pos_r) = if sustain && self.released {
            // 释放后：冻结 last、按剩余偏移线性推进（位置可能越界）。
            (
                base_l.wrapping_sub(self.last_l).wrapping_add(self.end_l),
                base_r.wrapping_sub(self.last_r).wrapping_add(self.end_r),
            )
        } else if self.mode == HOT_MODE_NOLOOP {
            // `SampleReaderNoLoop::get` = `buffer.get(pos + offset)`：**不回绕**，
            // 越界由有检查加载返回 0.0。
            (base_l, base_r)
        } else {
            (
                if base_l > self.end_l {
                    wrap_pos(base_l, self.start, self.end_l)
                } else {
                    base_l
                },
                if base_r > self.end_r {
                    wrap_pos(base_r, self.start, self.end_r)
                } else {
                    base_r
                },
            )
        };

        if self.interp == 0 {
            (
                self.load(pos_l, self.ptr_l, self.len_l),
                self.load(pos_r, self.ptr_r, self.len_r),
            )
        } else {
            // 线性插值：真实实现是第二次 `reader.get(index+1)`——**独立**回绕，
            // 不能用 `pos + 1`（`pos == loop_end` 时两者不同）。
            let (pos_l1, pos_r1) = self.advance_one(base_l, base_r, self.released);
            let frac = (t - index as f64) as f32;
            let l0 = self.load(pos_l, self.ptr_l, self.len_l);
            let l1 = self.load(pos_l1, self.ptr_l, self.len_l);
            let r0 = self.load(pos_r, self.ptr_r, self.len_r);
            let r1 = self.load(pos_r1, self.ptr_r, self.len_r);
            (l0 * (1.0 - frac) + l1 * frac, r0 * (1.0 - frac) + r1 * frac)
        }
    }

    /// `index + 1` 的回绕（与 `sample_step` 中首个位置同语义，供线性插值使用）。
    #[inline(always)]
    fn advance_one(&self, base_l: usize, base_r: usize, released_sustain: bool) -> (usize, usize) {
        let base_l = base_l + 1;
        let base_r = base_r + 1;
        if released_sustain {
            (
                base_l.wrapping_sub(self.last_l).wrapping_add(self.end_l),
                base_r.wrapping_sub(self.last_r).wrapping_add(self.end_r),
            )
        } else if self.mode == HOT_MODE_NOLOOP {
            (base_l, base_r)
        } else {
            (
                if base_l > self.end_l {
                    wrap_pos(base_l, self.start, self.end_l)
                } else {
                    base_l
                },
                if base_r > self.end_r {
                    wrap_pos(base_r, self.start, self.end_r)
                } else {
                    base_r
                },
            )
        }
    }
}

/// 一个可批 voice 的全部状态（lane = voice）。
pub struct BatchLane {
    // ── 采样 ──
    left: SideReader,
    right: SideReader,
    loop_mode: LoopMode,
    loop_start: usize,
    loop_end: usize,
    interpolator: Interpolator,
    time: f64,
    speed: f32,
    /// spawn 期速度倍率（`SIMDConstantControl` 的 `base`，控制消息变化时重算 speed）。
    speed_mult: f32,
    is_released: bool,
    // ── 包络 ──
    env: LaneEnvelope,
    // ── 增益（力度 × 等功率声像，spawn 期合并）──
    gain_l: f32,
    gain_r: f32,
    // ── 滤波（biquad DF1，无滤波器时 filter_enabled = false）──
    filter_enabled: bool,
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1l: f32,
    x2l: f32,
    y1l: f32,
    y2l: f32,
    x1r: f32,
    x2r: f32,
    y1r: f32,
    y2r: f32,
    // ── 治理/生命周期 ──
    releasing: bool,
    killed: bool,
    velocity: u8,
    exclusive_class: Option<u8>,
}

impl BatchLane {
    pub(crate) fn new(init: &BatchLaneInit, control: &VoiceControlData) -> BatchLane {
        let length_l = Some(init.loop_stop.unwrap_or(init.samples_l.len()));
        let length_r = Some(init.loop_stop.unwrap_or(init.samples_r.len()));
        let modified = init
            .envelope
            .with_envelope_control(control.envelope, init.sample_rate);
        let allow_release = init.loop_mode != LoopMode::OneShot;

        let mut lane = BatchLane {
            left: SideReader {
                buffer: BufferSamplers::new_f32(init.samples_l.clone()),
                offset: init.loop_offset,
                last: 0,
                length: length_l,
            },
            right: SideReader {
                buffer: BufferSamplers::new_f32(init.samples_r.clone()),
                offset: init.loop_offset,
                last: 0,
                length: length_r,
            },
            loop_mode: init.loop_mode,
            loop_start: init.loop_start,
            loop_end: init.loop_end,
            interpolator: init.interpolator,
            time: 0.0,
            // 镜像 `SIMDConstantControl::new(speed_mult, control, |vc| vc.voice_pitch_multiplier)`。
            speed: init.speed_mult * control.voice_pitch_multiplier,
            speed_mult: init.speed_mult,
            is_released: false,
            env: LaneEnvelope::new(
                init.envelope,
                modified,
                allow_release,
                init.sample_rate,
                init.group_len,
            ),
            gain_l: init.gain_l,
            gain_r: init.gain_r,
            filter_enabled: init.filter.is_some(),
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1l: 0.0,
            x2l: 0.0,
            y1l: 0.0,
            y2l: 0.0,
            x1r: 0.0,
            x2r: 0.0,
            y1r: 0.0,
            y2r: 0.0,
            releasing: false,
            killed: false,
            velocity: init.velocity,
            exclusive_class: init.exclusive_class,
        };
        if let Some(filter) = &init.filter {
            let c = filter.coefficients();
            lane.b0 = c.b0;
            lane.b1 = c.b1;
            lane.b2 = c.b2;
            lane.a1 = c.a1;
            lane.a2 = c.a2;
        }
        lane
    }

    /// 该 lane 是否走 biquad（同一 chunk 内必须一致，否则无法整块向量化）。
    #[inline(always)]
    pub(crate) fn filter_enabled(&self) -> bool {
        self.filter_enabled
    }

    /// 循环模式热标记。
    #[inline(always)]
    fn hot_mode(&self) -> u8 {
        match self.loop_mode {
            LoopMode::NoLoop | LoopMode::OneShot => HOT_MODE_NOLOOP,
            LoopMode::LoopContinuous => HOT_MODE_LOOP,
            LoopMode::LoopSustain => HOT_MODE_SUSTAIN,
        }
    }

    /// `LoopSustain` 且已进入释放后的线性外推分支。
    #[inline(always)]
    fn released_sustain(&self) -> bool {
        self.loop_mode == LoopMode::LoopSustain && self.is_released
    }

    /// 回绕后位置是否必然落在缓冲内（可走无检查加载）。
    ///
    /// 循环模式下位置 ∈ [loop_start, loop_end]；线性插值还要多读一个样本。
    #[inline(always)]
    fn in_range_now(&self, linear: bool) -> bool {
        let mode = self.hot_mode();
        if mode == HOT_MODE_NOLOOP || self.released_sustain() {
            return false;
        }
        let slack = if linear { 2 } else { 1 };
        self.loop_end.saturating_add(slack) <= self.left.buffer.slice().len()
            && self.loop_end.saturating_add(slack) <= self.right.buffer.slice().len()
    }

    /// 把 lane 的采样热状态展平（每 chunk 一次；见 [`HotLane`]）。
    #[inline(always)]
    fn hot(&self) -> HotLane {
        let left = self.left.buffer.slice();
        let right = self.right.buffer.slice();
        let linear = self.interpolator == Interpolator::Linear;
        HotLane {
            ptr_l: left.as_ptr(),
            ptr_r: right.as_ptr(),
            len_l: left.len(),
            len_r: right.len(),
            off_l: self.left.offset,
            off_r: self.right.offset,
            end_l: self.loop_end,
            end_r: self.loop_end,
            start: self.loop_start,
            last_l: self.left.last,
            last_r: self.right.last,
            time: self.time,
            speed: self.speed,
            gain_l: self.gain_l,
            gain_r: self.gain_r,
            mode: self.hot_mode(),
            interp: if linear { 1 } else { 0 },
            released: self.released_sustain(),
            in_range: self.in_range_now(linear),
        }
    }

    /// 回写热状态（每 chunk 一次）。
    #[inline(always)]
    fn apply_hot(&mut self, hot: &HotLane) {
        self.time = hot.time;
        self.left.last = hot.last_l;
        self.right.last = hot.last_r;
    }

    /// 一侧的采样读取（逐位镜像 `SampleReaderLoop`/`SampleReaderLoopSustain`/`SampleReaderNoLoop`）。
    #[inline(always)]
    fn read_side(
        side: &mut SideReader,
        loop_mode: LoopMode,
        loop_start: usize,
        loop_end: usize,
        is_released: bool,
        index: usize,
    ) -> f32 {
        let mut pos = index + side.offset;
        match loop_mode {
            LoopMode::NoLoop | LoopMode::OneShot => {}
            LoopMode::LoopContinuous => {
                if pos > loop_end {
                    let span = loop_end - loop_start;
                    let d = pos - loop_end - 1;
                    pos = loop_start + if d >= span { d % span } else { d };
                }
            }
            LoopMode::LoopSustain => {
                if !is_released {
                    side.last = pos;
                    if pos > loop_end {
                        let span = loop_end - loop_start;
                        let d = pos - loop_end - 1;
                        pos = loop_start + if d >= span { d % span } else { d };
                    }
                } else {
                    // 释放后：冻结 last、按剩余偏移线性推进（release 模式回绕语义，
                    // 用 wrapping 运算避免 debug 下 usize 下溢 panic，与 release 一致）。
                    pos = pos.wrapping_sub(side.last).wrapping_add(loop_end);
                }
            }
        }
        side.buffer.get(pos)
    }

    #[inline(always)]
    fn read_left(&mut self, index: usize) -> f32 {
        Self::read_side(
            &mut self.left,
            self.loop_mode,
            self.loop_start,
            self.loop_end,
            self.is_released,
            index,
        )
    }

    #[inline(always)]
    fn read_right(&mut self, index: usize) -> f32 {
        Self::read_side(
            &mut self.right,
            self.loop_mode,
            self.loop_start,
            self.loop_end,
            self.is_released,
            index,
        )
    }

    /// 采样（时间推进 + 回绕 + 抓取），不含包络/增益。
    #[inline(always)]
    fn next_sample_only(&mut self) -> (f32, f32) {
        let t = self.time;
        self.time = t + self.speed as f64;
        let index = t as i32;
        match self.interpolator {
            Interpolator::Nearest => {
                let l = self.read_left(index as usize);
                let r = self.read_right(index as usize);
                (l, r)
            }
            Interpolator::Linear => {
                // 与 `SIMDLinearSampleGrabber::get` 一致：先取 index 再取 index+1
                // （`SampleReaderLoopSustain` 的 last 会被两次读取更新）。
                let frac = (t - index as f64) as f32;
                let l0 = self.read_left(index as usize);
                let l1 = self.read_left(index as usize + 1);
                let r0 = self.read_right(index as usize);
                let r1 = self.read_right(index as usize + 1);
                (l0 * (1.0 - frac) + l1 * frac, r0 * (1.0 - frac) + r1 * frac)
            }
        }
    }

    /// 采样 + 包络（不含增益；消融计时用，不参与逐位对照）。
    #[inline(always)]
    fn next_sample_env(&mut self) -> (f32, f32) {
        let (sl, sr) = self.next_sample_only();
        let env = self.env.next_frame();
        (sl * env, sr * env)
    }

    /// 增益与包络的应用（顺序与真实链一致：`env * (gain * sample)`）。
    #[inline(always)]
    fn apply_env_gain(&self, env: f32, sl: f32, sr: f32) -> (f32, f32) {
        (env * (self.gain_l * sl), env * (self.gain_r * sr))
    }

    /// 采样 → 包络 → 增益（标量包络入口，供非批回退路径使用）。
    #[inline(always)]
    fn next_pre_filter(&mut self) -> (f32, f32) {
        let (sl, sr) = self.next_sample_only();
        let env = self.env.next_frame();
        self.apply_env_gain(env, sl, sr)
    }

    /// 采样 → 包络 → 增益（向量包络入口，供批内核使用）。
    #[inline(always)]
    fn next_pre_filter_simd<S: Simd>(&mut self) -> (f32, f32) {
        let (sl, sr) = self.next_sample_only();
        let env = self.env.next_frame_simd::<S>();
        self.apply_env_gain(env, sl, sr)
    }

    /// biquad（DF1，与 `BiQuadFilter::process` 逐项一致）。
    #[inline(always)]
    fn filter_frame(&mut self, l: f32, r: f32) -> (f32, f32) {
        if !self.filter_enabled {
            return (l, r);
        }
        let ol = self.b0 * l + self.b1 * self.x1l + self.b2 * self.x2l
            - self.a1 * self.y1l
            - self.a2 * self.y2l;
        self.x2l = self.x1l;
        self.x1l = l;
        self.y2l = self.y1l;
        self.y1l = ol;

        let or = self.b0 * r + self.b1 * self.x1r + self.b2 * self.x2r
            - self.a1 * self.y1r
            - self.a2 * self.y2r;
        self.x2r = self.x1r;
        self.x1r = r;
        self.y2r = self.y1r;
        self.y1r = or;

        (ol, or)
    }

    /// 标量渲染（单 lane），用于：非批路径（如通道使用 key 级线程池时）/ 尾部不足
    /// 一个 chunk 的 voice。逐位等价于真实链式生成器。
    pub(crate) fn render_scalar(&mut self, buffer: &mut [f32]) {
        let frames = buffer.len() / 2;
        for f in 0..frames {
            let (l, r) = self.next_pre_filter();
            let (l, r) = self.filter_frame(l, r);
            buffer[2 * f] += l;
            buffer[2 * f + 1] += r;
        }
    }

    #[inline(always)]
    fn past_end_side(&self, side: &SideReader, pos: usize) -> bool {
        match self.loop_mode {
            LoopMode::NoLoop | LoopMode::OneShot => match side.length {
                Some(len) => pos >= len.saturating_sub(side.offset),
                None => false,
            },
            // `SampleReaderLoop::is_past_end` 恒为 false；
            // `SampleReaderLoopSustain` 用 stop/length 与冻结的 last 判定（原样镜像）。
            LoopMode::LoopContinuous => false,
            LoopMode::LoopSustain => match side.length {
                Some(len) => pos >= len.saturating_sub(side.last.saturating_add(side.offset)),
                None => false,
            },
        }
    }

    /// 镜像 `SIMDStereoVoiceSampler::ended`。
    #[inline(always)]
    fn past_end(&self) -> bool {
        let pos = self.time as usize;
        self.past_end_side(&self.left, pos) || self.past_end_side(&self.right, pos)
    }

    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.releasing = true;
        if rel_type == ReleaseType::Kill {
            self.killed = true;
        }
        self.env.signal_release(rel_type);
        // 镜像 `SIMDSampleGrabber::signal_release` → reader：仅 `SampleReaderLoopSustain`
        // 有状态（Loop/NoLoop 为空实现），且与 rel_type / allow_release 无关。
        if self.loop_mode == LoopMode::LoopSustain {
            self.is_released = true;
        }
    }

    fn process_controls(&mut self, control: &VoiceControlData) {
        self.speed = self.speed_mult * control.voice_pitch_multiplier;
        self.env.modify_envelope(control.envelope);
    }
}

/// 标量渲染入口（内核单 chunk）。
///
/// 要求 `lanes.len()` 等于运行时 SIMD 宽度，且所有 lane 的 `filter_enabled` 一致。
/// 统一 chunk（模式/插值/释放状态一致且可证明无越界）走专用内核，否则回退通用内核。
pub(crate) fn render_batch_chunk(
    lanes: &mut [&mut BatchLane],
    out: &mut [f32],
    frames: usize,
) -> bool {
    simd_runtime_generate!(
        fn render(
            lanes: &mut [&mut BatchLane],
            out: &mut [f32],
            frames: usize,
            uniform: Option<u32>,
        ) -> bool {
            if lanes.len() != S::Vf32::WIDTH {
                return false;
            }
            match uniform {
                Some(tag) => match tag {
                    0 => render_uniform::<S, HOT_MODE_NOLOOP, false, false>(lanes, out, frames),
                    1 => render_uniform::<S, HOT_MODE_NOLOOP, true, false>(lanes, out, frames),
                    2 => render_uniform::<S, HOT_MODE_LOOP, false, false>(lanes, out, frames),
                    3 => render_uniform::<S, HOT_MODE_LOOP, true, false>(lanes, out, frames),
                    4 => render_uniform::<S, HOT_MODE_SUSTAIN, false, false>(lanes, out, frames),
                    5 => render_uniform::<S, HOT_MODE_SUSTAIN, true, false>(lanes, out, frames),
                    6 => render_uniform::<S, HOT_MODE_SUSTAIN, false, true>(lanes, out, frames),
                    _ => render_uniform::<S, HOT_MODE_SUSTAIN, true, true>(lanes, out, frames),
                },
                None => render_inner::<S, 3>(lanes, out, frames),
            }
            true
        }
    );

    let uniform = uniform_chunk(lanes).map(|(mode, linear, released)| {
        // 0/1 = NoLoop(+linear)…，4/5 = Sustain 未释放(+linear)，6/7 = Sustain 已释放(+linear)。
        let tag = if mode == HOT_MODE_SUSTAIN {
            4 + u8::from(linear) + u8::from(released) * 2
        } else {
            mode * 2 + u8::from(linear)
        };
        u32::from(tag)
    });
    render(lanes, out, frames, uniform)
}

/// 阶段消融入口（测试/调优用）：`mode` 0 = 仅采样；1 = +包络；2 = +增益；
/// 3 = 完整（+ biquad/混音）。
pub(crate) fn render_batch_chunk_mode(
    lanes: &mut [&mut BatchLane],
    out: &mut [f32],
    frames: usize,
    mode: u32,
) -> bool {
    simd_runtime_generate!(
        fn render(lanes: &mut [&mut BatchLane], out: &mut [f32], frames: usize, mode: u32) -> bool {
            if lanes.len() != S::Vf32::WIDTH {
                return false;
            }
            match mode {
                0 => render_inner::<S, 0>(lanes, out, frames),
                1 => render_inner::<S, 1>(lanes, out, frames),
                2 => render_inner::<S, 2>(lanes, out, frames),
                4 => render_inner::<S, 4>(lanes, out, frames),
                _ => render_inner::<S, 3>(lanes, out, frames),
            }
            true
        }
    );

    render(lanes, out, frames, mode)
}

fn render_inner<S: Simd, const MODE: u32>(
    lanes: &mut [&mut BatchLane],
    out: &mut [f32],
    frames: usize,
) {
    let width = S::Vf32::WIDTH;
    debug_assert_eq!(lanes.len(), width);

    // 必须在 `simd_invoke!` 内执行：它是唯一建立 `#[target_feature(enable = "avx2,fma")]`
    // 的入口（B0 关键坑：泛型方法体直接调用 intrinsic 会退化为函数调用，实测 0.55× → 3.48×）。
    simd_invoke!(S, {
        let mut sl_arr = [0.0f32; MAX_LANES];
        let mut sr_arr = [0.0f32; MAX_LANES];
        let mut filter = VecFilter::<S>::load(lanes, width);

        // 采样热状态展平（每 chunk 一次）。
        let mut hot_lanes = [HotLane::EMPTY; MAX_LANES];
        for k in 0..width {
            hot_lanes[k] = lanes[k].hot();
        }

        for step in 0..frames {
            // 1) 标量：逐 lane 采样（+ 包络 + 增益，按消融档位）。
            unsafe {
                for k in 0..width {
                    let hot = hot_lanes.get_unchecked_mut(k);
                    let (sl, sr) = if MODE == 4 {
                        hot.sample_step_flat()
                    } else {
                        hot.sample_step()
                    };
                    let (l, r) = match MODE {
                        0 | 4 => (sl, sr),
                        1 => {
                            let env = lanes.get_unchecked_mut(k).env.next_frame_simd::<S>();
                            (sl * env, sr * env)
                        }
                        _ => {
                            let env = lanes.get_unchecked_mut(k).env.next_frame_simd::<S>();
                            (env * (hot.gain_l * sl), env * (hot.gain_r * sr))
                        }
                    };
                    *sl_arr.get_unchecked_mut(k) = l;
                    *sr_arr.get_unchecked_mut(k) = r;
                }
            }
            let sl = S::Vf32::load_from_slice(&sl_arr);
            let sr = S::Vf32::load_from_slice(&sr_arr);

            // 2) 向量：biquad（lane = voice，时间维依赖链消失）。
            let (sl, sr) = if MODE >= 3 {
                filter.process(sl, sr)
            } else {
                (sl, sr)
            };

            // 3) 混音：水平求和后一次写入共享通道缓冲。
            let lsum = sl.horizontal_add();
            let rsum = sr.horizontal_add();
            unsafe {
                *out.get_unchecked_mut(2 * step) += lsum;
                *out.get_unchecked_mut(2 * step + 1) += rsum;
            }
        }

        filter.store(lanes, width);
        for k in 0..width {
            lanes[k].apply_hot(&hot_lanes[k]);
        }
    });
}

/// 向量 biquad 状态（chunk 级；装载后热循环内不碰 lane 内存）。
struct VecFilter<S: Simd> {
    enabled: bool,
    b0: S::Vf32,
    b1: S::Vf32,
    b2: S::Vf32,
    a1: S::Vf32,
    a2: S::Vf32,
    x1l: S::Vf32,
    x2l: S::Vf32,
    y1l: S::Vf32,
    y2l: S::Vf32,
    x1r: S::Vf32,
    x2r: S::Vf32,
    y1r: S::Vf32,
    y2r: S::Vf32,
}

impl<S: Simd> VecFilter<S> {
    #[inline(always)]
    fn load(lanes: &[&mut BatchLane], width: usize) -> VecFilter<S> {
        simd_invoke!(S, {
            macro_rules! lv {
                ($field:ident) => {{
                    let mut arr = [0.0f32; MAX_LANES];
                    for k in 0..width {
                        arr[k] = lanes[k].$field;
                    }
                    S::Vf32::load_from_slice(&arr)
                }};
            }
            VecFilter {
                enabled: lanes[0].filter_enabled,
                b0: lv!(b0),
                b1: lv!(b1),
                b2: lv!(b2),
                a1: lv!(a1),
                a2: lv!(a2),
                x1l: lv!(x1l),
                x2l: lv!(x2l),
                y1l: lv!(y1l),
                y2l: lv!(y2l),
                x1r: lv!(x1r),
                x2r: lv!(x2r),
                y1r: lv!(y1r),
                y2r: lv!(y2r),
            }
        })
    }

    #[inline(always)]
    fn store(&self, lanes: &mut [&mut BatchLane], width: usize) {
        simd_invoke!(S, {
            macro_rules! sv {
                ($field:ident, $value:expr) => {{
                    let mut arr = [0.0f32; MAX_LANES];
                    $value.copy_to_slice(&mut arr);
                    for k in 0..width {
                        lanes[k].$field = arr[k];
                    }
                }};
            }
            sv!(x1l, self.x1l);
            sv!(x2l, self.x2l);
            sv!(y1l, self.y1l);
            sv!(y2l, self.y2l);
            sv!(x1r, self.x1r);
            sv!(x2r, self.x2r);
            sv!(y1r, self.y1r);
            sv!(y2r, self.y2r);
        });
    }

    /// DF1，与 `BiQuadFilter::process` 逐项一致（无滤波器时原样返回，避免 -0.0 变号）。
    #[inline(always)]
    fn process(&mut self, sl: S::Vf32, sr: S::Vf32) -> (S::Vf32, S::Vf32) {
        if !self.enabled {
            return (sl, sr);
        }
        let ol = self.b0 * sl + self.b1 * self.x1l + self.b2 * self.x2l
            - self.a1 * self.y1l
            - self.a2 * self.y2l;
        self.x2l = self.x1l;
        self.x1l = sl;
        self.y2l = self.y1l;
        self.y1l = ol;

        let or = self.b0 * sr + self.b1 * self.x1r + self.b2 * self.x2r
            - self.a1 * self.y1r
            - self.a2 * self.y2r;
        self.x2r = self.x1r;
        self.x1r = sr;
        self.y2r = self.y1r;
        self.y1r = or;

        (ol, or)
    }
}

/// 统一 chunk 判定：所有 lane 的循环模式 / 插值器 / 释放状态一致，且循环模式
/// 可证明无越界（`in_range`）→ 可走无逐 lane 分支的专用内核。
fn uniform_chunk(lanes: &[&mut BatchLane]) -> Option<(u8, bool, bool)> {
    let first = lanes.first()?;
    let linear = first.interpolator == Interpolator::Linear;
    if !first.in_range_now(linear) {
        return None;
    }
    let mode = first.hot_mode();
    let released = first.released_sustain();
    for lane in lanes.iter().skip(1) {
        if lane.hot_mode() != mode
            || (lane.interpolator == Interpolator::Linear) != linear
            || lane.released_sustain() != released
            || !lane.in_range_now(linear)
        {
            return None;
        }
    }
    Some((mode, linear, released))
}

/// 统一 chunk 专用内核：循环模式/插值器/释放状态由 const 泛型给定，热循环内
/// **没有**逐 lane 的模式/插值/释放分支与 `in_range` 检查（B1.3 调优主要收益）。
///
/// 与通用内核逐位等价：位置计算、回绕、`last` 更新语义完全一致（见 `HotLane`）。
fn render_uniform<S: Simd, const MODE_TAG: u8, const LINEAR: bool, const RELEASED: bool>(
    lanes: &mut [&mut BatchLane],
    out: &mut [f32],
    frames: usize,
) {
    let width = S::Vf32::WIDTH;
    debug_assert_eq!(lanes.len(), width);
    /// 循环模式且未释放 → `uniform_chunk` 已证明回绕后位置必然在缓冲内。
    const fn in_range_const<const MODE_TAG: u8, const RELEASED: bool>() -> bool {
        !RELEASED && MODE_TAG != HOT_MODE_NOLOOP
    }
    simd_invoke!(S, {
        let mut sl_arr = [0.0f32; MAX_LANES];
        let mut sr_arr = [0.0f32; MAX_LANES];
        let mut filter = VecFilter::<S>::load(lanes, width);

        let mut hot_lanes = [HotLane::EMPTY; MAX_LANES];
        for k in 0..width {
            hot_lanes[k] = lanes[k].hot();
        }

        // `last` 只在 chunk 边界被读取（`is_past_end` / 释放后外推），但需要每帧
        // 的未回绕位置，因此按帧写入热状态（无读取）。
        let track_last = MODE_TAG == HOT_MODE_SUSTAIN && !RELEASED;

        macro_rules! load_sample {
            ($hot:expr, $pos:expr, $ptr:expr, $len:expr) => {{
                if in_range_const::<MODE_TAG, RELEASED>() {
                    // SAFETY: 调用点位于 `unsafe` 步进块内；`in_range_const` 保证
                    // 回绕后位置 + 线性余量 ≤ 缓冲长度（见 `uniform_chunk`）。
                    *$ptr.add($pos)
                } else {
                    $hot.load($pos, $ptr, $len)
                }
            }};
        }

        for step in 0..frames {
            unsafe {
                for k in 0..width {
                    let hot = hot_lanes.get_unchecked_mut(k);
                    let t = hot.time;
                    hot.time = t + hot.speed as f64;
                    let index = t as i32;
                    let base_l = index as usize + hot.off_l;
                    let base_r = index as usize + hot.off_r;

                    if track_last {
                        // `SampleReaderLoopSustain::get` 每次读取都更新 `last`
                        // （未回绕位置；线性插值连读 index/index+1 → 为 index+1）。
                        hot.last_l = if LINEAR { base_l + 1 } else { base_l };
                        hot.last_r = if LINEAR { base_r + 1 } else { base_r };
                    }

                    let (pos_l, pos_r, pos_l1, pos_r1) = if RELEASED {
                        let f = |base: usize, last: usize, end: usize| {
                            base.wrapping_sub(last).wrapping_add(end)
                        };
                        (
                            f(base_l, hot.last_l, hot.end_l),
                            f(base_r, hot.last_r, hot.end_r),
                            f(base_l + 1, hot.last_l, hot.end_l),
                            f(base_r + 1, hot.last_r, hot.end_r),
                        )
                    } else if MODE_TAG == HOT_MODE_NOLOOP {
                        (base_l, base_r, base_l + 1, base_r + 1)
                    } else {
                        let w = |base: usize, end: usize| {
                            if base > end {
                                wrap_pos(base, hot.start, end)
                            } else {
                                base
                            }
                        };
                        (
                            w(base_l, hot.end_l),
                            w(base_r, hot.end_r),
                            w(base_l + 1, hot.end_l),
                            w(base_r + 1, hot.end_r),
                        )
                    };

                    let (sl, sr) = if LINEAR {
                        let frac = (t - index as f64) as f32;
                        let l0 = load_sample!(hot, pos_l, hot.ptr_l, hot.len_l);
                        let l1 = load_sample!(hot, pos_l1, hot.ptr_l, hot.len_l);
                        let r0 = load_sample!(hot, pos_r, hot.ptr_r, hot.len_r);
                        let r1 = load_sample!(hot, pos_r1, hot.ptr_r, hot.len_r);
                        (l0 * (1.0 - frac) + l1 * frac, r0 * (1.0 - frac) + r1 * frac)
                    } else {
                        (
                            load_sample!(hot, pos_l, hot.ptr_l, hot.len_l),
                            load_sample!(hot, pos_r, hot.ptr_r, hot.len_r),
                        )
                    };

                    let env = lanes.get_unchecked_mut(k).env.next_frame_simd::<S>();
                    *sl_arr.get_unchecked_mut(k) = env * (hot.gain_l * sl);
                    *sr_arr.get_unchecked_mut(k) = env * (hot.gain_r * sr);
                }
            }
            let sl = S::Vf32::load_from_slice(&sl_arr);
            let sr = S::Vf32::load_from_slice(&sr_arr);
            let (sl, sr) = filter.process(sl, sr);
            let lsum = sl.horizontal_add();
            let rsum = sr.horizontal_add();
            unsafe {
                *out.get_unchecked_mut(2 * step) += lsum;
                *out.get_unchecked_mut(2 * step + 1) += rsum;
            }
        }

        filter.store(lanes, width);
        for k in 0..width {
            lanes[k].apply_hot(&hot_lanes[k]);
        }
    });
}

/// 批处理 voice：持有 [`BatchLane`]，`Voice` 接口直通 lane 状态。
pub(crate) struct StereoBatchVoice {
    lane: BatchLane,
}

impl StereoBatchVoice {
    pub(crate) fn new(init: &BatchLaneInit, control: &VoiceControlData) -> StereoBatchVoice {
        StereoBatchVoice {
            lane: BatchLane::new(init, control),
        }
    }
}

impl VoiceGeneratorBase for StereoBatchVoice {
    #[inline(always)]
    fn ended(&self) -> bool {
        self.lane.env.ended() || self.lane.past_end()
    }

    #[inline(always)]
    fn signal_release(&mut self, rel_type: ReleaseType) {
        self.lane.signal_release(rel_type);
    }

    #[inline(always)]
    fn process_controls(&mut self, control: &VoiceControlData) {
        self.lane.process_controls(control);
    }
}

impl VoiceSampleGenerator for StereoBatchVoice {
    fn render_to(&mut self, buffer: &mut [f32]) {
        self.lane.render_scalar(buffer);
    }
}

impl Voice for StereoBatchVoice {
    #[inline(always)]
    fn is_releasing(&self) -> bool {
        self.lane.releasing
    }

    #[inline(always)]
    fn is_killed(&self) -> bool {
        self.lane.killed
    }

    #[inline(always)]
    fn velocity(&self) -> u8 {
        self.lane.velocity
    }

    #[inline(always)]
    fn exclusive_class(&self) -> Option<u8> {
        self.lane.exclusive_class
    }

    #[inline(always)]
    fn batch_lane(&mut self) -> Option<&mut BatchLane> {
        Some(&mut self.lane)
    }
}
