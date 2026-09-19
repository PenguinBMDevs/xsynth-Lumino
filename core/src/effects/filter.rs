use crate::channel::ValueLerp;
use biquad::*;
use simdeez::prelude::*;
pub use xsynth_soundfonts::FilterType;

#[derive(Clone)]
pub(crate) struct BiQuadFilter {
    coeffs: Coefficients<f32>,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl BiQuadFilter {
    pub fn new(fil_type: FilterType, freq: f32, sample_rate: f32, q: Option<f32>) -> Self {
        let coeffs = Self::get_coeffs(fil_type, freq, sample_rate, q);

        Self {
            coeffs,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn get_coeffs(
        fil_type: FilterType,
        freq: f32,
        sample_rate: f32,
        q: Option<f32>,
    ) -> Coefficients<f32> {
        let q = match q {
            Some(q) => q,
            None => Q_BUTTERWORTH_F32,
        };
        let freq = sanitize_freq(freq, sample_rate);

        match fil_type {
            FilterType::LowPass => {
                Coefficients::<f32>::from_params(Type::LowPass, sample_rate.hz(), freq.hz(), q)
                    .unwrap()
            }
            FilterType::LowPassPole => Coefficients::<f32>::from_params(
                Type::SinglePoleLowPass,
                sample_rate.hz(),
                freq.hz(),
                q,
            )
            .unwrap(),
            FilterType::HighPass => {
                Coefficients::<f32>::from_params(Type::HighPass, sample_rate.hz(), freq.hz(), q)
                    .unwrap()
            }
            FilterType::BandPass => {
                Coefficients::<f32>::from_params(Type::BandPass, sample_rate.hz(), freq.hz(), q)
                    .unwrap()
            }
        }
    }

    pub fn set_coefficients(&mut self, coeffs: Coefficients<f32>) {
        self.coeffs = coeffs;
    }

    /// 读取当前系数（B1 批处理 lane 在 spawn 期快照系数；B0 原型同样使用）。
    pub(crate) fn coefficients(&self) -> &Coefficients<f32> {
        &self.coeffs
    }

    /// 直接形式 I biquad（与 `biquad` crate `DirectForm1` 的公式逐项一致）。
    ///
    /// 原实现调用 `biquad::DirectForm1::run`：跨 crate 且无 `#[inline]`，
    /// 每个样本一次真实函数调用（每 voice 每 8 帧 16 次，逐声部滤波的主要开销）。
    /// 就地实现后可被本 crate 内联；运算顺序不变（无 FMA contraction），输出逐位一致。
    #[inline(always)]
    pub fn process(&mut self, input: f32) -> f32 {
        let out = self.coeffs.b0 * input + self.coeffs.b1 * self.x1 + self.coeffs.b2 * self.x2
            - self.coeffs.a1 * self.y1
            - self.coeffs.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = out;

        out
    }

    #[inline(always)]
    pub fn process_simd<S: Simd>(&mut self, input: S::Vf32) -> S::Vf32 {
        let mut out = input;
        for i in 0..S::Vf32::WIDTH {
            out[i] = self.process(input[i]);
        }
        out
    }
}

fn sanitize_freq(freq: f32, sample_rate: f32) -> f32 {
    let nyquist = (sample_rate * 0.5).max(1.0);
    let max_freq = (nyquist - 1.0).max(1.0);
    freq.clamp(1.0, max_freq)
}

/// A multi-channel bi-quad audio filter.
///
/// Supports single pole low pass filter and two pole low pass, high pass
/// and band pass filters. For more information please see the `FilterType`
/// documentation.
///
/// Uses the `biquad` crate for signal processing.
pub struct MultiChannelBiQuad {
    channels: Vec<BiQuadFilter>,
    fil_type: FilterType,
    value: ValueLerp,
    q: Option<f32>,
    sample_rate: f32,
    /// 当前已生效系数的输入键 `(freq_bits, q_bits, fil_type)`：
    /// 仅当输入变化时才重算系数（`get_coeffs` 含三角函数，逐帧重算是纯浪费）。
    cached: Option<(u32, Option<u32>, FilterType)>,
}

impl MultiChannelBiQuad {
    /// Creates a new audio filter with the given parameters.
    ///
    /// - `channels`: Number of audio channels
    /// - `fil_type`: Type of the audio filter. See FilterType docs
    /// - `freq`: Cutoff frequency
    /// - `sample_rate`: Sample rate of the audio to be processed
    /// - `q`: The Q parameter of the cutoff filter. Use None for the default
    ///   Butterworth value.
    pub fn new(
        channels: usize,
        fil_type: FilterType,
        freq: f32,
        sample_rate: f32,
        q: Option<f32>,
    ) -> Self {
        Self {
            channels: (0..channels)
                .map(|_| BiQuadFilter::new(fil_type, freq, sample_rate, q))
                .collect(),
            fil_type,
            value: ValueLerp::new(freq, sample_rate as u32),
            q,
            sample_rate,
            cached: None,
        }
    }

    /// Changes the type of the audio filter.
    pub fn set_filter_type(&mut self, fil_type: FilterType, freq: f32, q: Option<f32>) {
        self.value.set_end(freq);
        // 类型/Q 变化必须失效系数缓存（频率由 `set_coefficients` 按键比较处理）。
        if self.fil_type != fil_type || self.q != q {
            self.cached = None;
        }
        self.fil_type = fil_type;
        self.q = q;
    }

    fn set_coefficients(&mut self, freq: f32, q: Option<f32>) {
        let key = (freq.to_bits(), q.map(f32::to_bits), self.fil_type);
        if self.cached == Some(key) {
            return;
        }
        let coeffs = BiQuadFilter::get_coeffs(self.fil_type, freq, self.sample_rate, q);
        for filter in self.channels.iter_mut() {
            filter.set_coefficients(coeffs);
        }
        self.cached = Some(key);
    }

    /// Filters the audio of the given sample buffer.
    pub fn process(&mut self, sample: &mut [f32]) {
        let channel_count = self.channels.len();
        for (i, s) in sample.iter_mut().enumerate() {
            if i % channel_count == 0 {
                let v = self.value.get_next();
                self.set_coefficients(v, self.q);
            }
            *s = self.channels[i % channel_count].process(*s);
        }
    }
}
