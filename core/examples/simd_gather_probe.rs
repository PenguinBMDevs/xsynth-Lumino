//! SIMD 调研微基准：线性插值采样中「逐 lane 标量加载」vs「AVX2 硬件 gather」。
//!
//! 用法：`cargo run -p xsynth-core --release --example simd_gather_probe`
//!
//! 目的：为「是否值得把 SIMDLinearSampleGrabber 改为 i32gather_ps」提供本机数据。
//! 两种访问模式：
//! - sequential：索引近似连续（speed≈1 的钢琴循环采样，生产主路径）；
//! - random：索引随机散布（极端音高/大样本库）。
//!
//! 输出：每 8 样本窗口的耗时（ns）与相对加速比。仅研究用，不参与生产路径。

use std::hint::black_box;
use std::time::Instant;

#[cfg(target_arch = "x86_64")]
mod avx2_probe {
    use std::arch::x86_64::*;
    use std::hint::black_box;

    /// 硬件 gather 版：每 8 样本窗口 2 次 `vgatherdps` + 混合。
    #[target_feature(enable = "avx2")]
    pub unsafe fn time_gather(buf: &[f32], idxs: &[[i32; 8]], reps: usize) -> f64 {
        let base = buf.as_ptr();
        let t0 = std::time::Instant::now();
        let mut acc = _mm256_setzero_ps();
        for _ in 0..reps {
            for idx in idxs {
                let v = _mm256_loadu_si256(idx.as_ptr() as *const __m256i);
                let a = _mm256_i32gather_ps::<4>(base, v);
                let next = _mm256_add_epi32(v, _mm256_set1_epi32(1));
                let b = _mm256_i32gather_ps::<4>(base, next);
                acc = _mm256_add_ps(acc, _mm256_add_ps(a, b));
            }
        }
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), acc);
        black_box(out);
        t0.elapsed().as_secs_f64()
    }
}

/// 标量版：8 个 lane 各 2 次非越界加载 + 混合（与生产 grabber 同构，去掉 reader 逻辑）。
fn time_scalar(buf: &[f32], idxs: &[[i32; 8]], reps: usize) -> f64 {
    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..reps {
        for idx in idxs {
            for &i in idx.iter() {
                let i = i as usize;
                let a = unsafe { *buf.get_unchecked(i) };
                let b = unsafe { *buf.get_unchecked(i + 1) };
                acc += a * 0.7 + b * 0.3;
            }
        }
    }
    black_box(acc);
    t0.elapsed().as_secs_f64()
}

#[cfg(target_arch = "x86_64")]
mod mix_probe {
    use std::arch::x86_64::*;
    use std::hint::black_box;

    /// 现状：逐 lane 标量 RMW（与 `SIMDStereoVoice::render_to` 整块路径同构）。
    #[target_feature(enable = "avx2")]
    pub unsafe fn mix_scalar(out: &mut [f32], l: &[f32; 8], r: &[f32; 8], reps: usize) {
        for _ in 0..reps {
            for k in 0..8 {
                *out.get_unchecked_mut(2 * k) += *l.get_unchecked(k);
                *out.get_unchecked_mut(2 * k + 1) += *r.get_unchecked(k);
            }
        }
        black_box(out.as_ptr());
    }

    /// 候选：两向量 unpack 成交织布局，2 次 256-bit load/add/store。
    #[target_feature(enable = "avx2")]
    pub unsafe fn mix_unpack(out: &mut [f32], lv: __m256, rv: __m256, reps: usize) {
        let lo = _mm256_unpacklo_ps(lv, rv);
        let hi = _mm256_unpackhi_ps(lv, rv);
        for _ in 0..reps {
            let a = _mm256_loadu_ps(out.as_ptr());
            let b = _mm256_loadu_ps(out.as_ptr().add(8));
            _mm256_storeu_ps(out.as_mut_ptr(), _mm256_add_ps(a, lo));
            _mm256_storeu_ps(out.as_mut_ptr().add(8), _mm256_add_ps(b, hi));
        }
        black_box(out.as_ptr());
    }
}

#[cfg(target_arch = "x86_64")]
mod reg_probe {
    use std::arch::x86_64::*;
    use std::hint::black_box;

    /// gather：索引驻留寄存器（与生产一致），每轮 2 次 gather。
    #[target_feature(enable = "avx2")]
    pub unsafe fn time_gather_reg(buf: &[f32], idx0: __m256i, reps: usize) -> f64 {
        let base = buf.as_ptr();
        let one = _mm256_set1_epi32(1);
        let mask = _mm256_set1_epi32(1023);
        let mut idx = idx0;
        let t0 = std::time::Instant::now();
        let mut acc = _mm256_setzero_ps();
        for _ in 0..reps {
            let next = _mm256_add_epi32(idx, one);
            let a = _mm256_i32gather_ps::<4>(base, idx);
            let b = _mm256_i32gather_ps::<4>(base, next);
            acc = _mm256_add_ps(acc, _mm256_add_ps(a, b));
            idx = _mm256_and_si256(_mm256_add_epi32(idx, one), mask);
        }
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), acc);
        black_box(out);
        t0.elapsed().as_secs_f64()
    }

    /// 标量：索引驻留栈/寄存器，每轮 16 次加载（公平对照）。
    #[target_feature(enable = "avx2")]
    pub unsafe fn time_scalar_reg(buf: &[f32], idx0: __m256i, reps: usize) -> f64 {
        let mut arr = [0i32; 8];
        _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, idx0);
        let t0 = std::time::Instant::now();
        let mut acc = 0.0f32;
        for i in 0..reps {
            let bump = (i & 1) as i32;
            let mut s = 0.0f32;
            for k in 0..8 {
                let p = ((*arr.get_unchecked(k) + bump) as usize) & 1023;
                s += *buf.get_unchecked(p) + *buf.get_unchecked(p + 1);
            }
            acc += s;
        }
        black_box(acc);
        t0.elapsed().as_secs_f64()
    }
}

fn time_mix_scalar(l: &[f32; 8], r: &[f32; 8], reps: usize) -> f64 {
    let mut out = vec![0.0f32; 16];
    let t0 = Instant::now();
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { mix_probe::mix_scalar(&mut out, l, r, reps) };
        return t0.elapsed().as_secs_f64();
    }
    for _ in 0..reps {
        for (k, (&a, &b)) in l.iter().zip(r.iter()).enumerate() {
            out[2 * k] += a;
            out[2 * k + 1] += b;
        }
    }
    black_box(&out);
    t0.elapsed().as_secs_f64()
}

fn time_mix_unpack(l: &[f32; 8], r: &[f32; 8], reps: usize) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        if !std::is_x86_feature_detected!("avx2") {
            return None;
        }
        use std::arch::x86_64::*;
        let mut out = vec![0.0f32; 16];
        let lv = unsafe { _mm256_loadu_ps(l.as_ptr()) };
        let rv = unsafe { _mm256_loadu_ps(r.as_ptr()) };
        let t0 = Instant::now();
        unsafe { mix_probe::mix_unpack(&mut out, lv, rv, reps) };
        Some(t0.elapsed().as_secs_f64())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (l, r, reps);
        None
    }
}

fn build_indices(mode: &str, n: usize, len: usize) -> Vec<[i32; 8]> {
    let mut out = Vec::with_capacity(n);
    let mut pos = 0usize;
    let mut seed = 0x9e37_79b9u32;
    for _ in 0..n {
        let mut idx = [0i32; 8];
        for slot in idx.iter_mut() {
            if mode == "sequential" {
                *slot = pos as i32;
                pos += 1;
                if pos + 2 >= len {
                    pos = 0;
                }
            } else {
                // xorshift 伪随机
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                *slot = (seed as usize % (len - 2)) as i32;
            }
        }
        out.push(idx);
    }
    out
}

fn time_gather(buf: &[f32], idxs: &[[i32; 8]], reps: usize) -> Option<f64> {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: 已检测 avx2；索引由构造保证 < len-2。
            return Some(unsafe { avx2_probe::time_gather(buf, idxs, reps) });
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (buf, idxs, reps);
        None
    }
}

fn main() {
    let len = 1 << 16; // 64K 样本 ≈ 256KB（L2 常驻），模拟钢琴循环采样窗口
    let buf: Vec<f32> = (0..len).map(|i| (i as f32 * 0.001).sin()).collect();
    let n = 2_000_000usize;
    let reps = 1usize;

    println!("buf={len} samples, windows={n}");
    for mode in ["sequential", "random"] {
        let idxs = build_indices(mode, n, len);
        let scalar = time_scalar(&buf, &idxs, reps);
        let per_window_scalar = scalar / (n as f64) * 1e9;
        match time_gather(&buf, &idxs, reps) {
            Some(g) => {
                let per_window_gather = g / (n as f64) * 1e9;
                println!(
                    "{mode:>10}: scalar={per_window_scalar:6.2} ns/8窗口  gather={per_window_gather:6.2} ns/8窗口  speedup={:.2}x",
                    per_window_scalar / per_window_gather
                );
            }
            None => println!("{mode:>10}: scalar={per_window_scalar:6.2} ns/8窗口  gather=N/A"),
        }
    }

    // 混音写回：现状（标量逐 lane RMW）vs 候选（unpack 交织 + 2 次向量 RMW）。
    let l = [0.1f32; 8];
    let r = [0.2f32; 8];
    let mix_reps = 20_000_000usize;
    let ms = time_mix_scalar(&l, &r, mix_reps) / mix_reps as f64 * 1e9;
    match time_mix_unpack(&l, &r, mix_reps) {
        Some(mu) => {
            let mu_ns = mu / mix_reps as f64 * 1e9;
            println!(
                "mix/8帧: scalar={ms:5.2} ns  unpack={mu_ns:5.2} ns  speedup={:.2}x",
                ms / mu_ns
            );
        }
        None => println!("mix/8帧: scalar={ms:5.2} ns  unpack=N/A"),
    }

    // 公平对照：索引驻留寄存器（与生产路径一致），标量 16 次加载 vs 2 次 gather。
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        use std::arch::x86_64::*;
        let small: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
        let idx = unsafe { _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7) };
        let reps = 20_000_000usize;
        let s = unsafe { reg_probe::time_scalar_reg(&small, idx, reps) } / reps as f64 * 1e9;
        let g = unsafe { reg_probe::time_gather_reg(&small, idx, reps) } / reps as f64 * 1e9;
        println!(
            "reg-index/8帧: scalar={s:5.2} ns  gather={g:5.2} ns  speedup={:.2}x",
            s / g
        );
    }
}
