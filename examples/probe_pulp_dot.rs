//! Step 0a of the v1.2 plan: validate that `pulp` delivers the inner
//! microkernel performance we need before committing to 7-11 weeks of
//! pure-Rust SIMD work.
//!
//! Run with:
//!
//!     cargo run --release --example probe_pulp_dot --no-default-features
//!
//! Output: ns/op for f32 dot product implemented four (or three, depending
//! on host arch) ways, at d ∈ {128, 384, 1536}.
//!
//! Decision thresholds (from `docs/v1.2-execution-plan.md`):
//! - pulp within 15% of std::arch dot at d=1536 → pulp is primary path.
//! - 15-30% behind → pulp primary, plan a hand-tuned hot-path variant for v1.3.
//! - >30% behind → fall back to raw std::arch for v1.2.
//!
//! Note on pulp API: this probe uses the pulp 0.22 `WithSimd` trait. If
//! method names need tweaking, fix them — the rest of the harness (scalar,
//! std::arch impls, faer comparison) is independent and will still produce
//! a useful measurement even if the pulp impl can't compile on first try.

use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use pulp::{Simd, WithSimd};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const DIMS: &[usize] = &[128, 384, 1536];

// Number of dot-product invocations per measurement. Tuned so that even
// the smallest d=128 case takes a measurable wall time on Apple Silicon.
const REPS: usize = 1_000_000;

// -- Scalar reference ---------------------------------------------------

#[inline(never)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += av * bv;
    }
    acc
}

// -- Pulp ---------------------------------------------------------------

struct DotKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<'a> WithSimd for DotKernel<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);

        let mut acc = simd.splat_f32s(0.0);
        for (av, bv) in a_head.iter().zip(b_head.iter()) {
            acc = simd.mul_add_e_f32s(*av, *bv, acc);
        }
        let mut sum = simd.reduce_sum_f32s(acc);
        for (av, bv) in a_tail.iter().zip(b_tail.iter()) {
            sum += av * bv;
        }
        sum
    }
}

#[inline(never)]
fn dot_pulp(arch: &pulp::Arch, a: &[f32], b: &[f32]) -> f32 {
    arch.dispatch(DotKernel { a, b })
}

/// Pulp dot with 4 independent accumulators (matching the std::arch
/// implementations). This is the apples-to-apples comparison — a naive
/// single-accumulator pulp impl is bottlenecked on FMA latency.
struct DotKernelUnrolled<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<'a> WithSimd for DotKernelUnrolled<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);

        let mut acc0 = simd.splat_f32s(0.0);
        let mut acc1 = simd.splat_f32s(0.0);
        let mut acc2 = simd.splat_f32s(0.0);
        let mut acc3 = simd.splat_f32s(0.0);

        let mut ia = a_head.chunks_exact(4);
        let mut ib = b_head.chunks_exact(4);
        for (ac, bc) in (&mut ia).zip(&mut ib) {
            acc0 = simd.mul_add_e_f32s(ac[0], bc[0], acc0);
            acc1 = simd.mul_add_e_f32s(ac[1], bc[1], acc1);
            acc2 = simd.mul_add_e_f32s(ac[2], bc[2], acc2);
            acc3 = simd.mul_add_e_f32s(ac[3], bc[3], acc3);
        }

        let mut sum = simd.reduce_sum_f32s(acc0)
            + simd.reduce_sum_f32s(acc1)
            + simd.reduce_sum_f32s(acc2)
            + simd.reduce_sum_f32s(acc3);

        // Vector chunks tail (remaining < 4 vector-chunks).
        for (av, bv) in ia.remainder().iter().zip(ib.remainder().iter()) {
            let acc = simd.splat_f32s(0.0);
            sum += simd.reduce_sum_f32s(simd.mul_add_e_f32s(*av, *bv, acc));
        }

        // Scalar tail.
        for (av, bv) in a_tail.iter().zip(b_tail.iter()) {
            sum += av * bv;
        }
        sum
    }
}

#[inline(never)]
fn dot_pulp_unrolled(arch: &pulp::Arch, a: &[f32], b: &[f32]) -> f32 {
    arch.dispatch(DotKernelUnrolled { a, b })
}

// -- AVX2 + FMA (x86_64) ------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = _mm256_setzero_ps();

    let chunks = n / 8;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
        let av = _mm256_loadu_ps(ap.add(i * 8));
        let bv = _mm256_loadu_ps(bp.add(i * 8));
        acc = _mm256_fmadd_ps(av, bv, acc);
    }

    // Horizontal sum of 8 lanes.
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    let mut sum = _mm_cvtss_f32(sum32);

    for i in (chunks * 8)..n {
        sum += a[i] * b[i];
    }
    sum
}

// -- NEON (aarch64) ----------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(never)]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    // Four independent accumulators to hide FMA latency.
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    let chunks = n / 16;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
        let off = i * 16;
        let av0 = vld1q_f32(ap.add(off));
        let av1 = vld1q_f32(ap.add(off + 4));
        let av2 = vld1q_f32(ap.add(off + 8));
        let av3 = vld1q_f32(ap.add(off + 12));
        let bv0 = vld1q_f32(bp.add(off));
        let bv1 = vld1q_f32(bp.add(off + 4));
        let bv2 = vld1q_f32(bp.add(off + 8));
        let bv3 = vld1q_f32(bp.add(off + 12));
        acc0 = vfmaq_f32(acc0, av0, bv0);
        acc1 = vfmaq_f32(acc1, av1, bv1);
        acc2 = vfmaq_f32(acc2, av2, bv2);
        acc3 = vfmaq_f32(acc3, av3, bv3);
    }

    let acc01 = vaddq_f32(acc0, acc1);
    let acc23 = vaddq_f32(acc2, acc3);
    let acc = vaddq_f32(acc01, acc23);
    let mut sum = vaddvq_f32(acc);

    // 4-wide tail.
    let mut tail_start = chunks * 16;
    let four_chunks = (n - tail_start) / 4;
    for i in 0..four_chunks {
        let off = tail_start + i * 4;
        let av = vld1q_f32(ap.add(off));
        let bv = vld1q_f32(bp.add(off));
        let prod = vmulq_f32(av, bv);
        sum += vaddvq_f32(prod);
    }
    tail_start += four_chunks * 4;
    for i in tail_start..n {
        sum += a[i] * b[i];
    }
    sum
}

// -- faer matmul shaped as 1×d × d×1 (production path today) -----------

#[inline(never)]
fn dot_faer(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let d = a.len();
    let q = MatRef::from_row_major_slice(a, 1, d);
    let x = MatRef::from_row_major_slice(b, 1, d);
    let mut out = [0.0f32; 1];
    let out_mut = MatMut::from_row_major_slice_mut(&mut out, 1, 1);
    matmul(out_mut, Accum::Replace, q, x.transpose(), 1.0, Par::Seq);
    out[0]
}

// -- Harness -----------------------------------------------------------

fn rand_vec(d: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..d).map(|_| rng.gen_range(-1.0f32..1.0)).collect()
}

fn ns_per_op(elapsed_ns: u128, reps: usize) -> f64 {
    elapsed_ns as f64 / reps as f64
}

fn time_dot(
    name: &str,
    reps: usize,
    a: &[f32],
    b: &[f32],
    mut f: impl FnMut(&[f32], &[f32]) -> f32,
) -> (f64, f32) {
    use std::hint::black_box;

    // Warmup.
    let _ = f(a, b);
    let _ = f(a, b);

    let t0 = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..reps {
        let av = black_box(a);
        let bv = black_box(b);
        acc += black_box(f(av, bv));
    }
    let _ = black_box(acc);
    let elapsed = t0.elapsed().as_nanos();
    let ns = ns_per_op(elapsed, reps);
    println!("    {:14} {:>9.2} ns/op", name, ns);
    (ns, acc)
}

fn verify(d: usize) {
    let a = rand_vec(d, 1);
    let b = rand_vec(d, 2);
    let arch = pulp::Arch::new();

    let s = dot_scalar(&a, &b);
    let p = dot_pulp(&arch, &a, &b);
    let pu = dot_pulp_unrolled(&arch, &a, &b);
    let f = dot_faer(&a, &b);

    let tol = 1e-3 * d as f32;
    assert!((s - p).abs() < tol, "scalar vs pulp mismatch at d={d}: {s} vs {p}");
    assert!((s - pu).abs() < tol, "scalar vs pulp_unrolled mismatch at d={d}: {s} vs {pu}");
    assert!((s - f).abs() < tol, "scalar vs faer mismatch at d={d}: {s} vs {f}");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let v = unsafe { dot_avx2_fma(&a, &b) };
            assert!(
                (s - v).abs() < tol,
                "scalar vs avx2 mismatch at d={d}: {s} vs {v}"
            );
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let v = unsafe { dot_neon(&a, &b) };
        assert!(
            (s - v).abs() < tol,
            "scalar vs neon mismatch at d={d}: {s} vs {v}"
        );
    }
}

fn main() {
    println!("Step 0a: pulp microkernel validation");
    println!("--------------------------------------");
    println!("Reps per measurement: {}\n", REPS);

    // Sanity check all impls agree at every dim.
    for &d in DIMS {
        verify(d);
    }
    println!("Correctness check: all impls agree to within float tolerance.\n");

    let arch = pulp::Arch::new();

    let mut summary: Vec<(usize, f64, f64, f64, Option<f64>, f64)> = Vec::new();

    for &d in DIMS {
        let a = rand_vec(d, 1);
        let b = rand_vec(d, 2);

        println!("=== d = {d} ===");
        let (ns_scalar, _) = time_dot("scalar", REPS, &a, &b, |a, b| dot_scalar(a, b));
        let (ns_pulp, _) = time_dot("pulp (naive)", REPS, &a, &b, |a, b| dot_pulp(&arch, a, b));
        let (ns_pulp_u, _) = time_dot("pulp x4", REPS, &a, &b, |a, b| dot_pulp_unrolled(&arch, a, b));

        let ns_intrinsics = {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    let (ns, _) =
                        time_dot("avx2+fma", REPS, &a, &b, |a, b| unsafe { dot_avx2_fma(a, b) });
                    Some(ns)
                } else {
                    None
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                let (ns, _) = time_dot("neon", REPS, &a, &b, |a, b| unsafe { dot_neon(a, b) });
                Some(ns)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                None
            }
        };

        let (ns_faer, _) = time_dot("faer matmul", REPS, &a, &b, |a, b| dot_faer(a, b));

        // Ratios vs scalar.
        let pulp_speedup = ns_scalar / ns_pulp;
        let pulp_u_speedup = ns_scalar / ns_pulp_u;
        let faer_speedup = ns_scalar / ns_faer;
        println!();
        println!("    pulp (naive) vs scalar:  {:.2}x", pulp_speedup);
        println!("    pulp x4      vs scalar:  {:.2}x", pulp_u_speedup);
        if let Some(ns) = ns_intrinsics {
            let intr_speedup = ns_scalar / ns;
            println!("    intrinsics   vs scalar:  {:.2}x", intr_speedup);
            let ratio_naive = ns_pulp / ns;
            let ratio_unrolled = ns_pulp_u / ns;
            println!(
                "    pulp (naive) / intrinsics: {:.3}  ({})",
                ratio_naive,
                bucket(ratio_naive)
            );
            println!(
                "    pulp x4      / intrinsics: {:.3}  ({})",
                ratio_unrolled,
                bucket(ratio_unrolled)
            );
        }
        println!("    faer matmul  vs scalar:  {:.2}x", faer_speedup);
        println!();

        summary.push((d, ns_scalar, ns_pulp, ns_pulp_u, ns_intrinsics, ns_faer));
    }

    println!("Summary (ns/op):");
    println!(
        "  {:>5}  {:>10}  {:>14}  {:>10}  {:>12}  {:>10}",
        "d", "scalar", "pulp (naive)", "pulp x4", "intrinsics", "faer"
    );
    for (d, s, p, pu, intr, f) in &summary {
        let intr_str = match intr {
            Some(v) => format!("{:.2}", v),
            None => "n/a".to_string(),
        };
        println!(
            "  {:>5}  {:>10.2}  {:>14.2}  {:>10.2}  {:>12}  {:>10.2}",
            d, s, p, pu, intr_str, f
        );
    }

    println!("\nDecision (based on d=1536 row, pulp x4 vs intrinsics):");
    println!("  - within 15%   → Plan A (pulp + fused kernel, 7-9 weeks)");
    println!("  - 15-30% behind → Plan A with hot-path variant for v1.3");
    println!("  - >30% behind  → Plan B (raw std::arch, 16-20 weeks)");
    println!("\nDocument the result in docs/v1.2-prototype-result.md.");
}

fn bucket(ratio: f64) -> &'static str {
    if ratio <= 1.15 {
        "within 15% — pulp viable"
    } else if ratio <= 1.30 {
        "15-30% behind — pulp primary, plan hot-path variant"
    } else {
        ">30% behind — fall back to std::arch"
    }
}
