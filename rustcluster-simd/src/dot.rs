//! f32 dot product — pulp-dispatched SIMD with a 4-accumulator inner
//! loop, plus a scalar reference for tests and platforms without SIMD.
//!
//! The 4-accumulator structure was validated against hand-tuned
//! `std::arch` NEON in v1.2 Step 0a (`docs/v1.2-prototype-result.md`):
//! at d=1536 on Apple Silicon, this kernel runs within 2.7% of the
//! reference NEON implementation. A single-accumulator pulp impl is
//! 3.47× slower — FMA latency, not throughput, is the binding
//! constraint with one accumulator.

use pulp::{Simd, WithSimd};

/// Scalar reference dot product. Used for property tests and as the
/// fallback path on platforms without a SIMD implementation.
#[inline]
pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_f32_scalar: length mismatch");
    let mut acc = 0.0f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        acc += av * bv;
    }
    acc
}

/// SIMD f32 dot product (pulp-dispatched at runtime).
///
/// Uses 4 independent accumulators to hide FMA latency. On Apple
/// Silicon (NEON, f32×4) this processes 16 elements per inner-loop
/// iteration; on AVX2 (f32×8) it processes 32; on AVX-512 (f32×16)
/// it processes 64.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");
    pulp::Arch::new().dispatch(DotKernel { a, b })
}

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

        // Vector chunks tail: any remaining < 4 vector-chunks.
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn close(scalar: f32, simd: f32, a: &[f32], b: &[f32]) -> bool {
        // Wilkinson-style error bound for FP dot product: the
        // difference between two valid evaluation orders of
        // sum(a_i * b_i) is bounded by approximately n·eps·sum|a_i·b_i|.
        // Scalar is sequential, SIMD is multi-accumulator; both are
        // valid, but their rounding patterns diverge. We use 2x the
        // theoretical bound as headroom against adversarial cases,
        // plus a small absolute floor.
        let n = a.len();
        let abs_sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x * y).abs()).sum();
        let tol = 2.0 * (n as f32) * f32::EPSILON * abs_sum + 1e-4;
        (scalar - simd).abs() <= tol
    }

    #[test]
    fn empty() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
        assert_eq!(dot_f32_scalar(&[], &[]), 0.0);
    }

    #[test]
    fn length_one() {
        assert_eq!(dot_f32(&[2.0], &[3.0]), 6.0);
    }

    #[test]
    fn known_value_short() {
        // 1·5 + 2·6 + 3·7 + 4·8 = 70
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(dot_f32(&a, &b), 70.0);
        assert_eq!(dot_f32_scalar(&a, &b), 70.0);
    }

    #[test]
    fn aligned_lengths() {
        // Lengths at vector-width and 4-chunk multiples — these are
        // the boundary cases where head/tail/remainder splits land
        // exactly. Test all of them.
        for &n in &[4usize, 8, 16, 32, 64, 128, 384, 1536] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) - n as f32 / 2.0) * 0.02).collect();
            let s = dot_f32_scalar(&a, &b);
            let v = dot_f32(&a, &b);
            assert!(close(s, v, &a, &b), "n={n}: scalar={s} simd={v}");
        }
    }

    proptest! {
        #[test]
        fn dot_f32_matches_scalar(
            (a, b) in (0usize..2000).prop_flat_map(|n| {
                (
                    prop::collection::vec(-100.0f32..100.0, n),
                    prop::collection::vec(-100.0f32..100.0, n),
                )
            }),
        ) {
            let s = dot_f32_scalar(&a, &b);
            let v = dot_f32(&a, &b);
            prop_assert!(
                close(s, v, &a, &b),
                "n={} scalar={} simd={}", a.len(), s, v
            );
        }
    }
}
