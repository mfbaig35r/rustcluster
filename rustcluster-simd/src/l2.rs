//! f32 squared L2 distance — single-pass direct kernel.
//!
//! Computes `sum((a_i - b_i)^2)` in a single pass over both inputs.
//! Pulp 0.22 has no `sub_f32s`, so the per-chunk difference uses
//! `mul_add_e_f32s(b, splat(-1.0), a) = -1·b + a = a - b`. The
//! `-1·b` product is exact (sign flip), so this is bit-exact with a
//! native subtract.
//!
//! For batched L2 search, the production code in
//! `rustcluster::index::kernel` uses the identity
//! `‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b` with precomputed norms — strictly
//! cheaper there because each norm is reused across many queries.
//! This kernel is for single-pair use; identity-based batched L2
//! lives in `kernel.rs`.

use pulp::{Simd, WithSimd};

#[inline]
pub fn l2sq_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2sq_f32_scalar: length mismatch");
    let mut acc = 0.0f32;
    for (av, bv) in a.iter().zip(b.iter()) {
        let d = av - bv;
        acc += d * d;
    }
    acc
}

/// SIMD f32 squared L2 distance (pulp-dispatched at runtime).
///
/// Same 4-accumulator structure as `dot_f32`. Per inner-loop iteration
/// it processes 16 elements (NEON), 32 (AVX2), or 64 (AVX-512).
#[inline]
pub fn l2sq_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "l2sq_f32: length mismatch");
    pulp::Arch::new().dispatch(L2sqKernel { a, b })
}

struct L2sqKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<'a> WithSimd for L2sqKernel<'a> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);

        let neg_one = simd.splat_f32s(-1.0);
        let mut acc0 = simd.splat_f32s(0.0);
        let mut acc1 = simd.splat_f32s(0.0);
        let mut acc2 = simd.splat_f32s(0.0);
        let mut acc3 = simd.splat_f32s(0.0);

        let mut ia = a_head.chunks_exact(4);
        let mut ib = b_head.chunks_exact(4);
        for (ac, bc) in (&mut ia).zip(&mut ib) {
            let d0 = simd.mul_add_e_f32s(neg_one, bc[0], ac[0]);
            let d1 = simd.mul_add_e_f32s(neg_one, bc[1], ac[1]);
            let d2 = simd.mul_add_e_f32s(neg_one, bc[2], ac[2]);
            let d3 = simd.mul_add_e_f32s(neg_one, bc[3], ac[3]);
            acc0 = simd.mul_add_e_f32s(d0, d0, acc0);
            acc1 = simd.mul_add_e_f32s(d1, d1, acc1);
            acc2 = simd.mul_add_e_f32s(d2, d2, acc2);
            acc3 = simd.mul_add_e_f32s(d3, d3, acc3);
        }

        let mut sum = simd.reduce_sum_f32s(acc0)
            + simd.reduce_sum_f32s(acc1)
            + simd.reduce_sum_f32s(acc2)
            + simd.reduce_sum_f32s(acc3);

        // Vector chunks tail: any remaining < 4 vector-chunks.
        for (av, bv) in ia.remainder().iter().zip(ib.remainder().iter()) {
            let d = simd.mul_add_e_f32s(neg_one, *bv, *av);
            let z = simd.splat_f32s(0.0);
            sum += simd.reduce_sum_f32s(simd.mul_add_e_f32s(d, d, z));
        }

        // Scalar tail.
        for (av, bv) in a_tail.iter().zip(b_tail.iter()) {
            let d = av - bv;
            sum += d * d;
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn close(scalar: f32, simd: f32, n: usize) -> bool {
        // l2sq summands are (a_i - b_i)^2 ≥ 0, so abs_sum == result.
        // Wilkinson bound simplifies to n·eps·result. 2× safety
        // factor + small absolute floor.
        let max_mag = scalar.abs().max(simd.abs());
        let tol = 2.0 * (n as f32) * f32::EPSILON * max_mag + 1e-4;
        (scalar - simd).abs() <= tol
    }

    #[test]
    fn empty() {
        assert_eq!(l2sq_f32(&[], &[]), 0.0);
        assert_eq!(l2sq_f32_scalar(&[], &[]), 0.0);
    }

    #[test]
    fn length_one() {
        assert_eq!(l2sq_f32(&[5.0], &[2.0]), 9.0);
    }

    #[test]
    fn known_value_short() {
        // (1-5)^2 + (2-6)^2 + (3-7)^2 + (4-8)^2 = 16+16+16+16 = 64
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        assert_eq!(l2sq_f32(&a, &b), 64.0);
        assert_eq!(l2sq_f32_scalar(&a, &b), 64.0);
    }

    #[test]
    fn identical_inputs_give_zero() {
        let a: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.01 - 7.5).collect();
        assert_eq!(l2sq_f32(&a, &a), 0.0);
    }

    #[test]
    fn aligned_lengths() {
        for &n in &[4usize, 8, 16, 32, 64, 128, 384, 1536] {
            let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
            let b: Vec<f32> = (0..n).map(|i| ((i as f32) - n as f32 / 2.0) * 0.02).collect();
            let s = l2sq_f32_scalar(&a, &b);
            let v = l2sq_f32(&a, &b);
            assert!(close(s, v, n), "n={n}: scalar={s} simd={v}");
        }
    }

    proptest! {
        #[test]
        fn l2sq_f32_matches_scalar(
            (a, b) in (0usize..2000).prop_flat_map(|n| {
                (
                    prop::collection::vec(-100.0f32..100.0, n),
                    prop::collection::vec(-100.0f32..100.0, n),
                )
            }),
        ) {
            let s = l2sq_f32_scalar(&a, &b);
            let v = l2sq_f32(&a, &b);
            prop_assert!(
                close(s, v, a.len()),
                "n={} scalar={} simd={}", a.len(), s, v
            );
        }

        // L2 sanity: result is always non-negative.
        #[test]
        fn l2sq_f32_nonneg(
            (a, b) in (0usize..2000).prop_flat_map(|n| {
                (
                    prop::collection::vec(-100.0f32..100.0, n),
                    prop::collection::vec(-100.0f32..100.0, n),
                )
            }),
        ) {
            let v = l2sq_f32(&a, &b);
            prop_assert!(v >= 0.0, "negative l2sq: {}", v);
        }
    }
}
