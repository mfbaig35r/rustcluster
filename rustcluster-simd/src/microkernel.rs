//! Shared SIMD microkernels.
//!
//! `Tile4x4` is the 4-row × 4-col register-blocked dot microkernel
//! used by both `batched_dot_tile` (writes results to memory) and
//! the fused threshold-emit kernel (filters in registers, never
//! writes the score tile).
//!
//! Per inner-loop iteration: 4 q-vectors loaded + 4 x-vectors loaded
//! → 16 SIMD FMAs into 16 register accumulators. Half the
//! load-to-FMA ratio of a row-by-row dot.

use pulp::{Simd, WithSimd};

/// 4×4 register-blocked dot microkernel. Reads 4 q-rows and 4 x-rows
/// of length d, returns the 16 inner products as a flat array
/// `[a00, a01, a02, a03, a10, ..., a33]` (row-major: index = ii*4 + jj).
pub(crate) struct Tile4x4<'a> {
    pub q0: &'a [f32],
    pub q1: &'a [f32],
    pub q2: &'a [f32],
    pub q3: &'a [f32],
    pub x0: &'a [f32],
    pub x1: &'a [f32],
    pub x2: &'a [f32],
    pub x3: &'a [f32],
}

impl<'a> WithSimd for Tile4x4<'a> {
    type Output = [f32; 16];

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> [f32; 16] {
        let (q0h, q0t) = S::as_simd_f32s(self.q0);
        let (q1h, q1t) = S::as_simd_f32s(self.q1);
        let (q2h, q2t) = S::as_simd_f32s(self.q2);
        let (q3h, q3t) = S::as_simd_f32s(self.q3);
        let (x0h, x0t) = S::as_simd_f32s(self.x0);
        let (x1h, x1t) = S::as_simd_f32s(self.x1);
        let (x2h, x2t) = S::as_simd_f32s(self.x2);
        let (x3h, x3t) = S::as_simd_f32s(self.x3);

        let z = simd.splat_f32s(0.0);
        let mut a00 = z; let mut a01 = z; let mut a02 = z; let mut a03 = z;
        let mut a10 = z; let mut a11 = z; let mut a12 = z; let mut a13 = z;
        let mut a20 = z; let mut a21 = z; let mut a22 = z; let mut a23 = z;
        let mut a30 = z; let mut a31 = z; let mut a32 = z; let mut a33 = z;

        let n = q0h.len();
        for k in 0..n {
            let q0 = q0h[k]; let q1 = q1h[k]; let q2 = q2h[k]; let q3 = q3h[k];
            let x0 = x0h[k]; let x1 = x1h[k]; let x2 = x2h[k]; let x3 = x3h[k];

            a00 = simd.mul_add_e_f32s(q0, x0, a00);
            a01 = simd.mul_add_e_f32s(q0, x1, a01);
            a02 = simd.mul_add_e_f32s(q0, x2, a02);
            a03 = simd.mul_add_e_f32s(q0, x3, a03);
            a10 = simd.mul_add_e_f32s(q1, x0, a10);
            a11 = simd.mul_add_e_f32s(q1, x1, a11);
            a12 = simd.mul_add_e_f32s(q1, x2, a12);
            a13 = simd.mul_add_e_f32s(q1, x3, a13);
            a20 = simd.mul_add_e_f32s(q2, x0, a20);
            a21 = simd.mul_add_e_f32s(q2, x1, a21);
            a22 = simd.mul_add_e_f32s(q2, x2, a22);
            a23 = simd.mul_add_e_f32s(q2, x3, a23);
            a30 = simd.mul_add_e_f32s(q3, x0, a30);
            a31 = simd.mul_add_e_f32s(q3, x1, a31);
            a32 = simd.mul_add_e_f32s(q3, x2, a32);
            a33 = simd.mul_add_e_f32s(q3, x3, a33);
        }

        let mut out = [
            simd.reduce_sum_f32s(a00), simd.reduce_sum_f32s(a01),
            simd.reduce_sum_f32s(a02), simd.reduce_sum_f32s(a03),
            simd.reduce_sum_f32s(a10), simd.reduce_sum_f32s(a11),
            simd.reduce_sum_f32s(a12), simd.reduce_sum_f32s(a13),
            simd.reduce_sum_f32s(a20), simd.reduce_sum_f32s(a21),
            simd.reduce_sum_f32s(a22), simd.reduce_sum_f32s(a23),
            simd.reduce_sum_f32s(a30), simd.reduce_sum_f32s(a31),
            simd.reduce_sum_f32s(a32), simd.reduce_sum_f32s(a33),
        ];

        // Scalar tail (d % SIMD_LANES != 0).
        let tail_n = q0t.len();
        for k in 0..tail_n {
            let qa = [q0t[k], q1t[k], q2t[k], q3t[k]];
            let xa = [x0t[k], x1t[k], x2t[k], x3t[k]];
            out[0]  += qa[0] * xa[0];
            out[1]  += qa[0] * xa[1];
            out[2]  += qa[0] * xa[2];
            out[3]  += qa[0] * xa[3];
            out[4]  += qa[1] * xa[0];
            out[5]  += qa[1] * xa[1];
            out[6]  += qa[1] * xa[2];
            out[7]  += qa[1] * xa[3];
            out[8]  += qa[2] * xa[0];
            out[9]  += qa[2] * xa[1];
            out[10] += qa[2] * xa[2];
            out[11] += qa[2] * xa[3];
            out[12] += qa[3] * xa[0];
            out[13] += qa[3] * xa[1];
            out[14] += qa[3] * xa[2];
            out[15] += qa[3] * xa[3];
        }

        out
    }
}
