//! Tile-shaped batched dot product: `out = q (q_h, d) @ x (x_w, d)^T`.
//!
//! Replaces `faer::matmul` for the small-tile inner work in
//! `rustcluster::index::kernel::ip_batch`. The caller is expected to
//! tile the larger problem and wrap parallelism (rayon) on the outside;
//! this kernel itself is single-threaded.
//!
//! Implementation: 4×4 register-blocked microkernel. For each 4-row ×
//! 4-col block of output, 16 SIMD accumulators are kept in registers
//! across the d-inner-loop. Each k-step loads 4 q-vectors and 4
//! x-vectors and issues 16 FMAs — half the load-to-FMA ratio of a
//! row-by-row dot. Edge rows / cols (q_h or x_w not multiple of 4)
//! fall back to `dot_f32` per output.

use crate::dot::dot_f32;
use pulp::{Simd, WithSimd};

const TILE_M: usize = 4;
const TILE_N: usize = 4;

/// Scalar reference: `out[i, j] = sum_k q[i, k] * x[j, k]`.
pub fn batched_dot_tile_scalar(
    q: &[f32],
    q_h: usize,
    x: &[f32],
    x_w: usize,
    d: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), q_h * d, "q size mismatch");
    debug_assert_eq!(x.len(), x_w * d, "x size mismatch");
    debug_assert_eq!(out.len(), q_h * x_w, "out size mismatch");

    for i in 0..q_h {
        let q_row = &q[i * d..(i + 1) * d];
        for j in 0..x_w {
            let x_row = &x[j * d..(j + 1) * d];
            let mut acc = 0.0f32;
            for k in 0..d {
                acc += q_row[k] * x_row[k];
            }
            out[i * x_w + j] = acc;
        }
    }
}

/// SIMD batched dot tile. Computes `out (q_h, x_w) = q (q_h, d) @ x (x_w, d)^T`.
pub fn batched_dot_tile(
    q: &[f32],
    q_h: usize,
    x: &[f32],
    x_w: usize,
    d: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(q.len(), q_h * d, "q size mismatch");
    debug_assert_eq!(x.len(), x_w * d, "x size mismatch");
    debug_assert_eq!(out.len(), q_h * x_w, "out size mismatch");

    if q_h == 0 || x_w == 0 || d == 0 {
        return;
    }

    let q_h_main = q_h - (q_h % TILE_M);
    let x_w_main = x_w - (x_w % TILE_N);

    let arch = pulp::Arch::new();

    // Main 4×4 register-blocked path.
    for i in (0..q_h_main).step_by(TILE_M) {
        let q0 = &q[i * d..(i + 1) * d];
        let q1 = &q[(i + 1) * d..(i + 2) * d];
        let q2 = &q[(i + 2) * d..(i + 3) * d];
        let q3 = &q[(i + 3) * d..(i + 4) * d];
        for j in (0..x_w_main).step_by(TILE_N) {
            let x0 = &x[j * d..(j + 1) * d];
            let x1 = &x[(j + 1) * d..(j + 2) * d];
            let x2 = &x[(j + 2) * d..(j + 3) * d];
            let x3 = &x[(j + 3) * d..(j + 4) * d];
            let block = arch.dispatch(Tile4x4 {
                q0, q1, q2, q3, x0, x1, x2, x3,
            });
            out[i * x_w + j]           = block[0];
            out[i * x_w + j + 1]       = block[1];
            out[i * x_w + j + 2]       = block[2];
            out[i * x_w + j + 3]       = block[3];
            out[(i + 1) * x_w + j]     = block[4];
            out[(i + 1) * x_w + j + 1] = block[5];
            out[(i + 1) * x_w + j + 2] = block[6];
            out[(i + 1) * x_w + j + 3] = block[7];
            out[(i + 2) * x_w + j]     = block[8];
            out[(i + 2) * x_w + j + 1] = block[9];
            out[(i + 2) * x_w + j + 2] = block[10];
            out[(i + 2) * x_w + j + 3] = block[11];
            out[(i + 3) * x_w + j]     = block[12];
            out[(i + 3) * x_w + j + 1] = block[13];
            out[(i + 3) * x_w + j + 2] = block[14];
            out[(i + 3) * x_w + j + 3] = block[15];
        }
    }

    // Edge: remaining columns within main rows.
    if x_w_main < x_w {
        for i in 0..q_h_main {
            let q_row = &q[i * d..(i + 1) * d];
            for j in x_w_main..x_w {
                let x_row = &x[j * d..(j + 1) * d];
                out[i * x_w + j] = dot_f32(q_row, x_row);
            }
        }
    }

    // Edge: remaining rows.
    if q_h_main < q_h {
        for i in q_h_main..q_h {
            let q_row = &q[i * d..(i + 1) * d];
            for j in 0..x_w {
                let x_row = &x[j * d..(j + 1) * d];
                out[i * x_w + j] = dot_f32(q_row, x_row);
            }
        }
    }
}

/// 4×4 register-blocked microkernel. Reads 4 q-rows and 4 x-rows of
/// length d, returns the 16 inner products as a flat array
/// `[a00, a01, a02, a03, a10, ..., a33]`.
struct Tile4x4<'a> {
    q0: &'a [f32], q1: &'a [f32], q2: &'a [f32], q3: &'a [f32],
    x0: &'a [f32], x1: &'a [f32], x2: &'a [f32], x3: &'a [f32],
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn close_tile(scalar: &[f32], simd: &[f32], q: &[f32], x: &[f32], d: usize) -> bool {
        if scalar.len() != simd.len() {
            return false;
        }
        let q_h = q.len() / d;
        let x_w = x.len() / d;
        for i in 0..q_h {
            let q_row = &q[i * d..(i + 1) * d];
            for j in 0..x_w {
                let x_row = &x[j * d..(j + 1) * d];
                let abs_sum: f32 = q_row.iter().zip(x_row.iter()).map(|(a, b)| (a * b).abs()).sum();
                let tol = 2.0 * (d as f32) * f32::EPSILON * abs_sum + 1e-4;
                let diff = (scalar[i * x_w + j] - simd[i * x_w + j]).abs();
                if diff > tol {
                    eprintln!(
                        "mismatch at ({i}, {j}): scalar={} simd={} tol={}",
                        scalar[i * x_w + j],
                        simd[i * x_w + j],
                        tol
                    );
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn empty_dims() {
        let mut out: Vec<f32> = Vec::new();
        batched_dot_tile(&[], 0, &[], 0, 1, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn known_value_2x2() {
        // Falls into the edge-row+edge-col path (q_h, x_w both < 4).
        let q = [1.0, 2.0, 3.0, 4.0];
        let x = [5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];
        batched_dot_tile(&q, 2, &x, 2, 2, &mut out);
        assert_eq!(out, [17.0, 23.0, 39.0, 53.0]);
    }

    #[test]
    fn known_value_4x4() {
        // 4×4 exactly hits the main microkernel path.
        let q: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let x: Vec<f32> = (17..=32).map(|i| i as f32).collect();
        let d = 4;
        let mut out = vec![0.0f32; 16];
        let mut out_scalar = vec![0.0f32; 16];
        batched_dot_tile(&q, 4, &x, 4, d, &mut out);
        batched_dot_tile_scalar(&q, 4, &x, 4, d, &mut out_scalar);
        assert_eq!(out, out_scalar);
    }

    #[test]
    fn matches_scalar_at_realistic_shape() {
        let q_h = 16;
        let x_w = 16;
        let d = 384;
        let q: Vec<f32> = (0..q_h * d).map(|i| ((i as f32) * 0.001).sin()).collect();
        let x: Vec<f32> = (0..x_w * d).map(|i| ((i as f32) * 0.0013).cos()).collect();
        let mut out_scalar = vec![0.0f32; q_h * x_w];
        let mut out_simd = vec![0.0f32; q_h * x_w];
        batched_dot_tile_scalar(&q, q_h, &x, x_w, d, &mut out_scalar);
        batched_dot_tile(&q, q_h, &x, x_w, d, &mut out_simd);
        assert!(close_tile(&out_scalar, &out_simd, &q, &x, d));
    }

    #[test]
    fn edge_shapes() {
        // Various non-multiple-of-4 dims to exercise edge paths.
        let d = 384;
        let cases = [(1, 1), (3, 3), (5, 5), (7, 11), (4, 7), (7, 4), (13, 4), (4, 13)];
        for (q_h, x_w) in cases {
            let q: Vec<f32> = (0..q_h * d).map(|i| ((i as f32) * 0.003).sin()).collect();
            let x: Vec<f32> = (0..x_w * d).map(|i| ((i as f32) * 0.005).cos()).collect();
            let mut out_scalar = vec![0.0f32; q_h * x_w];
            let mut out_simd = vec![0.0f32; q_h * x_w];
            batched_dot_tile_scalar(&q, q_h, &x, x_w, d, &mut out_scalar);
            batched_dot_tile(&q, q_h, &x, x_w, d, &mut out_simd);
            assert!(
                close_tile(&out_scalar, &out_simd, &q, &x, d),
                "edge shape ({q_h}, {x_w}) failed"
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn batched_dot_tile_matches_scalar(
            (q_h, x_w, d) in (1usize..16, 1usize..16, 1usize..200),
            seed: u64,
        ) {
            let mut state = seed;
            let mut next = || -> f32 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32) / (u32::MAX as f32) * 200.0 - 100.0
            };
            let q: Vec<f32> = (0..q_h * d).map(|_| next()).collect();
            let x: Vec<f32> = (0..x_w * d).map(|_| next()).collect();
            let mut out_scalar = vec![0.0f32; q_h * x_w];
            let mut out_simd = vec![0.0f32; q_h * x_w];
            batched_dot_tile_scalar(&q, q_h, &x, x_w, d, &mut out_scalar);
            batched_dot_tile(&q, q_h, &x, x_w, d, &mut out_simd);
            prop_assert!(
                close_tile(&out_scalar, &out_simd, &q, &x, d),
                "mismatch at q_h={q_h} x_w={x_w} d={d}"
            );
        }
    }
}
