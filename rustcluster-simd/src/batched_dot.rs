//! Tile-shaped batched dot product: `out = q (q_h, d) @ x (x_w, d)^T`.
//!
//! Replaces `faer::matmul` for the small-tile inner work in
//! `rustcluster::index::kernel::ip_batch`. The caller is expected to
//! tile the larger problem and wrap parallelism (rayon) on the outside;
//! this kernel itself is single-threaded.
//!
//! Two-level blocking:
//!  1. **Outer X-cache block** (`CACHE_X_BLOCK` rows of x, ~1.5 MB at
//!     d=1536). Sized to fit per-core L2 alongside the Q chunk. Each
//!     x-cache-block is loaded from DRAM once and reused across all Q
//!     rows, keeping memory traffic O(q_h·d + x_w·d) instead of
//!     O(q_h·x_w·d) which is what a naive (i, j) loop would force at
//!     wide x_w.
//!  2. **Inner 4×4 register block**. For each 4-row × 4-col block of
//!     output, 16 SIMD accumulators are kept in registers across the
//!     d-inner-loop. Each k-step loads 4 q-vectors and 4 x-vectors and
//!     issues 16 FMAs — half the load-to-FMA ratio of a row-by-row dot.
//!
//! Edge rows / cols (q_h or x_w not multiple of 4) fall back to
//! `dot_f32` per output.

use crate::dot::dot_f32;
use crate::microkernel::Tile4x4;

const TILE_M: usize = 4;
const TILE_N: usize = 4;

/// Rows of x kept resident per outer cache block. Tuned empirically on
/// Apple Silicon M-series at d=1536, n=20k, nq=1000: 64 rows × 1536 × 4
/// = 384 KB beats both smaller (insufficient amortization) and larger
/// blocks (L2 contention with other threads / TLB pressure). See
/// `examples/probe_search.rs` and the M5 sweep in commit history.
/// Override with `RUSTCLUSTER_SIMD_X_BLOCK`.
const CACHE_X_BLOCK_DEFAULT: usize = 64;

fn cache_x_block() -> usize {
    if let Ok(v) = std::env::var("RUSTCLUSTER_SIMD_X_BLOCK") {
        if let Ok(n) = v.parse::<usize>() {
            if n >= 4 {
                return n - (n % 4);
            }
        }
    }
    CACHE_X_BLOCK_DEFAULT
}

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
    let cache_x = cache_x_block();

    // Outer x-cache-block loop: a CACHE_X_BLOCK-row chunk of x is
    // loaded from DRAM once and reused across all q rows in the inner
    // loops below. Without this, naive (i, j) iteration causes x to be
    // re-read q_h/4 times — disastrous when x_w is large (search shape).
    for j_outer in (0..x_w_main).step_by(cache_x) {
        let j_outer_end = (j_outer + cache_x).min(x_w_main);

        for i in (0..q_h_main).step_by(TILE_M) {
            let q0 = &q[i * d..(i + 1) * d];
            let q1 = &q[(i + 1) * d..(i + 2) * d];
            let q2 = &q[(i + 2) * d..(i + 3) * d];
            let q3 = &q[(i + 3) * d..(i + 4) * d];
            for j in (j_outer..j_outer_end).step_by(TILE_N) {
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
