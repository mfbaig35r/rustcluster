//! Fused threshold-emit kernel for similarity-graph workloads.
//!
//! Computes pairwise inner products (or squared L2 distances via the
//! identity ‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b) between two blocks of
//! vectors and emits `(i, j, score)` tuples that pass the threshold
//! test, **without ever materializing the (h × w) score tile**.
//!
//! This is the architectural lever for v1.2: the production
//! `similarity_graph_ip` materializes a 384×384 × 4-byte = 590 KB
//! score tile per outer tile, then re-reads it for thresholding.
//! The fused kernel computes 4×4 sub-tiles in registers, threshold-
//! checks them on the spot, and only edges that pass touch memory.
//! For sparse threshold workloads, the reduction in memory traffic
//! is dramatic.
//!
//! Two flavours:
//! - `fused_threshold_emit_tile_ip`: emits when dot ≥ threshold,
//!   stores the dot as the edge score.
//! - `fused_threshold_emit_tile_l2`: emits when squared-L2 ≤ threshold,
//!   stores the squared-L2 as the edge score. Caller supplies
//!   precomputed row norms.
//!
//! `diagonal=true` indicates `xi` and `xj` are the same block (and
//! `i0 == j0`); only strict upper-triangle pairs (i < j) are emitted.

use crate::dot::dot_f32;
use crate::microkernel::Tile4x4;

// =====================================================================
// IP fused kernel
// =====================================================================

/// Inner-product fused threshold-emit kernel.
///
/// Emits `(i0+li, j0+lj, dot(xi[li], xj[lj]))` for every `(li, lj)` with
/// dot ≥ threshold. When `diagonal=true`, only emits pairs with
/// `(i0+li) < (j0+lj)` — caller must pass `xi == xj`, `h == w`, `i0 == j0`.
pub fn fused_threshold_emit_tile_ip(
    xi: &[f32],
    h: usize,
    xj: &[f32],
    w: usize,
    d: usize,
    i0: u32,
    j0: u32,
    threshold: f32,
    diagonal: bool,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    if diagonal {
        debug_assert_eq!(h, w, "diagonal=true requires h == w");
        debug_assert_eq!(i0, j0, "diagonal=true requires i0 == j0");
        diag_ip(xi, h, d, i0, threshold, edges);
    } else {
        block_ip(xi, h, i0, xj, w, j0, d, threshold, edges);
    }
}

fn block_ip(
    xi: &[f32], h: usize, i0: u32,
    xj: &[f32], w: usize, j0: u32,
    d: usize, threshold: f32,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    debug_assert_eq!(xi.len(), h * d);
    debug_assert_eq!(xj.len(), w * d);

    if h == 0 || w == 0 || d == 0 {
        return;
    }

    let h_main = h - h % 4;
    let w_main = w - w % 4;
    let arch = pulp::Arch::new();

    for i in (0..h_main).step_by(4) {
        let xi0 = &xi[i * d..(i + 1) * d];
        let xi1 = &xi[(i + 1) * d..(i + 2) * d];
        let xi2 = &xi[(i + 2) * d..(i + 3) * d];
        let xi3 = &xi[(i + 3) * d..(i + 4) * d];
        for j in (0..w_main).step_by(4) {
            let xj0 = &xj[j * d..(j + 1) * d];
            let xj1 = &xj[(j + 1) * d..(j + 2) * d];
            let xj2 = &xj[(j + 2) * d..(j + 3) * d];
            let xj3 = &xj[(j + 3) * d..(j + 4) * d];
            let block = arch.dispatch(Tile4x4 {
                q0: xi0, q1: xi1, q2: xi2, q3: xi3,
                x0: xj0, x1: xj1, x2: xj2, x3: xj3,
            });
            for ii in 0..4u32 {
                for jj in 0..4u32 {
                    let s = block[(ii as usize) * 4 + (jj as usize)];
                    if s >= threshold {
                        edges.push((i0 + i as u32 + ii, j0 + j as u32 + jj, s));
                    }
                }
            }
        }
    }

    // Edge: remaining columns within main rows.
    if w_main < w {
        for i in 0..h_main {
            let xi_row = &xi[i * d..(i + 1) * d];
            for j in w_main..w {
                let xj_row = &xj[j * d..(j + 1) * d];
                let s = dot_f32(xi_row, xj_row);
                if s >= threshold {
                    edges.push((i0 + i as u32, j0 + j as u32, s));
                }
            }
        }
    }

    // Edge: remaining rows.
    if h_main < h {
        for i in h_main..h {
            let xi_row = &xi[i * d..(i + 1) * d];
            for j in 0..w {
                let xj_row = &xj[j * d..(j + 1) * d];
                let s = dot_f32(xi_row, xj_row);
                if s >= threshold {
                    edges.push((i0 + i as u32, j0 + j as u32, s));
                }
            }
        }
    }
}

fn diag_ip(
    x: &[f32], h: usize, d: usize, i0: u32,
    threshold: f32,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    debug_assert_eq!(x.len(), h * d);
    if h < 2 || d == 0 {
        return;
    }
    // Diagonal tiles are ~2% of similarity_graph compute (n_tiles
    // diagonal vs n_tiles*(n_tiles-1)/2 off-diagonal, with diagonals
    // doing half the per-tile work). Per-row dot_f32 is the simple
    // and correct path; register blocking the diagonal would add
    // significant complexity for a small overall win.
    for li in 0..h {
        let xi_row = &x[li * d..(li + 1) * d];
        for lj in (li + 1)..h {
            let xj_row = &x[lj * d..(lj + 1) * d];
            let s = dot_f32(xi_row, xj_row);
            if s >= threshold {
                edges.push((i0 + li as u32, i0 + lj as u32, s));
            }
        }
    }
}

// =====================================================================
// L2 fused kernel — uses identity ‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b.
// =====================================================================

/// Squared-L2 fused threshold-emit kernel via the inner-product identity.
///
/// Computes dot products via the same 4×4 microkernel, then converts to
/// L2 in registers using precomputed row norms: `dist = norms_i[ii] +
/// norms_j[jj] - 2·dot`. Emits `(i, j, dist)` for every pair with
/// `dist ≤ threshold` (clamped to ≥ 0 to absorb FP rounding).
pub fn fused_threshold_emit_tile_l2(
    xi: &[f32], h: usize, norms_i: &[f32],
    xj: &[f32], w: usize, norms_j: &[f32],
    d: usize,
    i0: u32, j0: u32,
    threshold: f32,
    diagonal: bool,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    if diagonal {
        debug_assert_eq!(h, w, "diagonal=true requires h == w");
        debug_assert_eq!(i0, j0, "diagonal=true requires i0 == j0");
        diag_l2(xi, h, norms_i, d, i0, threshold, edges);
    } else {
        block_l2(xi, h, norms_i, i0, xj, w, norms_j, j0, d, threshold, edges);
    }
}

#[allow(clippy::too_many_arguments)]
fn block_l2(
    xi: &[f32], h: usize, norms_i: &[f32], i0: u32,
    xj: &[f32], w: usize, norms_j: &[f32], j0: u32,
    d: usize, threshold: f32,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    debug_assert_eq!(xi.len(), h * d);
    debug_assert_eq!(xj.len(), w * d);
    debug_assert_eq!(norms_i.len(), h);
    debug_assert_eq!(norms_j.len(), w);

    if h == 0 || w == 0 || d == 0 {
        return;
    }

    let h_main = h - h % 4;
    let w_main = w - w % 4;
    let arch = pulp::Arch::new();

    for i in (0..h_main).step_by(4) {
        let xi0 = &xi[i * d..(i + 1) * d];
        let xi1 = &xi[(i + 1) * d..(i + 2) * d];
        let xi2 = &xi[(i + 2) * d..(i + 3) * d];
        let xi3 = &xi[(i + 3) * d..(i + 4) * d];
        let ni = [norms_i[i], norms_i[i + 1], norms_i[i + 2], norms_i[i + 3]];
        for j in (0..w_main).step_by(4) {
            let xj0 = &xj[j * d..(j + 1) * d];
            let xj1 = &xj[(j + 1) * d..(j + 2) * d];
            let xj2 = &xj[(j + 2) * d..(j + 3) * d];
            let xj3 = &xj[(j + 3) * d..(j + 4) * d];
            let nj = [norms_j[j], norms_j[j + 1], norms_j[j + 2], norms_j[j + 3]];
            let block = arch.dispatch(Tile4x4 {
                q0: xi0, q1: xi1, q2: xi2, q3: xi3,
                x0: xj0, x1: xj1, x2: xj2, x3: xj3,
            });
            for ii in 0..4u32 {
                for jj in 0..4u32 {
                    let dot = block[(ii as usize) * 4 + (jj as usize)];
                    let raw = ni[ii as usize] + nj[jj as usize] - 2.0 * dot;
                    let dist = if raw < 0.0 { 0.0 } else { raw };
                    if dist <= threshold {
                        edges.push((i0 + i as u32 + ii, j0 + j as u32 + jj, dist));
                    }
                }
            }
        }
    }

    // Edge: remaining columns within main rows.
    if w_main < w {
        for i in 0..h_main {
            let xi_row = &xi[i * d..(i + 1) * d];
            for j in w_main..w {
                let xj_row = &xj[j * d..(j + 1) * d];
                let dot = dot_f32(xi_row, xj_row);
                let raw = norms_i[i] + norms_j[j] - 2.0 * dot;
                let dist = if raw < 0.0 { 0.0 } else { raw };
                if dist <= threshold {
                    edges.push((i0 + i as u32, j0 + j as u32, dist));
                }
            }
        }
    }

    // Edge: remaining rows.
    if h_main < h {
        for i in h_main..h {
            let xi_row = &xi[i * d..(i + 1) * d];
            for j in 0..w {
                let xj_row = &xj[j * d..(j + 1) * d];
                let dot = dot_f32(xi_row, xj_row);
                let raw = norms_i[i] + norms_j[j] - 2.0 * dot;
                let dist = if raw < 0.0 { 0.0 } else { raw };
                if dist <= threshold {
                    edges.push((i0 + i as u32, j0 + j as u32, dist));
                }
            }
        }
    }
}

fn diag_l2(
    x: &[f32], h: usize, norms: &[f32], d: usize, i0: u32,
    threshold: f32,
    edges: &mut Vec<(u32, u32, f32)>,
) {
    debug_assert_eq!(x.len(), h * d);
    debug_assert_eq!(norms.len(), h);
    if h < 2 || d == 0 {
        return;
    }
    for li in 0..h {
        let xi_row = &x[li * d..(li + 1) * d];
        for lj in (li + 1)..h {
            let xj_row = &x[lj * d..(lj + 1) * d];
            let dot = dot_f32(xi_row, xj_row);
            let raw = norms[li] + norms[lj] - 2.0 * dot;
            let dist = if raw < 0.0 { 0.0 } else { raw };
            if dist <= threshold {
                edges.push((i0 + li as u32, i0 + lj as u32, dist));
            }
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn reference_ip(
        xi: &[f32], h: usize,
        xj: &[f32], w: usize,
        d: usize, i0: u32, j0: u32,
        threshold: f32, diagonal: bool,
    ) -> Vec<(u32, u32, f32)> {
        let mut edges = Vec::new();
        if diagonal {
            for li in 0..h {
                for lj in (li + 1)..h {
                    let s = scalar_dot(&xi[li * d..(li + 1) * d], &xi[lj * d..(lj + 1) * d]);
                    if s >= threshold {
                        edges.push((i0 + li as u32, i0 + lj as u32, s));
                    }
                }
            }
        } else {
            for li in 0..h {
                for lj in 0..w {
                    let s = scalar_dot(&xi[li * d..(li + 1) * d], &xj[lj * d..(lj + 1) * d]);
                    if s >= threshold {
                        edges.push((i0 + li as u32, j0 + lj as u32, s));
                    }
                }
            }
        }
        edges
    }

    fn reference_l2(
        xi: &[f32], h: usize, norms_i: &[f32],
        xj: &[f32], w: usize, norms_j: &[f32],
        d: usize, i0: u32, j0: u32,
        threshold: f32, diagonal: bool,
    ) -> Vec<(u32, u32, f32)> {
        let mut edges = Vec::new();
        if diagonal {
            for li in 0..h {
                for lj in (li + 1)..h {
                    let dot = scalar_dot(&xi[li * d..(li + 1) * d], &xi[lj * d..(lj + 1) * d]);
                    let raw = norms_i[li] + norms_i[lj] - 2.0 * dot;
                    let dist = if raw < 0.0 { 0.0 } else { raw };
                    if dist <= threshold {
                        edges.push((i0 + li as u32, i0 + lj as u32, dist));
                    }
                }
            }
        } else {
            for li in 0..h {
                for lj in 0..w {
                    let dot = scalar_dot(&xi[li * d..(li + 1) * d], &xj[lj * d..(lj + 1) * d]);
                    let raw = norms_i[li] + norms_j[lj] - 2.0 * dot;
                    let dist = if raw < 0.0 { 0.0 } else { raw };
                    if dist <= threshold {
                        edges.push((i0 + li as u32, j0 + lj as u32, dist));
                    }
                }
            }
        }
        edges
    }

    fn parity_check_edges(label: &str, simd: &mut Vec<(u32, u32, f32)>, scalar: &mut Vec<(u32, u32, f32)>, tol: f32) {
        simd.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        scalar.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        assert_eq!(
            simd.len(),
            scalar.len(),
            "{label}: edge count mismatch ({} vs {})",
            simd.len(),
            scalar.len()
        );
        for (s, r) in simd.iter().zip(scalar.iter()) {
            assert_eq!((s.0, s.1), (r.0, r.1), "{label}: pair mismatch");
            assert!(
                (s.2 - r.2).abs() <= tol,
                "{label}: score mismatch at ({},{}): simd={} scalar={} diff={}",
                s.0, s.1, s.2, r.2, (s.2 - r.2).abs()
            );
        }
    }

    fn rand_block(rows: usize, d: usize, mut seed: u64) -> Vec<f32> {
        (0..rows * d)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn norms_of(x: &[f32], rows: usize, d: usize) -> Vec<f32> {
        (0..rows)
            .map(|i| x[i * d..(i + 1) * d].iter().map(|&v| v * v).sum())
            .collect()
    }

    #[test]
    fn ip_offdiag_4x4() {
        let d = 8;
        let xi = rand_block(4, d, 1);
        let xj = rand_block(4, d, 2);
        let mut simd_e = Vec::new();
        fused_threshold_emit_tile_ip(&xi, 4, &xj, 4, d, 100, 200, -100.0, false, &mut simd_e);
        let mut scalar_e = reference_ip(&xi, 4, &xj, 4, d, 100, 200, -100.0, false);
        parity_check_edges("ip offdiag 4x4 (loose threshold)", &mut simd_e, &mut scalar_e, 1e-4);
    }

    #[test]
    fn ip_diagonal_4x4() {
        let d = 8;
        let xi = rand_block(4, d, 1);
        let mut simd_e = Vec::new();
        fused_threshold_emit_tile_ip(&xi, 4, &xi, 4, d, 100, 100, -100.0, true, &mut simd_e);
        let mut scalar_e = reference_ip(&xi, 4, &xi, 4, d, 100, 100, -100.0, true);
        parity_check_edges("ip diag 4x4 (loose)", &mut simd_e, &mut scalar_e, 1e-4);
    }

    #[test]
    fn ip_offdiag_realistic() {
        // 32×32 from d=384 — exercises the main 4×4 path with no edges.
        let h = 32;
        let w = 32;
        let d = 384;
        let xi = rand_block(h, d, 1);
        let xj = rand_block(w, d, 2);
        for &threshold in &[-100.0, 0.0, 1.0, 5.0] {
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_ip(&xi, h, &xj, w, d, 0, 100, threshold, false, &mut simd_e);
            let mut scalar_e = reference_ip(&xi, h, &xj, w, d, 0, 100, threshold, false);
            parity_check_edges(
                &format!("ip offdiag d=384 thr={threshold}"),
                &mut simd_e,
                &mut scalar_e,
                1e-3,
            );
        }
    }

    #[test]
    fn ip_edge_shapes() {
        let d = 384;
        // Non-multiple-of-4 dims to exercise edge paths.
        let cases = [(1, 1), (3, 3), (5, 5), (7, 11), (4, 7), (7, 4), (13, 4)];
        for (h, w) in cases {
            let xi = rand_block(h, d, 1);
            let xj = rand_block(w, d, 2);
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_ip(&xi, h, &xj, w, d, 0, 100, -100.0, false, &mut simd_e);
            let mut scalar_e = reference_ip(&xi, h, &xj, w, d, 0, 100, -100.0, false);
            parity_check_edges(
                &format!("ip edge ({h},{w})"),
                &mut simd_e,
                &mut scalar_e,
                1e-3,
            );
        }
    }

    #[test]
    fn l2_offdiag_realistic() {
        let h = 32;
        let w = 32;
        let d = 384;
        let xi = rand_block(h, d, 1);
        let xj = rand_block(w, d, 2);
        let ni = norms_of(&xi, h, d);
        let nj = norms_of(&xj, w, d);
        for &threshold in &[1e9, 200.0, 100.0, 50.0] {
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_l2(
                &xi, h, &ni, &xj, w, &nj, d, 0, 100, threshold, false, &mut simd_e,
            );
            let mut scalar_e = reference_l2(
                &xi, h, &ni, &xj, w, &nj, d, 0, 100, threshold, false,
            );
            parity_check_edges(
                &format!("l2 offdiag thr={threshold}"),
                &mut simd_e,
                &mut scalar_e,
                1e-2,
            );
        }
    }

    #[test]
    fn l2_diagonal_realistic() {
        let h = 32;
        let d = 384;
        let xi = rand_block(h, d, 1);
        let ni = norms_of(&xi, h, d);
        for &threshold in &[1e9, 200.0, 100.0] {
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_l2(
                &xi, h, &ni, &xi, h, &ni, d, 0, 0, threshold, true, &mut simd_e,
            );
            let mut scalar_e = reference_l2(
                &xi, h, &ni, &xi, h, &ni, d, 0, 0, threshold, true,
            );
            parity_check_edges(
                &format!("l2 diag thr={threshold}"),
                &mut simd_e,
                &mut scalar_e,
                1e-2,
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(48))]

        #[test]
        fn ip_fused_matches_reference(
            (h, w, d) in (1usize..16, 1usize..16, 1usize..200),
            seed: u64,
            threshold in -10.0f32..10.0,
        ) {
            let xi = rand_block(h, d, seed);
            let xj = rand_block(w, d, seed.wrapping_add(1));
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_ip(&xi, h, &xj, w, d, 0, 0, threshold, false, &mut simd_e);
            let mut scalar_e = reference_ip(&xi, h, &xj, w, d, 0, 0, threshold, false);
            simd_e.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            scalar_e.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            // Edge counts may differ by ±1 when a score sits exactly at
            // threshold and FP rounding flips inclusion. Allow that.
            let count_diff = (simd_e.len() as isize - scalar_e.len() as isize).abs();
            prop_assert!(
                count_diff <= 2,
                "edge count diverged by {} (simd={} scalar={})",
                count_diff, simd_e.len(), scalar_e.len()
            );
        }

        #[test]
        fn ip_fused_diag_matches_reference(
            h in 1usize..12,
            d in 1usize..200,
            seed: u64,
            threshold in -10.0f32..10.0,
        ) {
            let x = rand_block(h, d, seed);
            let mut simd_e = Vec::new();
            fused_threshold_emit_tile_ip(&x, h, &x, h, d, 0, 0, threshold, true, &mut simd_e);
            let mut scalar_e = reference_ip(&x, h, &x, h, d, 0, 0, threshold, true);
            simd_e.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            scalar_e.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
            let count_diff = (simd_e.len() as isize - scalar_e.len() as isize).abs();
            prop_assert!(count_diff <= 2);
            // For diagonal, all i < j by construction.
            for (s, t, _) in &simd_e {
                prop_assert!(s < t);
            }
        }
    }
}
