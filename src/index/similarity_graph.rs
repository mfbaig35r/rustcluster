//! Fused all-pairs similarity-graph kernel.
//!
//! For an indexed dataset `X (n, d)` and a threshold `t`, emit every pair
//! `(i, j)` with `i != j` whose similarity passes the threshold:
//! - L2: emit when `||x_i - x_j||² <= t`
//! - IP: emit when `x_i · x_j >= t`
//!
//! Performance comes from three things vs `range_search` in a loop:
//!  1. Cache-blocked tiles via faer GEMM — no full `n²` materialization.
//!  2. Upper-triangle iteration — half the compute when query == database.
//!  3. Per-tile output buffers, merged once at the end — no contention.
//!
//! Memory: with a loose threshold the edge list is O(n²); the caller picks
//! the threshold and is responsible for managing memory. A streaming
//! variant lands in v2.
//!
//! Tile size is dim-aware (see `pick_tile_size`). Override with the
//! `RUSTCLUSTER_TILE_SIZE` env var for benchmarking.

use faer::Par;
use ndarray::{s, ArrayView2};
use rayon::prelude::*;

use super::ids::IdMap;
use super::kernel::ip_batch;

/// Edge list result of a similarity-graph computation. All three vectors
/// have equal length.
#[derive(Debug, Default)]
pub struct EdgeList {
    pub src: Vec<u64>,
    pub dst: Vec<u64>,
    pub scores: Vec<f32>,
}

impl EdgeList {
    fn with_capacity(cap: usize) -> Self {
        Self {
            src: Vec::with_capacity(cap),
            dst: Vec::with_capacity(cap),
            scores: Vec::with_capacity(cap),
        }
    }

    fn append(&mut self, mut other: EdgeList) {
        self.src.append(&mut other.src);
        self.dst.append(&mut other.dst);
        self.scores.append(&mut other.scores);
    }

    pub fn len(&self) -> usize {
        self.src.len()
    }

    pub fn is_empty(&self) -> bool {
        self.src.is_empty()
    }
}

/// Pick the cache-blocked tile size based on `dim`. Reads
/// `RUSTCLUSTER_TILE_SIZE` env var as an override.
///
/// Numbers come from a sweep at each dim regime — the goal is enough work
/// per tile that faer's GEMM amortizes its setup cost, while keeping the
/// inner score buffer + input strips in cache. At very high `d` (≥1025)
/// small tiles waste time on per-tile overhead; sweep showed `tile=384`
/// outperforms `64` by ~17% at `d=1536`.
pub fn pick_tile_size(dim: usize) -> usize {
    if let Ok(v) = std::env::var("RUSTCLUSTER_TILE_SIZE") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    match dim {
        0..=64 => 256,
        65..=256 => 128,
        257..=1024 => 96,
        _ => 384,
    }
}

/// L2 similarity graph: emit pairs with squared distance <= threshold.
pub fn similarity_graph_l2(
    data: ArrayView2<f32>,
    norms_sq: &[f32],
    ids: &IdMap,
    threshold: f32,
    unique_pairs: bool,
) -> EdgeList {
    let (n, d) = data.dim();
    if n < 2 {
        return EdgeList::default();
    }
    let tile = pick_tile_size(d);
    let pairs = upper_triangle_tile_pairs(n, tile);

    let chunks: Vec<EdgeList> = pairs
        .par_iter()
        .map(|&(i0, j0)| {
            let i1 = (i0 + tile).min(n);
            let j1 = (j0 + tile).min(n);
            let xi = data.slice(s![i0..i1, ..]);
            let xj = data.slice(s![j0..j1, ..]);
            let ip = ip_batch(xi, xj, Par::Seq);
            let h = i1 - i0;
            let w = j1 - j0;
            // Estimate at ~1 edge per 32 candidates — heuristic for capacity.
            let mut local = EdgeList::with_capacity(h * w / 32 + 8);

            for li in 0..h {
                let gi = i0 + li;
                let qn = norms_sq[gi];
                let lj_start = if i0 == j0 { li + 1 } else { 0 };
                for lj in lj_start..w {
                    let gj = j0 + lj;
                    let raw = qn + norms_sq[gj] - 2.0 * ip[(li, lj)];
                    let dist = if raw < 0.0 { 0.0 } else { raw };
                    if dist <= threshold {
                        emit(&mut local, ids, gi, gj, dist, unique_pairs);
                    }
                }
            }
            local
        })
        .collect();

    concat_edges(chunks)
}

/// Inner-product similarity graph: emit pairs with score >= threshold.
///
/// When `RUSTCLUSTER_FUSED=1` (or `true`) is set in the environment,
/// dispatches to `similarity_graph_ip_fused_spike` — the v1.2 Step 0b
/// architectural test that processes sub-tile bands without
/// materializing the full score matrix.
pub fn similarity_graph_ip(
    data: ArrayView2<f32>,
    ids: &IdMap,
    threshold: f32,
    unique_pairs: bool,
) -> EdgeList {
    if fused_spike_enabled() {
        return similarity_graph_ip_fused_spike(data, ids, threshold, unique_pairs);
    }

    let (n, d) = data.dim();
    if n < 2 {
        return EdgeList::default();
    }
    let tile = pick_tile_size(d);
    let pairs = upper_triangle_tile_pairs(n, tile);

    let chunks: Vec<EdgeList> = pairs
        .par_iter()
        .map(|&(i0, j0)| {
            let i1 = (i0 + tile).min(n);
            let j1 = (j0 + tile).min(n);
            let xi = data.slice(s![i0..i1, ..]);
            let xj = data.slice(s![j0..j1, ..]);
            let ip = ip_batch(xi, xj, Par::Seq);
            let h = i1 - i0;
            let w = j1 - j0;
            let mut local = EdgeList::with_capacity(h * w / 32 + 8);

            for li in 0..h {
                let gi = i0 + li;
                let lj_start = if i0 == j0 { li + 1 } else { 0 };
                for lj in lj_start..w {
                    let gj = j0 + lj;
                    let score = ip[(li, lj)];
                    if score >= threshold {
                        emit(&mut local, ids, gi, gj, score, unique_pairs);
                    }
                }
            }
            local
        })
        .collect();

    concat_edges(chunks)
}

fn fused_spike_enabled() -> bool {
    matches!(
        std::env::var("RUSTCLUSTER_FUSED").as_deref(),
        Ok("1") | Ok("true")
    )
}

/// Sub-tile row band size for the Step 0b spike. Override with the
/// `RUSTCLUSTER_FUSED_SUB` env var. Default 16 — small enough that the
/// per-band score buffer fits L1 even at large tile widths.
fn spike_sub_tile() -> usize {
    if let Ok(v) = std::env::var("RUSTCLUSTER_FUSED_SUB") {
        if let Ok(n) = v.parse::<usize>() {
            if n > 0 {
                return n;
            }
        }
    }
    16
}

/// v1.2 Step 0b — fused-architecture spike for `similarity_graph_ip`.
///
/// Same outer tile structure as the production path, but inside each
/// tile we process a band of `spike_sub_tile()` rows at a time. The
/// per-band score buffer is `(sub_h × w) f32` — small enough to live
/// in L1 — and is dropped before the next band runs. The full
/// `(h × w)` score matrix is never materialized.
///
/// The inner microkernel is still faer matmul. This is deliberately
/// the architectural-only test: if even faer-backed sub-tiling closes
/// most of the FAISS gap at d=1536, the v1.2 microkernel work in
/// Plan A is bonus, not load-bearing.
pub fn similarity_graph_ip_fused_spike(
    data: ArrayView2<f32>,
    ids: &IdMap,
    threshold: f32,
    unique_pairs: bool,
) -> EdgeList {
    similarity_graph_ip_fused_spike_with(data, ids, threshold, unique_pairs, spike_sub_tile())
}

/// As `similarity_graph_ip_fused_spike`, with an explicit sub-tile size.
/// Used by the Step 0b probe to sweep over sub-tile sizes without
/// fiddling with env vars across runs.
pub fn similarity_graph_ip_fused_spike_with(
    data: ArrayView2<f32>,
    ids: &IdMap,
    threshold: f32,
    unique_pairs: bool,
    sub: usize,
) -> EdgeList {
    let (n, d) = data.dim();
    if n < 2 {
        return EdgeList::default();
    }
    let tile = pick_tile_size(d);
    let pairs = upper_triangle_tile_pairs(n, tile);

    let chunks: Vec<EdgeList> = pairs
        .par_iter()
        .map(|&(i0, j0)| {
            let i1 = (i0 + tile).min(n);
            let j1 = (j0 + tile).min(n);
            let h = i1 - i0;
            let w = j1 - j0;
            let xj = data.slice(s![j0..j1, ..]);
            let mut local = EdgeList::with_capacity(h * w / 32 + 8);

            let mut sub_start = 0;
            while sub_start < h {
                let sub_end = (sub_start + sub).min(h);
                let sub_h = sub_end - sub_start;
                let xi_sub = data.slice(s![i0 + sub_start..i0 + sub_end, ..]);
                let ip = ip_batch(xi_sub, xj, Par::Seq);

                for sli in 0..sub_h {
                    let li = sub_start + sli;
                    let gi = i0 + li;
                    let lj_start = if i0 == j0 { li + 1 } else { 0 };
                    for lj in lj_start..w {
                        let gj = j0 + lj;
                        let score = ip[(sli, lj)];
                        if score >= threshold {
                            emit(&mut local, ids, gi, gj, score, unique_pairs);
                        }
                    }
                }
                sub_start += sub;
            }
            local
        })
        .collect();

    concat_edges(chunks)
}

#[inline]
fn emit(out: &mut EdgeList, ids: &IdMap, gi: usize, gj: usize, score: f32, unique_pairs: bool) {
    let id_i = ids.external(gi);
    let id_j = ids.external(gj);
    out.src.push(id_i);
    out.dst.push(id_j);
    out.scores.push(score);
    if !unique_pairs {
        out.src.push(id_j);
        out.dst.push(id_i);
        out.scores.push(score);
    }
}

fn upper_triangle_tile_pairs(n: usize, tile: usize) -> Vec<(usize, usize)> {
    let n_tiles = n.div_ceil(tile);
    let mut pairs = Vec::with_capacity(n_tiles * (n_tiles + 1) / 2);
    for it in 0..n_tiles {
        let i0 = it * tile;
        for jt in it..n_tiles {
            let j0 = jt * tile;
            pairs.push((i0, j0));
        }
    }
    pairs
}

fn concat_edges(chunks: Vec<EdgeList>) -> EdgeList {
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut out = EdgeList::with_capacity(total);
    for c in chunks {
        out.append(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn make_ids_sequential(n: usize) -> IdMap {
        let mut m = IdMap::new();
        m.extend_sequential(n).unwrap();
        m
    }

    #[test]
    fn ip_emits_both_directions_by_default() {
        let data = arr2(&[[1.0f32, 0.0], [0.7071, 0.7071], [0.0, 1.0]]);
        let ids = make_ids_sequential(3);
        let g = similarity_graph_ip(data.view(), &ids, 0.5, false);
        // Pairs above 0.5: (0,1)=~0.707, (1,2)=~0.707. (0,2)=0. So 2 unique
        // pairs, 4 total edges with both directions.
        assert_eq!(g.len(), 4);
        let pair_set: std::collections::HashSet<(u64, u64)> = g
            .src
            .iter()
            .zip(g.dst.iter())
            .map(|(&a, &b)| (a, b))
            .collect();
        assert!(pair_set.contains(&(0, 1)));
        assert!(pair_set.contains(&(1, 0)));
        assert!(pair_set.contains(&(1, 2)));
        assert!(pair_set.contains(&(2, 1)));
    }

    #[test]
    fn ip_unique_pairs_emits_one_direction() {
        let data = arr2(&[[1.0f32, 0.0], [0.7071, 0.7071], [0.0, 1.0]]);
        let ids = make_ids_sequential(3);
        let g = similarity_graph_ip(data.view(), &ids, 0.5, true);
        assert_eq!(g.len(), 2);
        // i < j on every emitted edge.
        for (s, d) in g.src.iter().zip(g.dst.iter()) {
            assert!(s < d);
        }
    }

    #[test]
    fn l2_threshold_inclusive() {
        let data = arr2(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]);
        let norms_sq: Vec<f32> = data.outer_iter().map(|r| r.dot(&r)).collect();
        let ids = make_ids_sequential(4);
        // Pairs with sq-dist <= 2.0: (0,1)=1, (0,2)=1, (1,2)=2.
        let g = similarity_graph_l2(data.view(), &norms_sq, &ids, 2.0, true);
        assert_eq!(g.len(), 3);
        for &s in &g.scores {
            assert!(s <= 2.0 + 1e-5);
        }
    }

    #[test]
    fn no_self_loops() {
        let data = arr2(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let ids = make_ids_sequential(3);
        let g = similarity_graph_ip(data.view(), &ids, -1e9, false);
        for (s, d) in g.src.iter().zip(g.dst.iter()) {
            assert_ne!(s, d, "self-loop emitted");
        }
    }

    #[test]
    fn external_ids_used() {
        let data = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
        let mut ids = IdMap::new();
        ids.extend_explicit(&[100, 200]).unwrap();
        let g = similarity_graph_ip(data.view(), &ids, -1.0, true);
        assert_eq!(g.src, vec![100]);
        assert_eq!(g.dst, vec![200]);
    }

    #[test]
    fn empty_below_two() {
        let data = arr2::<f32, _>(&[[1.0, 2.0]]);
        let ids = make_ids_sequential(1);
        let g = similarity_graph_ip(data.view(), &ids, 0.0, false);
        assert!(g.is_empty());
    }

    #[test]
    fn tile_size_dim_aware() {
        assert_eq!(pick_tile_size(16), 256);
        assert_eq!(pick_tile_size(128), 128);
        assert_eq!(pick_tile_size(512), 96);
        assert_eq!(pick_tile_size(1536), 384);
    }

    #[test]
    fn parity_with_brute_force_ip() {
        // Random-ish 50×16 matrix, threshold 0.0 — compare counts and that
        // the kernel's edges match a brute-force reference.
        let n = 50usize;
        let d = 16usize;
        let mut data = ndarray::Array2::<f32>::zeros((n, d));
        let mut seed: u32 = 1;
        for v in data.iter_mut() {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (seed >> 8) as f32 / (u32::MAX >> 8) as f32 - 0.5;
        }
        let norms: Vec<f32> = data.outer_iter().map(|r| r.dot(&r)).collect();
        let _ = norms;

        let ids = make_ids_sequential(n);
        let g = similarity_graph_ip(data.view(), &ids, 0.0, true);

        let mut bf: Vec<(u64, u64, f32)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut s = 0.0f32;
                for k in 0..d {
                    s += data[(i, k)] * data[(j, k)];
                }
                if s >= 0.0 {
                    bf.push((i as u64, j as u64, s));
                }
            }
        }
        assert_eq!(g.len(), bf.len(), "edge count mismatch");

        let mut got: Vec<(u64, u64, f32)> = g
            .src
            .iter()
            .zip(g.dst.iter())
            .zip(g.scores.iter())
            .map(|((&a, &b), &s)| (a, b, s))
            .collect();
        got.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        bf.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        for (a, b) in got.iter().zip(bf.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-4);
        }
    }
}
