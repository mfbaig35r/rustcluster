//! Distance and inner-product kernels for flat indexes.
//!
//! Hot path is `ip_batch` (Q @ X^T). For batched queries it dispatches
//! to `rustcluster_simd::batched_dot_tile` — a 4×4 register-blocked
//! pure-Rust SIMD kernel that matches or beats faer at d=1536 on
//! Apple Silicon (see `examples/probe_batched_dot.rs`). For single
//! queries (`nq == 1`) it falls back to a parallel `dot_f32` per
//! database vector — GEMM setup overhead doesn't pay off at GEMV
//! scale.
//!
//! With the `faer-fallback` feature enabled, `ip_batch` reverts to the
//! v1.1 path (faer matmul + scalar GEMV). Kept for one release as an
//! escape hatch; remove once v1.2 is field-validated.

use faer::Par;
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Inner-product matrix `Q @ X^T`, shape `(nq, n)`.
///
/// `par` selects the matmul parallelism strategy:
/// - `Par::Rayon(...)` — distribute work across rayon threads. Use this
///   when the caller itself is **not** already in a rayon context (e.g. the
///   PyO3 entry points for `search` and `range_search`).
/// - `Par::Seq` — single-threaded matmul. Use this when the caller IS already
///   running under rayon (e.g. `similarity_graph`'s per-tile loop), to avoid
///   nested-parallelism contention.
pub fn ip_batch(queries: ArrayView2<f32>, data: ArrayView2<f32>, par: Par) -> Array2<f32> {
    let (nq, dq) = queries.dim();
    let (n, dx) = data.dim();
    debug_assert_eq!(dq, dx, "dim mismatch in ip_batch");
    let d = dq;

    if nq == 0 || n == 0 {
        return Array2::<f32>::zeros((nq, n));
    }

    if nq == 1 {
        return gemv_one_query(queries, data);
    }

    let q_slice = queries.as_slice().expect("contiguous queries");
    let x_slice = data.as_slice().expect("contiguous data");
    let mut out_data: Vec<f32> = vec![0.0; nq * n];

    #[cfg(not(feature = "faer-fallback"))]
    {
        if matches!(par, Par::Seq) {
            rustcluster_simd::batched_dot_tile(q_slice, nq, x_slice, n, d, &mut out_data);
        } else {
            // Par::Rayon — parallelize over chunks of query rows. 32 rows
            // per chunk gives ample work for batched_dot_tile's register
            // blocking while keeping enough chunks for rayon load balance.
            const Q_CHUNK: usize = 32;
            out_data
                .par_chunks_mut(Q_CHUNK * n)
                .enumerate()
                .for_each(|(chunk_idx, out_chunk)| {
                    let q_start = chunk_idx * Q_CHUNK;
                    let q_chunk_h = out_chunk.len() / n;
                    let q_chunk = &q_slice[q_start * d..(q_start + q_chunk_h) * d];
                    rustcluster_simd::batched_dot_tile(
                        q_chunk, q_chunk_h, x_slice, n, d, out_chunk,
                    );
                });
        }
    }

    #[cfg(feature = "faer-fallback")]
    {
        use faer::linalg::matmul::matmul;
        use faer::mat::{MatMut, MatRef};
        use faer::Accum;

        let q_ref = MatRef::from_row_major_slice(q_slice, nq, d);
        let x_ref = MatRef::from_row_major_slice(x_slice, n, d);
        let out_mut = MatMut::from_row_major_slice_mut(&mut out_data, nq, n);
        matmul(
            out_mut,
            Accum::Replace,
            q_ref,
            x_ref.transpose(),
            1.0_f32,
            par,
        );
    }

    Array2::from_shape_vec((nq, n), out_data).expect("size matches by construction")
}

#[cfg(not(feature = "faer-fallback"))]
fn gemv_one_query(queries: ArrayView2<f32>, data: ArrayView2<f32>) -> Array2<f32> {
    let (_, d) = queries.dim();
    let (n, _) = data.dim();
    let q = queries.row(0);
    let q_slice = q.as_slice().expect("contiguous query row");
    let data_slice = data.as_slice().expect("contiguous data");
    let mut out = Array2::<f32>::zeros((1, n));
    let row = out.row_mut(0);
    let row_slice = row.into_slice().unwrap();
    row_slice.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let x = &data_slice[i * d..(i + 1) * d];
        *dst = rustcluster_simd::dot_f32(q_slice, x);
    });
    out
}

#[cfg(feature = "faer-fallback")]
fn gemv_one_query(queries: ArrayView2<f32>, data: ArrayView2<f32>) -> Array2<f32> {
    let (_, d) = queries.dim();
    let (n, _) = data.dim();
    let q = queries.row(0);
    let q_slice = q.as_slice().expect("contiguous query row");
    let data_slice = data.as_slice().expect("contiguous data");
    let mut out = Array2::<f32>::zeros((1, n));
    let row = out.row_mut(0);
    let row_slice = row.into_slice().unwrap();
    row_slice.par_iter_mut().enumerate().for_each(|(i, dst)| {
        let x = &data_slice[i * d..(i + 1) * d];
        let mut acc = 0.0f32;
        for k in 0..d {
            acc += q_slice[k] * x[k];
        }
        *dst = acc;
    });
    out
}

/// Convert an inner-product matrix to squared-L2 distances in place.
///
/// `||q - x||² = ||q||² + ||x||² - 2 q·x`. Caller supplies precomputed norms.
pub fn ip_to_l2_sq(ip: &mut Array2<f32>, q_norms_sq: &[f32], x_norms_sq: &[f32]) {
    let (nq, n) = ip.dim();
    debug_assert_eq!(q_norms_sq.len(), nq);
    debug_assert_eq!(x_norms_sq.len(), n);

    ip.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let qn = q_norms_sq[i];
            let row_slice = row.as_slice_mut().expect("contiguous row");
            for j in 0..n {
                // -2*q.x + ||q||² + ||x||²; clamp tiny negatives from rounding.
                let v = qn + x_norms_sq[j] - 2.0 * row_slice[j];
                row_slice[j] = if v < 0.0 { 0.0 } else { v };
            }
        });
}

/// Compute squared L2 norms of each row.
pub fn row_norms_sq(data: ArrayView2<f32>) -> Vec<f32> {
    let (n, d) = data.dim();
    let slice = data.as_slice().expect("contiguous");
    (0..n)
        .into_par_iter()
        .map(|i| {
            let row = &slice[i * d..(i + 1) * d];
            let mut acc = 0.0f32;
            for &v in row {
                acc += v * v;
            }
            acc
        })
        .collect()
}
