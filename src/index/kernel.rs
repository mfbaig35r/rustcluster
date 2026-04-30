//! Distance and inner-product kernels for flat indexes.
//!
//! For batch search we lean on faer GEMM: `Q (nq×d) @ X^T (d×n) → S (nq×n)`,
//! then convert to L2 via the trick `||q-x||² = ||q||² + ||x||² - 2 q·x`.
//!
//! For single queries (`nq == 1`) we fall back to a manual GEMV loop —
//! faer's setup overhead doesn't pay back at that size.
//!
//! Inputs and outputs are wrapped as faer `MatRef` / `MatMut` views over the
//! ndarray storage — zero copy on the way in and out. This is the difference
//! between losing 2x to FAISS at d=1536 and matching it.

use faer::linalg::matmul::matmul;
use faer::mat::{MatMut, MatRef};
use faer::{Accum, Par};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Inner-product matrix `Q @ X^T`, shape `(nq, n)`.
///
/// Uses faer GEMM for batch sizes ≥ 2; manual loop otherwise.
pub fn ip_batch(queries: ArrayView2<f32>, data: ArrayView2<f32>) -> Array2<f32> {
    let (nq, dq) = queries.dim();
    let (n, dx) = data.dim();
    debug_assert_eq!(dq, dx, "dim mismatch in ip_batch");
    let d = dq;

    if nq == 0 || n == 0 {
        return Array2::<f32>::zeros((nq, n));
    }

    if nq == 1 {
        // GEMV-shaped: one query against many database vectors.
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
        return out;
    }

    // Batched: zero-copy faer GEMM. Wrap ndarray storage as MatRef / MatMut,
    // compute scores = Q · X^T directly into the output buffer.
    let q_slice = queries.as_slice().expect("contiguous queries");
    let x_slice = data.as_slice().expect("contiguous data");
    let q_ref = MatRef::from_row_major_slice(q_slice, nq, d);
    let x_ref = MatRef::from_row_major_slice(x_slice, n, d);

    let mut out_data: Vec<f32> = vec![0.0; nq * n];
    {
        let out_mut = MatMut::from_row_major_slice_mut(&mut out_data, nq, n);
        matmul(
            out_mut,
            Accum::Replace,
            q_ref,
            x_ref.transpose(),
            1.0_f32,
            Par::Seq,
        );
    }
    Array2::from_shape_vec((nq, n), out_data).expect("size matches by construction")
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
