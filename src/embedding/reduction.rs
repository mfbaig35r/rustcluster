//! Randomized PCA for dimensionality reduction.
//!
//! Halko-Martinsson-Tropp algorithm for fast approximate truncated SVD.
//! Much faster than full SVD when target rank << min(n, d).
//!
//! Reference: Halko, Martinsson, Tropp (2011), "Finding Structure with
//! Randomness: Probabilistic Algorithms for Constructing Approximate
//! Matrix Decompositions."

use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::distance::Scalar;

/// Result of randomized PCA.
pub struct PcaProjection {
    /// Projection matrix, shape (input_dim, output_dim), stored flat row-major.
    pub components: Vec<f64>,
    /// Mean of each dimension (for centering).
    pub mean: Vec<f64>,
    pub input_dim: usize,
    pub output_dim: usize,
}

/// Compute a randomized PCA projection from data.
///
/// `data`: flat row-major f64, shape (n, d)
/// `target_dim`: desired output dimensionality
/// `oversampling`: extra dimensions for accuracy (default 10)
pub fn compute_pca(
    data: &[f64],
    n: usize,
    d: usize,
    target_dim: usize,
    oversampling: usize,
    rng: &mut StdRng,
) -> PcaProjection {
    let k = (target_dim + oversampling).min(d).min(n);

    // Step 1: Compute and subtract column means
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            mean[j] += data[i * d + j];
        }
    }
    for j in 0..d {
        mean[j] /= n as f64;
    }

    // Centered data (we'll work with it implicitly in the matmul)

    // Step 2: Generate random Gaussian matrix Omega: (d, k)
    let normal = StandardNormal;
    let omega: Vec<f64> = (0..d * k).map(|_| normal.sample(rng)).collect();

    // Step 3: Y = (X - mean) * Omega, shape (n, k)
    // Parallelized over rows of X
    let mut y = vec![0.0f64; n * k];
    y.par_chunks_mut(k)
        .enumerate()
        .for_each(|(i, y_row)| {
            for col in 0..k {
                let mut acc = 0.0f64;
                for j in 0..d {
                    acc += (data[i * d + j] - mean[j]) * omega[j * k + col];
                }
                y_row[col] = acc;
            }
        });

    // Step 4: QR decomposition of Y → Q (n, k)
    // Using modified Gram-Schmidt (simple, stable enough for this purpose)
    let q = qr_thin_q(&y, n, k);

    // Step 5: B = Q^T * (X - mean), shape (k, d)
    let mut b = vec![0.0f64; k * d];
    // Each row of B: b_row[j] = sum_i q[i * k + row] * (data[i * d + j] - mean[j])
    for row in 0..k {
        for j in 0..d {
            let mut acc = 0.0f64;
            for i in 0..n {
                acc += q[i * k + row] * (data[i * d + j] - mean[j]);
            }
            b[row * d + j] = acc;
        }
    }

    // Step 6: SVD of B (k x d matrix, k is small)
    // We need the right singular vectors V (d x k)
    // B = U_b * S * V^T  →  V columns are the principal directions
    //
    // Instead of full SVD, compute B^T * B (d x d is too big if d=1536)
    // Better: B * B^T (k x k, small!), get eigendecomposition
    // B * B^T = U_b * S^2 * U_b^T
    // Then V = B^T * U_b * S^{-1}

    // Compute C = B * B^T (k x k)
    let mut c = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut acc = 0.0f64;
            for l in 0..d {
                acc += b[i * d + l] * b[j * d + l];
            }
            c[i * k + j] = acc;
            c[j * k + i] = acc;
        }
    }

    // Eigendecomposition of C (symmetric k x k) via symmetric QR iteration
    // C is small (at most ~140 x 140), so this is fast
    let (eigenvalues, eigenvectors) = symmetric_eigen(&c, k);

    // eigenvalues/eigenvectors are sorted descending by eigenvalue
    let actual_dim = target_dim.min(k);

    // Compute V = B^T * U_b * S^{-1} for top actual_dim components
    // V shape: (d, actual_dim)
    let mut components = vec![0.0f64; d * actual_dim];
    for comp in 0..actual_dim {
        let eigenval = eigenvalues[comp];
        let s_inv = if eigenval > 1e-30 { 1.0 / eigenval.sqrt() } else { 0.0 };

        // u_col = eigenvectors column comp (stored as eigenvectors[i * k + comp])
        // v_col = B^T * u_col * s_inv
        for j in 0..d {
            let mut acc = 0.0f64;
            for i in 0..k {
                acc += b[i * d + j] * eigenvectors[i * k + comp];
            }
            components[j * actual_dim + comp] = acc * s_inv;
        }
    }

    PcaProjection {
        components,
        mean,
        input_dim: d,
        output_dim: actual_dim,
    }
}

/// Project data using a pre-computed PCA projection.
/// Input: data (n * input_dim), projection
/// Output: projected (n * output_dim)
pub fn project_data<F: Scalar>(
    data: &[F],
    n: usize,
    projection: &PcaProjection,
) -> Vec<F> {
    let d = projection.input_dim;
    let target = projection.output_dim;
    let mut out = vec![F::zero(); n * target];

    out.par_chunks_mut(target)
        .enumerate()
        .for_each(|(i, out_row)| {
            for col in 0..target {
                let mut acc = 0.0f64;
                for j in 0..d {
                    let val = data[i * d + j].to_f64_lossy() - projection.mean[j];
                    acc += val * projection.components[j * target + col];
                }
                out_row[col] = F::from_f64_lossy(acc);
            }
        });

    out
}

/// Thin QR decomposition via modified Gram-Schmidt.
/// Input: A (n x k), flat row-major.
/// Returns Q (n x k), flat row-major, columns orthonormal.
fn qr_thin_q(a: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut q = a.to_vec();

    for col in 0..k {
        // Subtract projections onto previous columns
        for prev in 0..col {
            let mut dot = 0.0f64;
            for row in 0..n {
                dot += q[row * k + col] * q[row * k + prev];
            }
            for row in 0..n {
                q[row * k + col] -= dot * q[row * k + prev];
            }
        }

        // Normalize
        let mut norm_sq = 0.0f64;
        for row in 0..n {
            norm_sq += q[row * k + col] * q[row * k + col];
        }
        let norm = norm_sq.sqrt();
        if norm > 1e-30 {
            let inv = 1.0 / norm;
            for row in 0..n {
                q[row * k + col] *= inv;
            }
        }
    }

    q
}

/// Eigendecomposition of a small symmetric matrix via deflated power iteration.
/// Returns (eigenvalues, eigenvectors) sorted descending by eigenvalue.
/// eigenvectors stored flat row-major: eigenvectors[i * n + comp] = i-th element of comp-th eigenvector.
fn symmetric_eigen(mat: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut eigenvalues = Vec::with_capacity(n);
    let mut eigenvectors = vec![0.0f64; n * n];

    // Work on a copy that we deflate
    let mut work = mat.to_vec();

    for comp in 0..n {
        // Power iteration for top eigenvector of current deflated matrix
        let mut v = vec![0.0f64; n];
        // Initialize with a non-zero vector
        v[comp % n] = 1.0;

        let max_power_iters = 200;
        for _ in 0..max_power_iters {
            // w = work * v
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                let mut acc = 0.0f64;
                for j in 0..n {
                    acc += work[i * n + j] * v[j];
                }
                w[i] = acc;
            }

            // Normalize
            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-30 {
                break;
            }
            for x in w.iter_mut() {
                *x /= norm;
            }

            // Check convergence: |w - v| < eps
            let diff: f64 = w.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            v = w;
            if diff < 1e-12 {
                break;
            }
        }

        // Eigenvalue = v^T * work * v
        let mut eigenval = 0.0f64;
        for i in 0..n {
            let mut row_dot = 0.0f64;
            for j in 0..n {
                row_dot += work[i * n + j] * v[j];
            }
            eigenval += v[i] * row_dot;
        }

        eigenvalues.push(eigenval);
        for i in 0..n {
            eigenvectors[i * n + comp] = v[i];
        }

        // Deflate: work -= eigenval * v * v^T
        for i in 0..n {
            for j in 0..n {
                work[i * n + j] -= eigenval * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_pca_reduces_dimension() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;
        let d = 20;
        let target = 5;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();

        let proj = compute_pca(&data, n, d, target, 5, &mut rng);
        assert_eq!(proj.input_dim, d);
        assert_eq!(proj.output_dim, target);

        let projected = project_data::<f64>(&data, n, &proj);
        assert_eq!(projected.len(), n * target);
    }

    #[test]
    fn test_pca_preserves_variance() {
        // Data with clear principal direction along axis 0
        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let d = 10;
        let mut data = vec![0.0f64; n * d];
        for i in 0..n {
            data[i * d] = rng.gen_range(-10.0..10.0); // high variance
            for j in 1..d {
                data[i * d + j] = rng.gen_range(-0.1..0.1); // low variance
            }
        }

        let proj = compute_pca(&data, n, d, 2, 5, &mut rng);
        let projected = project_data::<f64>(&data, n, &proj);

        // First PC should capture most of the variance
        let var_pc1: f64 = (0..n).map(|i| projected[i * 2] * projected[i * 2]).sum::<f64>() / n as f64;
        let var_pc2: f64 = (0..n).map(|i| projected[i * 2 + 1] * projected[i * 2 + 1]).sum::<f64>() / n as f64;
        assert!(var_pc1 > var_pc2 * 5.0, "PC1 var={var_pc1}, PC2 var={var_pc2}");
    }

    #[test]
    fn test_pca_f32_projection() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 50;
        let d = 10;
        let data_f64: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();

        let proj = compute_pca(&data_f64, n, d, 3, 5, &mut rng);
        let projected = project_data::<f32>(&data_f32, n, &proj);
        assert_eq!(projected.len(), n * 3);
    }

    #[test]
    fn test_qr_orthonormal() {
        let a = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let q = qr_thin_q(&a, 4, 3);

        // Check columns are unit norm
        for col in 0..3 {
            let norm: f64 = (0..4).map(|row| q[row * 3 + col] * q[row * 3 + col]).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Column {col} norm = {norm}");
        }

        // Check columns are orthogonal
        for c1 in 0..3 {
            for c2 in (c1 + 1)..3 {
                let dot: f64 = (0..4).map(|row| q[row * 3 + c1] * q[row * 3 + c2]).sum();
                assert!(dot.abs() < 1e-10, "Dot({c1},{c2}) = {dot}");
            }
        }
    }
}
