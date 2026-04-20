//! Randomized PCA for dimensionality reduction.
//!
//! Halko-Martinsson-Tropp algorithm for fast approximate truncated SVD.
//! Much faster than full SVD when target rank << min(n, d).
//!
//! Uses faer for optimized dense matrix multiplication (GEMM) and SVD.
//! The big matmuls use faer's cache-blocked, SIMD-accelerated kernels.
//!
//! Reference: Halko, Martinsson, Tropp (2011), "Finding Structure with
//! Randomness: Probabilistic Algorithms for Constructing Approximate
//! Matrix Decompositions."

use faer::prelude::*;
use faer::Mat;
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

    // Step 1: Compute column means
    let mut mean = vec![0.0f64; d];
    for i in 0..n {
        for j in 0..d {
            mean[j] += data[i * d + j];
        }
    }
    for j in 0..d {
        mean[j] /= n as f64;
    }

    // Step 2: Generate random Gaussian matrix Omega: (d, k)
    let normal = StandardNormal;
    let omega: Vec<f64> = (0..d * k).map(|_| normal.sample(rng)).collect();

    // Step 3: Y = (X - mean) * Omega, shape (n, k)
    // This is THE bottleneck (323K × 1536 × 138 = 69 billion multiply-adds).
    // faer's blocked, SIMD-accelerated GEMM replaces our naive row-parallel loops.
    let x_centered = Mat::from_fn(n, d, |i, j| data[i * d + j] - mean[j]);
    let omega_mat = Mat::from_fn(d, k, |i, j| omega[i * k + j]);
    let y_mat = &x_centered * &omega_mat;

    // Step 4: QR decomposition of Y → Q (n, k)
    // Using modified Gram-Schmidt — works well for k ≤ ~140.
    let y_flat = faer_to_flat_row_major(&y_mat);
    let q_flat = qr_thin_q(&y_flat, n, k);

    // Step 5: B = Q^T * X_centered, shape (k, d)
    // Second big matmul — faer GEMM for cache-blocked performance.
    let q_mat = Mat::from_fn(n, k, |i, j| q_flat[i * k + j]);
    let b_mat = q_mat.transpose() * &x_centered;

    // Step 6: Thin SVD of B (k × d matrix, k is small).
    // B = U_b * S * V^T → V columns are the principal directions.
    // faer's SVD is much more robust than our old power-iteration eigendecomp.
    // Singular values are sorted in descending order.
    let svd = b_mat
        .thin_svd()
        .expect("SVD of small matrix should not fail");

    let actual_dim = target_dim.min(k);

    // Extract top actual_dim right singular vectors as components.
    // V from thin SVD has shape (d, k), columns are right singular vectors.
    let v = svd.V();
    let mut components = vec![0.0f64; d * actual_dim];
    for j in 0..d {
        for comp in 0..actual_dim {
            components[j * actual_dim + comp] = v[(j, comp)];
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
pub fn project_data<F: Scalar>(data: &[F], n: usize, projection: &PcaProjection) -> Vec<F> {
    let d = projection.input_dim;
    let target = projection.output_dim;

    // For large n, use faer GEMM; for small n, manual loops are fine.
    if n >= 1000 {
        let x_centered = Mat::from_fn(n, d, |i, j| {
            data[i * d + j].to_f64_lossy() - projection.mean[j]
        });
        let comp_mat = Mat::from_fn(d, target, |i, j| projection.components[i * target + j]);
        let result_mat = &x_centered * &comp_mat;

        let mut out = vec![F::zero(); n * target];
        for i in 0..n {
            for j in 0..target {
                out[i * target + j] = F::from_f64_lossy(result_mat[(i, j)]);
            }
        }
        out
    } else {
        // Small n — avoid faer allocation overhead
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
}

/// Convert a faer Mat to flat row-major Vec<f64>.
fn faer_to_flat_row_major(mat: &Mat<f64>) -> Vec<f64> {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = mat[(i, j)];
        }
    }
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
        let var_pc1: f64 = (0..n)
            .map(|i| projected[i * 2] * projected[i * 2])
            .sum::<f64>()
            / n as f64;
        let var_pc2: f64 = (0..n)
            .map(|i| projected[i * 2 + 1] * projected[i * 2 + 1])
            .sum::<f64>()
            / n as f64;
        assert!(
            var_pc1 > var_pc2 * 5.0,
            "PC1 var={var_pc1}, PC2 var={var_pc2}"
        );
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
        let a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let q = qr_thin_q(&a, 4, 3);

        // Check columns are unit norm
        for col in 0..3 {
            let norm: f64 = (0..4)
                .map(|row| q[row * 3 + col] * q[row * 3 + col])
                .sum::<f64>()
                .sqrt();
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
