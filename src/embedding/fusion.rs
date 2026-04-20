//! Multi-view fusion for embedding clustering.
//!
//! Weighted concatenation of two embedding views, re-normalized to the
//! unit hypersphere. Spherical K-means on the fused representation
//! maximizes a weighted sum of per-view cosine similarities.

use rayon::prelude::*;

/// Fuse two embedding views via weighted concatenation and re-normalization.
///
/// z_tilde = [sqrt(w_a) * z_a, sqrt(w_b) * z_b]
/// z = z_tilde / ||z_tilde||
///
/// Both inputs must be L2-normalized. Output is L2-normalized.
pub fn fuse_views(
    data_a: &[f64],
    data_b: &[f64],
    n: usize,
    d_a: usize,
    d_b: usize,
    weight_a: f64,
    weight_b: f64,
) -> (Vec<f64>, usize) {
    let d_fused = d_a + d_b;
    let sqrt_wa = weight_a.sqrt();
    let sqrt_wb = weight_b.sqrt();

    let mut out = vec![0.0f64; n * d_fused];

    out.par_chunks_mut(d_fused)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..d_a {
                row[j] = sqrt_wa * data_a[i * d_a + j];
            }
            for j in 0..d_b {
                row[d_a + j] = sqrt_wb * data_b[i * d_b + j];
            }
            // L2-normalize
            let norm_sq: f64 = row.iter().map(|v| v * v).sum();
            if norm_sq > 1e-30 {
                let inv = 1.0 / norm_sq.sqrt();
                for v in row.iter_mut() {
                    *v *= inv;
                }
            }
        });

    (out, d_fused)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_shape() {
        let a = vec![1.0, 0.0, 0.0, 1.0]; // 2 points, d=2
        let b = vec![0.0, 1.0, 1.0, 0.0]; // 2 points, d=2
        let (fused, d) = fuse_views(&a, &b, 2, 2, 2, 0.5, 0.5);
        assert_eq!(d, 4);
        assert_eq!(fused.len(), 8);
    }

    #[test]
    fn test_fuse_unit_norm() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![0.0, 1.0, 1.0, 0.0];
        let (fused, d) = fuse_views(&a, &b, 2, 2, 2, 0.7, 0.3);
        for i in 0..2 {
            let norm: f64 = fused[i * d..(i + 1) * d]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Row {i} norm = {norm}");
        }
    }

    #[test]
    fn test_fuse_weight_one_zero() {
        // With w_b=0, the fused representation should depend only on a
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let (fused, _) = fuse_views(&a, &b, 1, 2, 2, 1.0, 0.0);
        // First 2 dims should have all the weight, last 2 should be 0
        assert!(fused[2].abs() < 1e-10);
        assert!(fused[3].abs() < 1e-10);
    }
}
