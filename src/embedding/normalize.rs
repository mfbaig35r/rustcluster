//! L2 normalization for embedding vectors.

use crate::distance::Scalar;
use rayon::prelude::*;

/// L2-normalize each row of a flat row-major buffer.
/// Returns a new normalized buffer + count of zero-norm rows.
pub fn l2_normalize_rows<F: Scalar>(data: &[F], n: usize, d: usize) -> (Vec<F>, usize) {
    let mut out = data.to_vec();
    let zeros = l2_normalize_rows_inplace(&mut out, n, d);
    (out, zeros)
}

/// L2-normalize each row in-place. Returns count of zero-norm rows (left as zeros).
pub fn l2_normalize_rows_inplace<F: Scalar>(data: &mut [F], n: usize, d: usize) -> usize {
    let zero_count = std::sync::atomic::AtomicUsize::new(0);

    data.par_chunks_mut(d).for_each(|row| {
        let norm_sq: f64 = row
            .iter()
            .map(|&v| {
                let vf = v.to_f64_lossy();
                vf * vf
            })
            .sum();

        if norm_sq > 1e-30 {
            let inv_norm = F::from_f64_lossy(1.0 / norm_sq.sqrt());
            for v in row.iter_mut() {
                *v = *v * inv_norm;
            }
        } else {
            zero_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });

    zero_count.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_vector_unchanged() {
        let mut data = vec![1.0f64, 0.0, 0.0];
        l2_normalize_rows_inplace(&mut data, 1, 3);
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!(data[1].abs() < 1e-10);
    }

    #[test]
    fn test_normalizes_to_unit() {
        let mut data = vec![3.0f64, 4.0];
        l2_normalize_rows_inplace(&mut data, 1, 2);
        let norm: f64 = data.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_vector_stays_zero() {
        let mut data = vec![0.0f64, 0.0, 0.0];
        let zeros = l2_normalize_rows_inplace(&mut data, 1, 3);
        assert_eq!(zeros, 1);
        assert!(data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_multiple_rows() {
        let mut data = vec![3.0f64, 4.0, 0.0, 0.0, 1.0, 1.0];
        let zeros = l2_normalize_rows_inplace(&mut data, 3, 2);
        assert_eq!(zeros, 1); // row [0,0]
                              // Row 0: [0.6, 0.8]
        assert!((data[0] - 0.6).abs() < 1e-10);
        assert!((data[1] - 0.8).abs() < 1e-10);
        // Row 1: [0, 0] (zero)
        // Row 2: normalized [1/sqrt2, 1/sqrt2]
        let norm2: f64 = data[4] * data[4] + data[5] * data[5];
        assert!((norm2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_f32() {
        let mut data = vec![3.0f32, 4.0];
        l2_normalize_rows_inplace(&mut data, 1, 2);
        let norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
