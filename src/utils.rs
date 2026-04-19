use ndarray::ArrayView2;

use crate::error::KMeansError;

/// Squared Euclidean distance between two equal-length slices.
///
/// This is the hot inner kernel — structured as a simple counted loop over
/// contiguous slices so LLVM can auto-vectorize to SIMD (SSE2/AVX2 on x86,
/// NEON on ARM).
#[inline(always)]
pub fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

/// Find the nearest centroid for a single point.
///
/// `centroids` is a flat row-major buffer of shape (k, d).
/// Returns (cluster_index, squared_distance).
#[inline]
pub fn assign_nearest(point: &[f64], centroids: &[f64], k: usize, d: usize) -> (usize, f64) {
    debug_assert_eq!(centroids.len(), k * d);
    debug_assert_eq!(point.len(), d);

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;

    for cluster in 0..k {
        let centroid = &centroids[cluster * d..(cluster + 1) * d];
        let dist = squared_euclidean(point, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = cluster;
        }
    }

    (best_idx, best_dist)
}

/// Validate that input data is non-empty and contains only finite values.
pub fn validate_data(data: &ArrayView2<f64>) -> Result<(), KMeansError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(KMeansError::EmptyInput);
    }

    // Check for NaN/Inf — iterate the raw slice (C-contiguous guaranteed by caller)
    if let Some(slice) = data.as_slice() {
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(KMeansError::NonFinite);
        }
    } else {
        // Fallback for non-contiguous (shouldn't happen after boundary check)
        for row in data.rows() {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(KMeansError::NonFinite);
            }
        }
    }

    Ok(())
}

/// Validate that prediction data matches the fitted model's dimensionality.
pub fn validate_predict_data(
    data: &ArrayView2<f64>,
    expected_d: usize,
) -> Result<(), KMeansError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(KMeansError::EmptyInput);
    }
    if d != expected_d {
        return Err(KMeansError::DimensionMismatch {
            expected: expected_d,
            got: d,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_squared_euclidean_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        // (3^2 + 3^2 + 3^2) = 27
        assert!((squared_euclidean(&a, &b) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_identical() {
        let a = [1.0, 2.0, 3.0];
        assert!((squared_euclidean(&a, &a) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_zero_vectors() {
        let a = [0.0, 0.0];
        let b = [0.0, 0.0];
        assert!((squared_euclidean(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_nearest() {
        // Two centroids in 2D: (0,0) and (10,10)
        let centroids = [0.0, 0.0, 10.0, 10.0];
        let k = 2;
        let d = 2;

        // Point near first centroid
        let (idx, dist) = assign_nearest(&[1.0, 1.0], &centroids, k, d);
        assert_eq!(idx, 0);
        assert!((dist - 2.0).abs() < 1e-10);

        // Point near second centroid
        let (idx, dist) = assign_nearest(&[9.0, 9.0], &centroids, k, d);
        assert_eq!(idx, 1);
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_nearest_three_centroids() {
        // Three centroids: (0,0), (5,5), (10,0)
        let centroids = [0.0, 0.0, 5.0, 5.0, 10.0, 0.0];
        let k = 3;
        let d = 2;

        let (idx, _) = assign_nearest(&[4.0, 4.0], &centroids, k, d);
        assert_eq!(idx, 1); // closest to (5,5)

        let (idx, _) = assign_nearest(&[9.0, 1.0], &centroids, k, d);
        assert_eq!(idx, 2); // closest to (10,0)
    }

    #[test]
    fn test_validate_data_ok() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(validate_data(&data.view()).is_ok());
    }

    #[test]
    fn test_validate_data_empty() {
        let data = ndarray::Array2::<f64>::zeros((0, 2));
        assert!(matches!(
            validate_data(&data.view()),
            Err(KMeansError::EmptyInput)
        ));
    }

    #[test]
    fn test_validate_data_nan() {
        let data = array![[1.0, f64::NAN], [3.0, 4.0]];
        assert!(matches!(
            validate_data(&data.view()),
            Err(KMeansError::NonFinite)
        ));
    }

    #[test]
    fn test_validate_data_inf() {
        let data = array![[1.0, f64::INFINITY], [3.0, 4.0]];
        assert!(matches!(
            validate_data(&data.view()),
            Err(KMeansError::NonFinite)
        ));
    }

    #[test]
    fn test_validate_predict_data_dimension_mismatch() {
        let data = array![[1.0, 2.0, 3.0]];
        assert!(matches!(
            validate_predict_data(&data.view(), 2),
            Err(KMeansError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }
}
