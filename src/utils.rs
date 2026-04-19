use ndarray::ArrayView2;
use num_traits::Float;

use crate::distance::{Distance, Scalar};
use crate::error::ClusterError;

// ---- Generic kernels (Layer 3) ----

/// Squared Euclidean distance between two equal-length slices.
///
/// The hot inner kernel — simple counted loop for LLVM auto-vectorization.
#[inline(always)]
pub fn squared_euclidean_generic<F: Scalar>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = F::zero();
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc = acc + diff * diff;
    }
    acc
}

/// Find nearest centroid using a generic distance metric and scalar type.
#[inline]
pub fn assign_nearest_with<F: Scalar, D: Distance<F>>(
    point: &[F],
    centroids: &[F],
    k: usize,
    d: usize,
) -> (usize, F) {
    debug_assert_eq!(centroids.len(), k * d);
    debug_assert_eq!(point.len(), d);

    let mut best_idx = 0;
    let mut best_dist = F::max_value();

    for cluster in 0..k {
        let centroid = &centroids[cluster * d..(cluster + 1) * d];
        let dist = D::distance(point, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_idx = cluster;
        }
    }

    (best_idx, best_dist)
}

/// Find nearest and second-nearest centroid, generic over distance and scalar.
#[inline]
pub fn assign_nearest_two_with<F: Scalar, D: Distance<F>>(
    point: &[F],
    centroids: &[F],
    k: usize,
    d: usize,
) -> (usize, F, F) {
    debug_assert!(k >= 2);
    debug_assert_eq!(centroids.len(), k * d);
    debug_assert_eq!(point.len(), d);

    let mut best_idx = 0;
    let mut best_dist = F::max_value();
    let mut second_dist = F::max_value();

    for cluster in 0..k {
        let centroid = &centroids[cluster * d..(cluster + 1) * d];
        let dist = D::distance(point, centroid);
        if dist < best_dist {
            second_dist = best_dist;
            best_dist = dist;
            best_idx = cluster;
        } else if dist < second_dist {
            second_dist = dist;
        }
    }

    (best_idx, best_dist, second_dist)
}

// ---- Validation (generic over Scalar) ----

/// Validate that input data is non-empty and contains only finite values.
pub fn validate_data_generic<F: Scalar>(data: &ArrayView2<F>) -> Result<(), ClusterError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }

    if let Some(slice) = data.as_slice() {
        if slice.iter().any(|v| !v.is_finite()) {
            return Err(ClusterError::NonFinite);
        }
    } else {
        for row in data.rows() {
            if row.iter().any(|v| !v.is_finite()) {
                return Err(ClusterError::NonFinite);
            }
        }
    }

    Ok(())
}

/// Validate that prediction data matches the fitted model's dimensionality.
pub fn validate_predict_data_generic<F: Scalar>(
    data: &ArrayView2<F>,
    expected_d: usize,
) -> Result<(), ClusterError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }
    if d != expected_d {
        return Err(ClusterError::DimensionMismatch {
            expected: expected_d,
            got: d,
        });
    }
    Ok(())
}

// ---- f64-specific wrappers (backward compat for bench API) ----

/// Squared Euclidean distance (f64).
#[inline(always)]
pub fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_generic(a, b)
}

/// Find nearest centroid (f64, squared Euclidean).
#[inline]
pub fn assign_nearest(point: &[f64], centroids: &[f64], k: usize, d: usize) -> (usize, f64) {
    use crate::distance::SquaredEuclidean;
    assign_nearest_with::<f64, SquaredEuclidean>(point, centroids, k, d)
}

/// Find nearest and second-nearest centroid (f64, squared Euclidean).
#[inline]
pub fn assign_nearest_two(
    point: &[f64],
    centroids: &[f64],
    k: usize,
    d: usize,
) -> (usize, f64, f64) {
    use crate::distance::SquaredEuclidean;
    assign_nearest_two_with::<f64, SquaredEuclidean>(point, centroids, k, d)
}

/// Validate f64 data.
pub fn validate_data(data: &ArrayView2<f64>) -> Result<(), ClusterError> {
    validate_data_generic(data)
}

/// Validate f64 prediction data.
pub fn validate_predict_data(
    data: &ArrayView2<f64>,
    expected_d: usize,
) -> Result<(), ClusterError> {
    validate_predict_data_generic(data, expected_d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_squared_euclidean_basic() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((squared_euclidean(&a, &b) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_f32() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((squared_euclidean_generic(&a, &b) - 27.0).abs() < 1e-4);
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
        let centroids = [0.0, 0.0, 10.0, 10.0];
        let k = 2;
        let d = 2;

        let (idx, dist) = assign_nearest(&[1.0, 1.0], &centroids, k, d);
        assert_eq!(idx, 0);
        assert!((dist - 2.0).abs() < 1e-10);

        let (idx, dist) = assign_nearest(&[9.0, 9.0], &centroids, k, d);
        assert_eq!(idx, 1);
        assert!((dist - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_nearest_three_centroids() {
        let centroids = [0.0, 0.0, 5.0, 5.0, 10.0, 0.0];
        let (idx, _) = assign_nearest(&[4.0, 4.0], &centroids, 3, 2);
        assert_eq!(idx, 1);
        let (idx, _) = assign_nearest(&[9.0, 1.0], &centroids, 3, 2);
        assert_eq!(idx, 2);
    }

    #[test]
    fn test_assign_nearest_two() {
        let centroids = [0.0, 0.0, 5.0, 5.0, 10.0, 0.0];
        let (idx, best, second) = assign_nearest_two(&[1.0, 1.0], &centroids, 3, 2);
        assert_eq!(idx, 0);
        assert!((best - 2.0).abs() < 1e-10);
        assert!((second - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_assign_nearest_two_symmetric() {
        let centroids = [0.0, 0.0, 10.0, 0.0];
        let (idx, best, second) = assign_nearest_two(&[5.0, 0.0], &centroids, 2, 2);
        assert!((best - 25.0).abs() < 1e-10);
        assert!((second - 25.0).abs() < 1e-10);
        assert!(idx == 0 || idx == 1);
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
            Err(ClusterError::EmptyInput)
        ));
    }

    #[test]
    fn test_validate_data_nan() {
        let data = array![[1.0, f64::NAN], [3.0, 4.0]];
        assert!(matches!(
            validate_data(&data.view()),
            Err(ClusterError::NonFinite)
        ));
    }

    #[test]
    fn test_validate_data_inf() {
        let data = array![[1.0, f64::INFINITY], [3.0, 4.0]];
        assert!(matches!(
            validate_data(&data.view()),
            Err(ClusterError::NonFinite)
        ));
    }

    #[test]
    fn test_validate_predict_data_dimension_mismatch() {
        let data = array![[1.0, 2.0, 3.0]];
        assert!(matches!(
            validate_predict_data(&data.view(), 2),
            Err(ClusterError::DimensionMismatch {
                expected: 2,
                got: 3
            })
        ));
    }

    #[test]
    fn test_validate_f32_data() {
        let data = ndarray::array![[1.0f32, 2.0], [3.0, 4.0]];
        assert!(validate_data_generic(&data.view()).is_ok());

        let bad = ndarray::array![[1.0f32, f32::NAN]];
        assert!(matches!(
            validate_data_generic(&bad.view()),
            Err(ClusterError::NonFinite)
        ));
    }
}
