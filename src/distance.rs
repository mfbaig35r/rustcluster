#[allow(unused_imports)]
use num_traits::Float;

/// Marker trait for supported floating-point types (f32, f64).
pub trait Scalar: Float + Send + Sync + std::iter::Sum + std::fmt::Debug + 'static {
    fn to_f64_lossy(self) -> f64;
    fn from_f64_lossy(v: f64) -> Self;
}

impl Scalar for f64 {
    #[inline(always)]
    fn to_f64_lossy(self) -> f64 {
        self
    }
    #[inline(always)]
    fn from_f64_lossy(v: f64) -> Self {
        v
    }
}

impl Scalar for f32 {
    #[inline(always)]
    fn to_f64_lossy(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn from_f64_lossy(v: f64) -> Self {
        v as f32
    }
}

/// Trait for computing distance between two equal-length slices.
///
/// Uses static dispatch via `D::distance(a, b)` — no vtable overhead.
pub trait Distance<F: Scalar>: Send + Sync + Clone + Copy + 'static {
    /// Compute the raw distance value between two slices.
    fn distance(a: &[F], b: &[F]) -> F;

    /// Convert a raw distance value to the actual metric distance.
    ///
    /// For SquaredEuclidean, this is `.sqrt()`. For all other metrics,
    /// the raw value IS the metric distance (identity).
    ///
    /// Used by HDBSCAN for core distances and mutual reachability.
    #[inline(always)]
    fn to_metric(raw: f64) -> f64 {
        raw // default: identity
    }
}

/// Squared Euclidean distance: sum of (a_i - b_i)^2.
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclidean;

impl<F: Scalar> Distance<F> for SquaredEuclidean {
    #[inline(always)]
    fn distance(a: &[F], b: &[F]) -> F {
        crate::utils::squared_euclidean_generic(a, b)
    }

    #[inline(always)]
    fn to_metric(raw: f64) -> f64 {
        raw.sqrt()
    }
}

/// Cosine distance: 1 - cosine_similarity(a, b).
///
/// Range: [0, 2]. 0 = identical direction, 1 = orthogonal, 2 = opposite.
#[derive(Debug, Clone, Copy)]
pub struct CosineDistance;

impl<F: Scalar> Distance<F> for CosineDistance {
    #[inline(always)]
    fn distance(a: &[F], b: &[F]) -> F {
        debug_assert_eq!(a.len(), b.len());
        let mut dot = F::zero();
        let mut norm_a = F::zero();
        let mut norm_b = F::zero();
        for i in 0..a.len() {
            dot = dot + a[i] * b[i];
            norm_a = norm_a + a[i] * a[i];
            norm_b = norm_b + b[i] * b[i];
        }
        let denom = (norm_a * norm_b).sqrt();
        if denom <= F::zero() {
            F::one()
        } else {
            F::one() - dot / denom
        }
    }
}

/// Manhattan distance (L1): sum of |a_i - b_i|.
#[derive(Debug, Clone, Copy)]
pub struct ManhattanDistance;

impl<F: Scalar> Distance<F> for ManhattanDistance {
    #[inline(always)]
    fn distance(a: &[F], b: &[F]) -> F {
        debug_assert_eq!(a.len(), b.len());
        let mut acc = F::zero();
        for i in 0..a.len() {
            acc = acc + (a[i] - b[i]).abs();
        }
        acc
    }
}

/// Metric identifier for runtime dispatch at the Python boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Euclidean,
    Cosine,
    Manhattan,
}

impl std::str::FromStr for Metric {
    type Err = crate::error::ClusterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "euclidean" | "l2" => Ok(Metric::Euclidean),
            "cosine" => Ok(Metric::Cosine),
            "manhattan" | "cityblock" | "l1" => Ok(Metric::Manhattan),
            _ => Err(crate::error::ClusterError::InvalidMetric(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_f64() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        assert!((SquaredEuclidean::distance(&a, &b) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_f32() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((SquaredEuclidean::distance(&a, &b) - 27.0).abs() < 1e-4);
    }

    #[test]
    fn test_squared_euclidean_to_metric() {
        assert!((<SquaredEuclidean as Distance<f64>>::to_metric(9.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_identical() {
        let a = [1.0f64, 2.0, 3.0];
        assert!(CosineDistance::distance(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0f64, 0.0];
        let b = [0.0f64, 1.0];
        assert!((CosineDistance::distance(&a, &b) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_to_metric_identity() {
        assert!((<CosineDistance as Distance<f64>>::to_metric(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_basic() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 6.0, 0.0];
        // |3| + |4| + |3| = 10
        assert!((ManhattanDistance::distance(&a, &b) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_identical() {
        let a = [1.0f64, 2.0, 3.0];
        assert!(ManhattanDistance::distance(&a, &a).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan_f32() {
        let a = [1.0f32, 2.0];
        let b = [4.0f32, 6.0];
        assert!((ManhattanDistance::distance(&a, &b) - 7.0).abs() < 1e-4);
    }

    #[test]
    fn test_manhattan_to_metric_identity() {
        assert!((<ManhattanDistance as Distance<f64>>::to_metric(5.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_metric_from_str() {
        assert_eq!("euclidean".parse::<Metric>().unwrap(), Metric::Euclidean);
        assert_eq!("cosine".parse::<Metric>().unwrap(), Metric::Cosine);
        assert_eq!("manhattan".parse::<Metric>().unwrap(), Metric::Manhattan);
        assert_eq!("cityblock".parse::<Metric>().unwrap(), Metric::Manhattan);
        assert_eq!("l1".parse::<Metric>().unwrap(), Metric::Manhattan);
        assert!("hamming".parse::<Metric>().is_err());
    }
}
