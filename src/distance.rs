#[allow(unused_imports)]
use num_traits::Float;

/// Marker trait for supported floating-point types (f32, f64).
pub trait Scalar:
    Float + Send + Sync + std::iter::Sum + std::fmt::Debug + 'static
{
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
/// Monomorphization produces separate optimized code for each (F, D) pair.
pub trait Distance<F: Scalar>: Send + Sync + Clone + Copy + 'static {
    fn distance(a: &[F], b: &[F]) -> F;
}

/// Squared Euclidean distance: sum of (a_i - b_i)^2.
///
/// Note: Hamerly's bound updates assume this returns a value where `.sqrt()`
/// gives the actual metric distance. Non-Euclidean metrics would need a
/// different accelerated algorithm or fall back to Lloyd.
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclidean;

impl<F: Scalar> Distance<F> for SquaredEuclidean {
    #[inline(always)]
    fn distance(a: &[F], b: &[F]) -> F {
        crate::utils::squared_euclidean_generic(a, b)
    }
}

/// Cosine distance: 1 - cosine_similarity(a, b).
///
/// Range: [0, 2]. 0 = identical direction, 1 = orthogonal, 2 = opposite.
/// For zero-norm vectors, returns 1.0 (treated as orthogonal).
///
/// Note: Hamerly's algorithm is NOT compatible with cosine distance
/// (its bound updates assume Euclidean metric). K-means with cosine
/// distance must use Lloyd's algorithm.
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
            F::one() // zero vectors are treated as orthogonal
        } else {
            F::one() - dot / denom
        }
    }
}

/// Metric identifier for runtime dispatch at the Python boundary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Euclidean,
    Cosine,
}

impl Metric {
    pub fn from_str(s: &str) -> Result<Self, crate::error::ClusterError> {
        match s.to_lowercase().as_str() {
            "euclidean" => Ok(Metric::Euclidean),
            "cosine" => Ok(Metric::Cosine),
            _ => Err(crate::error::ClusterError::InvalidMetric(s.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_f64_via_trait() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        assert!((SquaredEuclidean::distance(&a, &b) - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean_f32_via_trait() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        assert!((SquaredEuclidean::distance(&a, &b) - 27.0).abs() < 1e-4);
    }

    #[test]
    fn test_cosine_identical() {
        let a = [1.0f64, 2.0, 3.0];
        assert!(CosineDistance::distance(&a, &a).abs() < 1e-10); // distance = 0
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = [1.0f64, 0.0];
        let b = [0.0f64, 1.0];
        assert!((CosineDistance::distance(&a, &b) - 1.0).abs() < 1e-10); // distance = 1
    }

    #[test]
    fn test_cosine_opposite() {
        let a = [1.0f64, 0.0];
        let b = [-1.0f64, 0.0];
        assert!((CosineDistance::distance(&a, &b) - 2.0).abs() < 1e-10); // distance = 2
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = [0.0f64, 0.0];
        let b = [1.0f64, 2.0];
        assert!((CosineDistance::distance(&a, &b) - 1.0).abs() < 1e-10); // treated as orthogonal
    }

    #[test]
    fn test_cosine_f32() {
        let a = [1.0f32, 0.0];
        let b = [0.0f32, 1.0];
        assert!((CosineDistance::distance(&a, &b) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_metric_from_str() {
        assert_eq!(Metric::from_str("euclidean").unwrap(), Metric::Euclidean);
        assert_eq!(Metric::from_str("cosine").unwrap(), Metric::Cosine);
        assert!(Metric::from_str("manhattan").is_err());
    }
}
