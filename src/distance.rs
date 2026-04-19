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
}
