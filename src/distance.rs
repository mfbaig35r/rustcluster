/// Trait for computing distance between two equal-length slices.
///
/// Uses static dispatch via `D::distance(a, b)` — no vtable overhead.
/// Monomorphization produces identical code to calling the kernel directly.
pub trait Distance: Send + Sync + Clone + Copy + 'static {
    /// Compute the distance between two equal-length slices.
    fn distance(a: &[f64], b: &[f64]) -> f64;
}

/// Squared Euclidean distance: sum of (a_i - b_i)^2.
///
/// Note: Hamerly's bound updates assume this returns a value where `.sqrt()`
/// gives the actual metric distance. Non-Euclidean metrics would need a
/// different accelerated algorithm or fall back to Lloyd.
#[derive(Debug, Clone, Copy)]
pub struct SquaredEuclidean;

impl Distance for SquaredEuclidean {
    #[inline(always)]
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        crate::utils::squared_euclidean(a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_euclidean_via_trait() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert!((SquaredEuclidean::distance(&a, &b) - 27.0).abs() < 1e-10);
    }
}
