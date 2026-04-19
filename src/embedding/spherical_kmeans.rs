//! Spherical K-means clustering on the unit hypersphere.
//!
//! Maximizes sum of dot products (cosine similarity) between points and
//! their assigned centroids. Centroids are re-normalized after each update.
//!
//! Reference: Dhillon & Modha (2001), "Concept Decompositions for Large
//! Sparse Text Data using Clustering."

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, WeightedIndex};
use rayon::prelude::*;

use crate::distance::Scalar;
use crate::error::ClusterError;

/// Result of spherical K-means.
pub struct SphericalKMeansState<F: Scalar> {
    pub centroids: Array2<F>,  // (k, d), unit-norm rows
    pub labels: Vec<usize>,
    pub objective: f64,        // sum of dot products (higher = better)
    pub n_iter: usize,
}

// ---- Hot kernels ----

/// Dot product between two slices. Auto-vectorized by LLVM.
#[inline(always)]
pub fn dot_product<F: Scalar>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = F::zero();
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}

/// Find centroid with maximum dot product (spherical assignment).
#[inline]
fn assign_max_dot<F: Scalar>(
    point: &[F],
    centroids: &[F],
    k: usize,
    d: usize,
) -> (usize, F) {
    let mut best_idx = 0;
    let mut best_dot = F::neg_infinity();
    for cluster in 0..k {
        let centroid = &centroids[cluster * d..(cluster + 1) * d];
        let dot = dot_product(point, centroid);
        if dot > best_dot {
            best_dot = dot;
            best_idx = cluster;
        }
    }
    (best_idx, best_dot)
}

// ---- Initialization ----

/// Spherical k-means++ initialization.
/// Weights by (1 - max_dot_to_chosen_center) instead of squared distance.
fn spherical_kmeans_plus_plus<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
    rng: &mut StdRng,
) -> Array2<F> {
    let mut centroids = Array2::<F>::zeros((k, d));

    // Pick first center uniformly at random
    let first = rng.gen_range(0..n);
    let first_row = &data[first * d..(first + 1) * d];
    centroids.row_mut(0).assign(&ndarray::ArrayView1::from(first_row));

    // Distances as (1 - max_dot), in f64 for sampling precision
    let mut min_dists = vec![f64::MAX; n];

    for c in 1..k {
        let last = centroids.row(c - 1);
        let last_slice = last.as_slice().expect("contiguous");

        for i in 0..n {
            let point = &data[i * d..(i + 1) * d];
            let sim = dot_product(point, last_slice).to_f64_lossy();
            let dist = 1.0 - sim; // cosine distance
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        // Clamp negatives (numerical noise)
        for d in min_dists.iter_mut() {
            if *d < 0.0 { *d = 0.0; }
        }

        let total: f64 = min_dists.iter().sum();
        let next = if total <= 0.0 {
            rng.gen_range(0..n)
        } else {
            let dist = WeightedIndex::new(&min_dists).expect("non-negative weights");
            dist.sample(rng)
        };

        let next_row = &data[next * d..(next + 1) * d];
        centroids.row_mut(c).assign(&ndarray::ArrayView1::from(next_row));
    }

    centroids
}

// ---- Main algorithm ----

/// Run spherical K-means with n_init restarts.
pub fn run_spherical_kmeans<F: Scalar>(
    data: &[F],         // unit-normalized, n*d flat
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
) -> Result<SphericalKMeansState<F>, ClusterError> {
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }
    if k == 0 || k > n {
        return Err(ClusterError::InvalidClusters { k, n });
    }

    let mut best: Option<SphericalKMeansState<F>> = None;

    for i in 0..n_init {
        let run_seed = seed.wrapping_add(i as u64);
        let result = run_single(data, n, d, k, max_iter, tol, run_seed)?;

        let is_better = match &best {
            None => true,
            Some(prev) => result.objective > prev.objective, // maximize!
        };

        if is_better {
            best = Some(result);
        }
    }

    Ok(best.unwrap())
}

fn run_single<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<SphericalKMeansState<F>, ClusterError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = spherical_kmeans_plus_plus(data, n, d, k, &mut rng);

    let mut labels = vec![0usize; n];
    let mut objective = f64::NEG_INFINITY;
    let mut n_iter = 0;

    // Pre-allocate f64 accumulation buffers
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];

    for iter in 0..max_iter {
        let centroids_slice = centroids.as_slice().expect("contiguous");

        // Assignment step (parallel): maximize dot product
        let assignments: Vec<(usize, F)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data[i * d..(i + 1) * d];
                assign_max_dot(point, centroids_slice, k, d)
            })
            .collect();

        // Unpack labels and compute objective
        let mut new_objective = 0.0f64;
        for (i, &(label, sim)) in assignments.iter().enumerate() {
            labels[i] = label;
            new_objective += sim.to_f64_lossy();
        }
        objective = new_objective;

        // Centroid update: accumulate in f64, then normalize
        sums.fill(0.0);
        counts.fill(0);

        for i in 0..n {
            let label = labels[i];
            counts[label] += 1;
            let sum_start = label * d;
            let point_start = i * d;
            for j in 0..d {
                sums[sum_start + j] += data[point_start + j].to_f64_lossy();
            }
        }

        // Save old centroids for convergence check
        let old_centroids = centroids.clone();
        let centroid_slice = centroids.as_slice_mut().expect("contiguous");

        for cluster in 0..k {
            let start = cluster * d;
            if counts[cluster] > 0 {
                // Compute norm of the sum vector
                let mut norm_sq = 0.0f64;
                for j in 0..d {
                    norm_sq += sums[start + j] * sums[start + j];
                }

                if norm_sq > 1e-30 {
                    // Re-normalize to unit length
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    for j in 0..d {
                        centroid_slice[start + j] = F::from_f64_lossy(sums[start + j] * inv_norm);
                    }
                } else {
                    // Degenerate: reseed from farthest point
                    let farthest = assignments
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or_else(|| rng.gen_range(0..n));
                    let point_start = farthest * d;
                    for j in 0..d {
                        centroid_slice[start + j] = data[point_start + j];
                    }
                }
            } else {
                // Empty cluster: reseed
                let farthest = assignments
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or_else(|| rng.gen_range(0..n));
                let point_start = farthest * d;
                for j in 0..d {
                    centroid_slice[start + j] = data[point_start + j];
                }
            }
        }

        n_iter = iter + 1;

        // Convergence: max angular shift between old and new centroids
        let old_slice = old_centroids.as_slice().expect("contiguous");
        let new_slice = centroids.as_slice().expect("contiguous");
        let mut max_shift = 0.0f64;
        for cluster in 0..k {
            let start = cluster * d;
            let end = start + d;
            let dot: f64 = old_slice[start..end]
                .iter()
                .zip(&new_slice[start..end])
                .map(|(&a, &b)| a.to_f64_lossy() * b.to_f64_lossy())
                .sum();
            // Angular shift = acos(dot), but we just need to compare against tol
            // Use 1 - dot as a proxy (monotonic with angular shift for dot in [0,1])
            let shift = 1.0 - dot.clamp(-1.0, 1.0);
            if shift > max_shift {
                max_shift = shift;
            }
        }

        if max_shift < tol {
            break;
        }
    }

    Ok(SphericalKMeansState {
        centroids,
        labels,
        objective,
        n_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::normalize::l2_normalize_rows;

    fn make_spherical_data() -> (Vec<f64>, usize, usize) {
        // Two clusters: points near [1,0,0,...] and [-1,0,0,...]
        let d = 8;
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Cluster A: near positive x-axis
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = 5.0;
            for j in 1..d {
                point[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&point);
        }
        // Cluster B: near negative x-axis
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = -5.0;
            for j in 1..d {
                point[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&point);
        }

        let (normalized, _) = l2_normalize_rows(&data, 100, d);
        (normalized, 100, d)
    }

    #[test]
    fn test_dot_product_basic() {
        let a = [1.0f64, 0.0, 0.0];
        let b = [0.0f64, 1.0, 0.0];
        assert!(dot_product(&a, &b).abs() < 1e-10); // orthogonal
        assert!((dot_product(&a, &a) - 1.0).abs() < 1e-10); // self
    }

    #[test]
    fn test_assign_max_dot() {
        // Two unit-norm centroids: [1,0] and [0,1]
        let centroids = [1.0f64, 0.0, 0.0, 1.0];
        let point = [0.9, 0.1]; // closer to first
        let (idx, _) = assign_max_dot(&point, &centroids, 2, 2);
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_spherical_kmeans_well_separated() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_kmeans(&data, n, d, 2, 100, 1e-6, 42, 3).unwrap();

        assert_eq!(result.labels.len(), 100);
        assert_eq!(result.centroids.shape(), &[2, d]);
        assert!(result.objective > 0.0);

        // First 50 should share a label, last 50 another
        let label_a = result.labels[0];
        let label_b = result.labels[50];
        assert_ne!(label_a, label_b);
        for i in 0..50 { assert_eq!(result.labels[i], label_a); }
        for i in 50..100 { assert_eq!(result.labels[i], label_b); }
    }

    #[test]
    fn test_centroids_are_unit_norm() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_kmeans(&data, n, d, 2, 100, 1e-6, 42, 1).unwrap();

        for row in result.centroids.rows() {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Centroid norm = {norm}");
        }
    }

    #[test]
    fn test_reproducibility() {
        let (data, n, d) = make_spherical_data();
        let r1 = run_spherical_kmeans(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();
        let r2 = run_spherical_kmeans(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.objective - r2.objective).abs() < 1e-10);
    }

    #[test]
    fn test_n_init_improves() {
        let (data, n, d) = make_spherical_data();
        let single = run_spherical_kmeans(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();
        let multi = run_spherical_kmeans(&data, n, d, 2, 50, 1e-6, 42, 5).unwrap();
        assert!(multi.objective >= single.objective - 1e-10);
    }

    #[test]
    fn test_single_cluster() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_kmeans(&data, n, d, 1, 50, 1e-6, 42, 1).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_k_equals_n() {
        let d = 4;
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let result = run_spherical_kmeans(&data, 3, d, 3, 50, 1e-6, 42, 1).unwrap();
        let mut sorted = result.labels.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_f32() {
        let d = 8;
        let data_f64 = make_spherical_data().0;
        let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();
        let result = run_spherical_kmeans(&data_f32, 100, d, 2, 50, 1e-6, 42, 1).unwrap();
        assert_eq!(result.labels.len(), 100);
        // Centroids should be unit norm in f32
        for row in result.centroids.rows() {
            let norm: f32 = row.iter().map(|&v| v * v).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "Centroid norm = {norm}");
        }
    }

    #[test]
    fn test_empty_input_fails() {
        let data: Vec<f64> = vec![];
        assert!(matches!(run_spherical_kmeans(&data, 0, 2, 1, 10, 1e-6, 42, 1), Err(ClusterError::EmptyInput)));
    }

    #[test]
    fn test_k_greater_than_n_fails() {
        let data = vec![1.0, 0.0, 0.0, 1.0];
        assert!(matches!(run_spherical_kmeans(&data, 2, 2, 5, 10, 1e-6, 42, 1), Err(ClusterError::InvalidClusters { .. })));
    }
}
