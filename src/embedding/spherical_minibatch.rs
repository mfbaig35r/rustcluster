//! Mini-batch spherical K-means.
//!
//! Processes random subsets per iteration with batch-resultant centroid
//! updates. Faster wall-clock convergence than full-pass at quality cost.
//!
//! Uses unnormalized running resultant r_k as the sufficient statistic.
//! After each batch: blend r_k with batch sums, re-normalize to unit sphere.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::distance::Scalar;
use crate::error::ClusterError;

use super::spherical_kmeans::{
    dot_product, spherical_kmeans_plus_plus, IterationTelemetry, SphericalKMeansState,
};

/// Run mini-batch spherical K-means.
pub fn run_spherical_minibatch<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
    batch_size: usize,
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
    if batch_size == 0 {
        return Err(ClusterError::InvalidBatchSize(0));
    }

    let mut best: Option<SphericalKMeansState<F>> = None;

    for i in 0..n_init {
        let run_seed = seed.wrapping_add(i as u64);
        let result = run_minibatch_single(data, n, d, k, batch_size, max_iter, tol, run_seed)?;
        let is_better = match &best {
            None => true,
            Some(prev) => result.objective > prev.objective,
        };
        if is_better {
            best = Some(result);
        }
    }

    Ok(best.unwrap())
}

fn run_minibatch_single<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
    batch_size: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<SphericalKMeansState<F>, ClusterError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let effective_batch = batch_size.min(n);

    // Initialize centroids via spherical k-means++
    let mut centroids = spherical_kmeans_plus_plus(data, n, d, k, &mut rng);

    // Unnormalized running resultants (sufficient statistic for directional mean)
    let centroid_slice = centroids.as_slice().expect("contiguous");
    let mut resultants = vec![0.0f64; k * d];
    for j in 0..k {
        for dim in 0..d {
            resultants[j * d + dim] = centroid_slice[j * d + dim].to_f64_lossy();
        }
    }
    let mut centroid_counts = vec![1.0f64; k]; // start at 1 to avoid div-by-zero

    let mut n_iter = 0;
    let mut prev_objective = f64::NEG_INFINITY;
    let mut no_improvement = 0usize;
    let mut telemetry = Vec::with_capacity(max_iter.min(200));

    for iter in 0..max_iter {
        let iter_start = std::time::Instant::now();

        // Sample batch indices
        let batch_indices = sample(&mut rng, n, effective_batch);
        let centroids_slice = centroids.as_slice().expect("contiguous");

        // Assign batch points
        let assignments: Vec<(usize, usize, F)> = batch_indices
            .into_iter()
            .map(|idx| {
                let point = &data[idx * d..(idx + 1) * d];
                let mut best_j = 0;
                let mut best_dot = F::neg_infinity();
                for j in 0..k {
                    let centroid = &centroids_slice[j * d..(j + 1) * d];
                    let dot = dot_product(point, centroid);
                    if dot > best_dot {
                        best_dot = dot;
                        best_j = j;
                    }
                }
                (idx, best_j, best_dot)
            })
            .collect();

        // Batch objective
        let batch_objective: f64 = assignments
            .iter()
            .map(|&(_, _, dot)| dot.to_f64_lossy())
            .sum();

        // Streaming centroid update via batch resultants
        // Per-cluster batch sums
        let mut batch_sums = vec![0.0f64; k * d];
        let mut batch_counts = vec![0.0f64; k];

        for &(point_idx, cluster, _) in &assignments {
            batch_counts[cluster] += 1.0;
            for dim in 0..d {
                batch_sums[cluster * d + dim] += data[point_idx * d + dim].to_f64_lossy();
            }
        }

        // Blend resultants with batch sums
        let centroid_slice = centroids.as_slice_mut().expect("contiguous");
        for j in 0..k {
            if batch_counts[j] > 0.0 {
                let rho = batch_counts[j] / (centroid_counts[j] + batch_counts[j]);
                centroid_counts[j] += batch_counts[j];

                for dim in 0..d {
                    resultants[j * d + dim] =
                        (1.0 - rho) * resultants[j * d + dim] + rho * batch_sums[j * d + dim];
                }

                // Re-normalize centroid from resultant
                let mut norm_sq = 0.0f64;
                for dim in 0..d {
                    norm_sq += resultants[j * d + dim] * resultants[j * d + dim];
                }
                if norm_sq > 1e-30 {
                    let inv = 1.0 / norm_sq.sqrt();
                    for dim in 0..d {
                        centroid_slice[j * d + dim] =
                            F::from_f64_lossy(resultants[j * d + dim] * inv);
                    }
                }
            }
        }

        n_iter = iter + 1;

        // Early stopping via relative objective change
        let rel_change = if prev_objective.abs() > 1e-30 {
            (batch_objective - prev_objective).abs() / prev_objective.abs()
        } else {
            f64::MAX
        };

        if iter > 0 && tol > 0.0 && rel_change < tol {
            no_improvement += 1;
        } else {
            no_improvement = 0;
        }
        prev_objective = batch_objective;

        let elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;
        telemetry.push(IterationTelemetry {
            iteration: n_iter,
            objective: batch_objective,
            rel_objective_change: rel_change,
            churn: 0.0, // not tracked for mini-batch
            max_angular_shift: 0.0,
            wall_clock_ms: elapsed_ms,
        });

        if no_improvement >= 10 {
            break;
        }
    }

    // Final full-pass assignment for labels and objective
    let centroids_slice = centroids.as_slice().expect("contiguous");
    let final_assignments: Vec<(usize, F)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let point = &data[i * d..(i + 1) * d];
            let mut best_j = 0;
            let mut best_dot = F::neg_infinity();
            for j in 0..k {
                let centroid = &centroids_slice[j * d..(j + 1) * d];
                let dot = dot_product(point, centroid);
                if dot > best_dot {
                    best_dot = dot;
                    best_j = j;
                }
            }
            (best_j, best_dot)
        })
        .collect();

    let mut labels = vec![0usize; n];
    let mut objective = 0.0f64;
    for (i, &(cluster, dot)) in final_assignments.iter().enumerate() {
        labels[i] = cluster;
        objective += dot.to_f64_lossy();
    }

    Ok(SphericalKMeansState {
        centroids,
        labels,
        objective,
        n_iter,
        telemetry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::normalize::l2_normalize_rows;

    fn make_data() -> (Vec<f64>, usize, usize) {
        let d = 8;
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);
        use rand::Rng;

        for _ in 0..50 {
            let mut p = vec![0.0f64; d];
            p[0] = 5.0;
            for j in 1..d {
                p[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&p);
        }
        for _ in 0..50 {
            let mut p = vec![0.0f64; d];
            p[0] = -5.0;
            for j in 1..d {
                p[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&p);
        }
        let (normalized, _) = l2_normalize_rows(&data, 100, d);
        (normalized, 100, d)
    }

    #[test]
    fn test_minibatch_well_separated() {
        let (data, n, d) = make_data();
        let result = run_spherical_minibatch(&data, n, d, 2, 50, 100, 0.0, 42, 1).unwrap();
        assert_eq!(result.labels.len(), 100);
        let la = result.labels[0];
        let lb = result.labels[50];
        assert_ne!(la, lb);
        for i in 0..50 {
            assert_eq!(result.labels[i], la);
        }
        for i in 50..100 {
            assert_eq!(result.labels[i], lb);
        }
    }

    #[test]
    fn test_minibatch_reproducibility() {
        let (data, n, d) = make_data();
        let r1 = run_spherical_minibatch(&data, n, d, 2, 50, 50, 0.0, 42, 1).unwrap();
        let r2 = run_spherical_minibatch(&data, n, d, 2, 50, 50, 0.0, 42, 1).unwrap();
        assert_eq!(r1.labels, r2.labels);
    }

    #[test]
    fn test_minibatch_centroids_unit_norm() {
        let (data, n, d) = make_data();
        let result = run_spherical_minibatch(&data, n, d, 2, 50, 50, 0.0, 42, 1).unwrap();
        for row in result.centroids.rows() {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-10, "Centroid norm = {norm}");
        }
    }

    #[test]
    fn test_minibatch_invalid_batch_size() {
        let (data, n, d) = make_data();
        assert!(matches!(
            run_spherical_minibatch(&data, n, d, 2, 0, 50, 0.0, 42, 1),
            Err(ClusterError::InvalidBatchSize(_))
        ));
    }

    #[test]
    fn test_minibatch_large_batch() {
        // batch_size > n should work (clamped)
        let (data, n, d) = make_data();
        let result = run_spherical_minibatch(&data, n, d, 2, 10000, 50, 0.0, 42, 1).unwrap();
        assert_eq!(result.labels.len(), 100);
    }
}
