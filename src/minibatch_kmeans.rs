//! Mini-batch K-means clustering.
//!
//! Processes random subsets (mini-batches) of the data per iteration.
//! Much faster than full K-means for large datasets, converges to
//! slightly worse solutions.
//!
//! Reference: Sculley, "Web-Scale K-Means Clustering" (WWW 2010).

use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::seq::index::sample;
use rand::SeedableRng;
use rayon::prelude::*;

use crate::distance::{CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean};
use crate::error::ClusterError;
use crate::kmeans::kmeans_plus_plus_init;
use crate::utils::{assign_nearest_with, validate_data_generic};

/// Result of a fitted Mini-batch K-means model.
pub struct MiniBatchKMeansState<F: Scalar> {
    pub centroids: Array2<F>,
    pub labels: Vec<usize>,
    pub inertia: f64,
    pub n_iter: usize,
}

// ---- Public entry points ----

/// Run Mini-batch K-means with runtime metric selection (f64).
pub fn run_minibatch_kmeans_with_metric(
    data: &ArrayView2<f64>,
    k: usize,
    batch_size: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    max_no_improvement: usize,
    metric: Metric,
) -> Result<MiniBatchKMeansState<f64>, ClusterError> {
    match metric {
        Metric::Euclidean => run_minibatch_generic::<f64, SquaredEuclidean>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
        Metric::Cosine => run_minibatch_generic::<f64, CosineDistance>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
        Metric::Manhattan => run_minibatch_generic::<f64, ManhattanDistance>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
    }
}

/// Run Mini-batch K-means with runtime metric selection (f32).
pub fn run_minibatch_kmeans_with_metric_f32(
    data: &ArrayView2<f32>,
    k: usize,
    batch_size: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    max_no_improvement: usize,
    metric: Metric,
) -> Result<MiniBatchKMeansState<f32>, ClusterError> {
    match metric {
        Metric::Euclidean => run_minibatch_generic::<f32, SquaredEuclidean>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
        Metric::Cosine => run_minibatch_generic::<f32, CosineDistance>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
        Metric::Manhattan => run_minibatch_generic::<f32, ManhattanDistance>(
            data, k, batch_size, max_iter, tol, seed, max_no_improvement,
        ),
    }
}

// ---- Generic implementation ----

fn run_minibatch_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    k: usize,
    batch_size: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    max_no_improvement: usize,
) -> Result<MiniBatchKMeansState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();
    if k == 0 || k > n {
        return Err(ClusterError::InvalidClusters { k, n });
    }
    if batch_size == 0 {
        return Err(ClusterError::InvalidBatchSize(0));
    }
    if max_iter == 0 {
        return Err(ClusterError::InvalidMaxIter(0));
    }
    if max_no_improvement == 0 {
        return Err(ClusterError::InvalidMaxNoImprovement(0));
    }

    let effective_batch = batch_size.min(n);
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize centroids via kmeans++
    let mut centroids = kmeans_plus_plus_init::<F, D>(data, k, &mut rng);

    // Per-centroid update counts for streaming mean
    let mut centroid_counts = vec![0usize; k];

    // Early stopping: track EWA of inertia
    let mut ewa_inertia: Option<f64> = None;
    let mut no_improvement_count = 0usize;
    let ewa_alpha = 0.1; // smoothing factor for exponentially weighted average

    let mut n_iter = 0;

    for iter in 0..max_iter {
        // Sample batch indices
        let batch_indices = sample(&mut rng, n, effective_batch);
        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

        // Assign batch points to nearest centroid
        let assignments: Vec<(usize, usize, F)> = batch_indices
            .into_iter()
            .map(|idx| {
                let point = &data_slice[idx * d..(idx + 1) * d];
                let (cluster, dist) = assign_nearest_with::<F, D>(point, centroids_slice, k, d);
                (idx, cluster, dist)
            })
            .collect();

        // Compute batch inertia for early stopping
        let batch_inertia: f64 = assignments.iter().map(|&(_, _, dist)| dist.to_f64_lossy()).sum();
        let batch_inertia_mean = batch_inertia / effective_batch as f64;

        // Streaming centroid update
        let centroid_slice = centroids.as_slice_mut().expect("centroids are C-contiguous");
        for &(point_idx, cluster, _) in &assignments {
            centroid_counts[cluster] += 1;
            let lr = 1.0 / centroid_counts[cluster] as f64;
            let lr_f = F::from_f64_lossy(lr);
            let one_minus_lr = F::from_f64_lossy(1.0 - lr);
            let c_start = cluster * d;
            let p_start = point_idx * d;
            for j in 0..d {
                centroid_slice[c_start + j] =
                    one_minus_lr * centroid_slice[c_start + j] + lr_f * data_slice[p_start + j];
            }
        }

        n_iter = iter + 1;

        // Early stopping via EWA inertia
        match ewa_inertia {
            None => {
                ewa_inertia = Some(batch_inertia_mean);
            }
            Some(prev) => {
                let new_ewa = ewa_alpha * batch_inertia_mean + (1.0 - ewa_alpha) * prev;
                if tol > 0.0 && (prev - new_ewa).abs() / prev.max(1e-15) < tol {
                    no_improvement_count += 1;
                } else {
                    no_improvement_count = 0;
                }
                ewa_inertia = Some(new_ewa);

                if no_improvement_count >= max_no_improvement {
                    break;
                }
            }
        }
    }

    // Final pass: assign ALL points to nearest centroid (parallel)
    let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");
    let final_assignments: Vec<(usize, F)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let point = &data_slice[i * d..(i + 1) * d];
            assign_nearest_with::<F, D>(point, centroids_slice, k, d)
        })
        .collect();

    let mut labels = vec![0usize; n];
    let mut inertia = 0.0f64;
    for (i, &(cluster, dist)) in final_assignments.iter().enumerate() {
        labels[i] = cluster;
        inertia += dist.to_f64_lossy();
    }

    Ok(MiniBatchKMeansState {
        centroids,
        labels,
        inertia,
        n_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn well_separated_data() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, -0.1],
            [-0.1, 0.2],
            [100.0, 100.0],
            [100.1, 100.1],
            [100.2, 99.9],
            [99.9, 100.2],
        ]
    }

    #[test]
    fn test_minibatch_converges_on_separated_clusters() {
        let data = well_separated_data();
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 4, 100, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();

        assert_eq!(result.labels.len(), 8);
        let label_a = result.labels[0];
        let label_b = result.labels[4];
        assert_ne!(label_a, label_b);
        for i in 0..4 { assert_eq!(result.labels[i], label_a); }
        for i in 4..8 { assert_eq!(result.labels[i], label_b); }
    }

    #[test]
    fn test_minibatch_reproducibility() {
        let data = well_separated_data();
        let r1 = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 4, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        let r2 = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 4, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.inertia - r2.inertia).abs() < 1e-10);
    }

    #[test]
    fn test_minibatch_single_cluster() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 1, 3, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_minibatch_batch_size_larger_than_n() {
        let data = well_separated_data();
        // batch_size=100 > n=8, should clamp and work
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 100, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        assert_eq!(result.labels.len(), 8);
    }

    #[test]
    fn test_minibatch_k_greater_than_n_fails() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 5, 2, 50, 0.0, 42, 10, Metric::Euclidean,
        );
        assert!(matches!(result, Err(ClusterError::InvalidClusters { .. })));
    }

    #[test]
    fn test_minibatch_empty_input_fails() {
        let data = Array2::<f64>::zeros((0, 2));
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 1, 1, 50, 0.0, 42, 10, Metric::Euclidean,
        );
        assert!(matches!(result, Err(ClusterError::EmptyInput)));
    }

    #[test]
    fn test_minibatch_invalid_batch_size() {
        let data = well_separated_data();
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 0, 50, 0.0, 42, 10, Metric::Euclidean,
        );
        assert!(matches!(result, Err(ClusterError::InvalidBatchSize(_))));
    }

    #[test]
    fn test_minibatch_cosine() {
        let data = array![
            [1.0, 0.0], [0.9, 0.1], [0.8, 0.2],
            [0.0, 1.0], [0.1, 0.9], [0.2, 0.8],
        ];
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 4, 50, 0.0, 42, 10, Metric::Cosine,
        ).unwrap();
        assert_eq!(result.labels.len(), 6);
    }

    #[test]
    fn test_minibatch_f32() {
        let data = array![
            [0.0f32, 0.0], [0.1, 0.1],
            [100.0, 100.0], [100.1, 100.1],
        ];
        let result = run_minibatch_kmeans_with_metric_f32(
            &data.view(), 2, 4, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        assert_eq!(result.labels.len(), 4);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_minibatch_early_stopping() {
        let data = well_separated_data();
        // With tol > 0, should stop before max_iter
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 2, 8, 1000, 1e-4, 42, 3, Metric::Euclidean,
        ).unwrap();
        assert!(result.n_iter < 1000);
    }

    #[test]
    fn test_minibatch_identical_points() {
        let data = array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
        let result = run_minibatch_kmeans_with_metric(
            &data.view(), 1, 4, 50, 0.0, 42, 10, Metric::Euclidean,
        ).unwrap();
        assert!(result.inertia < 1e-10);
    }
}
