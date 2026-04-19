//! DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
//!
//! Two-phase algorithm:
//! 1. Parallel neighborhood scan — find all neighbors within eps for each point
//! 2. Sequential BFS cluster expansion — grow clusters through core points
//!
//! Complexity: O(n^2) for the naive neighbor scan. Spatial indexing (KD-tree)
//! would improve to O(n log n) average-case for low dimensions — future work.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::collections::VecDeque;

use crate::distance::{Distance, Scalar, SquaredEuclidean};
use crate::error::ClusterError;
use crate::utils::validate_data_generic;

/// Result of a fitted DBSCAN model.
pub struct DbscanState<F: Scalar> {
    /// Cluster labels: -1 for noise, >= 0 for cluster ID.
    pub labels: Vec<i64>,
    /// Indices of core points in the input data.
    pub core_sample_indices: Vec<usize>,
    /// Coordinates of core points (n_core x d).
    pub components: Array2<F>,
    /// Number of clusters found (excluding noise).
    pub n_clusters: usize,
}

// ---- Public entry points ----

/// Run DBSCAN with f64 data and Euclidean distance.
pub fn run_dbscan(
    data: &ArrayView2<f64>,
    eps: f64,
    min_samples: usize,
) -> Result<DbscanState<f64>, ClusterError> {
    run_dbscan_generic::<f64, SquaredEuclidean>(data, eps, min_samples)
}

/// Run DBSCAN with f32 data and Euclidean distance.
pub fn run_dbscan_f32(
    data: &ArrayView2<f32>,
    eps: f64,
    min_samples: usize,
) -> Result<DbscanState<f32>, ClusterError> {
    run_dbscan_generic::<f32, SquaredEuclidean>(data, eps, min_samples)
}

// ---- Generic implementation ----

fn run_dbscan_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    eps: f64,
    min_samples: usize,
) -> Result<DbscanState<F>, ClusterError> {
    // Validate inputs
    validate_data_generic(data)?;
    if eps <= 0.0 || !eps.is_finite() {
        return Err(ClusterError::InvalidEps(eps));
    }
    if min_samples == 0 {
        return Err(ClusterError::InvalidMinSamples(0));
    }

    let (n, d) = data.dim();
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    // Compare squared distances against eps^2 to avoid sqrt in the hot path
    let eps_sq = F::from_f64_lossy(eps * eps);

    // ---- Phase 1: Parallel neighborhood scan ----
    let neighbors: Vec<Vec<usize>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let point_i = &data_slice[i * d..(i + 1) * d];
            let mut nbrs = Vec::new();
            for j in 0..n {
                let point_j = &data_slice[j * d..(j + 1) * d];
                if D::distance(point_i, point_j) <= eps_sq {
                    nbrs.push(j);
                }
            }
            nbrs
        })
        .collect();

    // Identify core points
    let is_core: Vec<bool> = neighbors
        .iter()
        .map(|nbrs| nbrs.len() >= min_samples)
        .collect();

    // ---- Phase 2: Sequential BFS cluster expansion ----
    let mut labels = vec![-1i64; n];
    let mut cluster_id = 0i64;

    for i in 0..n {
        // Skip already labeled or non-core points
        if labels[i] != -1 || !is_core[i] {
            continue;
        }

        // Start a new cluster from this core point
        labels[i] = cluster_id;
        let mut queue = VecDeque::new();
        queue.push_back(i);

        while let Some(p) = queue.pop_front() {
            for &q in &neighbors[p] {
                if labels[q] == -1 {
                    // Label this point (border or core)
                    labels[q] = cluster_id;

                    // Only expand through core points
                    if is_core[q] {
                        queue.push_back(q);
                    }
                }
            }
        }

        cluster_id += 1;
    }

    // Collect core sample indices and components
    let core_sample_indices: Vec<usize> = is_core
        .iter()
        .enumerate()
        .filter(|(_, &c)| c)
        .map(|(i, _)| i)
        .collect();

    let n_core = core_sample_indices.len();
    let components = if n_core > 0 {
        let mut comp = Array2::<F>::zeros((n_core, d));
        for (row, &idx) in core_sample_indices.iter().enumerate() {
            let point = &data_slice[idx * d..(idx + 1) * d];
            comp.row_mut(row).assign(&ndarray::ArrayView1::from(point));
        }
        comp
    } else {
        Array2::<F>::zeros((0, d))
    };

    Ok(DbscanState {
        labels,
        core_sample_indices,
        components,
        n_clusters: cluster_id as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_two_clusters() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let result = run_dbscan(&data.view(), 0.5, 2).unwrap();
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);

        // First 3 share a cluster, last 3 share another
        let c1 = result.labels[0];
        let c2 = result.labels[3];
        assert_ne!(c1, c2);
        assert!(c1 >= 0 && c2 >= 0);
        assert_eq!(result.labels[1], c1);
        assert_eq!(result.labels[2], c1);
        assert_eq!(result.labels[4], c2);
        assert_eq!(result.labels[5], c2);
    }

    #[test]
    fn test_noise_detection() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [50.0, 50.0], // outlier
        ];
        let result = run_dbscan(&data.view(), 0.5, 2).unwrap();
        // Outlier should be noise
        assert_eq!(result.labels[3], -1);
        // First 3 should be in a cluster
        assert!(result.labels[0] >= 0);
    }

    #[test]
    fn test_all_noise() {
        let data = array![
            [0.0, 0.0],
            [10.0, 10.0],
            [20.0, 20.0],
        ];
        let result = run_dbscan(&data.view(), 0.1, 2).unwrap();
        assert!(result.labels.iter().all(|&l| l == -1));
        assert_eq!(result.n_clusters, 0);
        assert!(result.core_sample_indices.is_empty());
    }

    #[test]
    fn test_all_one_cluster() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [0.3, 0.0],
        ];
        let result = run_dbscan(&data.view(), 1.0, 2).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_min_samples_one() {
        // Every point is a core point when min_samples=1
        let data = array![
            [0.0, 0.0],
            [100.0, 100.0],
        ];
        let result = run_dbscan(&data.view(), 0.1, 1).unwrap();
        // Each point is its own cluster (too far for eps=0.1)
        assert_eq!(result.n_clusters, 2);
        assert!(result.labels.iter().all(|&l| l >= 0));
        assert_eq!(result.core_sample_indices.len(), 2);
    }

    #[test]
    fn test_chain() {
        // Points in a chain, each within eps of the next
        let data = array![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],
            [2.0, 0.0],
        ];
        let result = run_dbscan(&data.view(), 0.6, 2).unwrap();
        // All should be connected in one cluster
        assert_eq!(result.n_clusters, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_core_sample_indices() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [50.0, 50.0], // outlier, not core
        ];
        let result = run_dbscan(&data.view(), 0.5, 2).unwrap();
        // First 3 are core (each has >= 2 neighbors within eps)
        assert!(result.core_sample_indices.contains(&0));
        assert!(result.core_sample_indices.contains(&1));
        assert!(result.core_sample_indices.contains(&2));
        assert!(!result.core_sample_indices.contains(&3));
    }

    #[test]
    fn test_components_shape() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
        ];
        let result = run_dbscan(&data.view(), 0.5, 2).unwrap();
        let n_core = result.core_sample_indices.len();
        assert_eq!(result.components.shape(), &[n_core, 2]);
    }

    #[test]
    fn test_identical_points() {
        let data = array![
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0],
        ];
        let result = run_dbscan(&data.view(), 0.1, 2).unwrap();
        assert_eq!(result.n_clusters, 1);
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_single_point() {
        let data = array![[1.0, 2.0]];
        // min_samples=2, so single point can't be core → noise
        let result = run_dbscan(&data.view(), 1.0, 2).unwrap();
        assert_eq!(result.labels[0], -1);
        assert_eq!(result.n_clusters, 0);
    }

    #[test]
    fn test_invalid_eps() {
        let data = array![[1.0, 2.0]];
        assert!(matches!(run_dbscan(&data.view(), 0.0, 2), Err(ClusterError::InvalidEps(_))));
        assert!(matches!(run_dbscan(&data.view(), -1.0, 2), Err(ClusterError::InvalidEps(_))));
    }

    #[test]
    fn test_invalid_min_samples() {
        let data = array![[1.0, 2.0]];
        assert!(matches!(run_dbscan(&data.view(), 1.0, 0), Err(ClusterError::InvalidMinSamples(_))));
    }

    #[test]
    fn test_empty_input() {
        let data = Array2::<f64>::zeros((0, 2));
        assert!(matches!(run_dbscan(&data.view(), 1.0, 2), Err(ClusterError::EmptyInput)));
    }

    #[test]
    fn test_nan_input() {
        let data = array![[1.0, f64::NAN], [2.0, 3.0]];
        assert!(matches!(run_dbscan(&data.view(), 1.0, 2), Err(ClusterError::NonFinite)));
    }

    // ---- f32 tests ----

    #[test]
    fn test_f32_two_clusters() {
        let data = array![
            [0.0f32, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let result = run_dbscan_f32(&data.view(), 0.5, 2).unwrap();
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels.len(), 6);
    }

    #[test]
    fn test_f32_matches_f64() {
        let data_f64 = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let data_f32 = array![
            [0.0f32, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [10.0, 10.0],
            [10.1, 10.1],
            [10.2, 10.0],
        ];
        let r64 = run_dbscan(&data_f64.view(), 0.5, 2).unwrap();
        let r32 = run_dbscan_f32(&data_f32.view(), 0.5, 2).unwrap();
        assert_eq!(r64.labels, r32.labels);
    }
}
