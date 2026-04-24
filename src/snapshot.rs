//! Cluster snapshot: frozen cluster state for incremental assignment.
//!
//! After fitting a clustering model, create a `ClusterSnapshot` to persist
//! the cluster topology and assign new points without re-clustering.
//!
//! Supported algorithms: KMeans, MiniBatchKMeans, EmbeddingCluster.

use std::sync::Arc;

use rayon::prelude::*;

use crate::distance::{
    CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean,
};
use crate::embedding::{normalize, reduction};
use crate::error::ClusterError;
use crate::utils::assign_nearest_two_with;

/// Which algorithm produced this snapshot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SnapshotAlgorithm {
    KMeans,
    MiniBatchKMeans,
    EmbeddingCluster,
}

impl SnapshotAlgorithm {
    pub fn as_str(&self) -> &'static str {
        match self {
            SnapshotAlgorithm::KMeans => "kmeans",
            SnapshotAlgorithm::MiniBatchKMeans => "minibatch_kmeans",
            SnapshotAlgorithm::EmbeddingCluster => "embedding_cluster",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, ClusterError> {
        match s {
            "kmeans" => Ok(SnapshotAlgorithm::KMeans),
            "minibatch_kmeans" => Ok(SnapshotAlgorithm::MiniBatchKMeans),
            "embedding_cluster" => Ok(SnapshotAlgorithm::EmbeddingCluster),
            _ => Err(ClusterError::SnapshotFormat(format!(
                "unknown algorithm: {s}"
            ))),
        }
    }
}

/// Preprocessing pipeline applied to new data before assignment.
#[derive(Debug, Clone)]
pub enum Preprocessing {
    /// No preprocessing — input must match centroid dimensionality.
    None,
    /// L2-normalize rows (EmbeddingCluster without PCA).
    L2Normalize,
    /// Full embedding pipeline: L2-normalize → PCA project → L2-normalize.
    EmbeddingPipeline {
        input_dim: usize,
        pca: reduction::PcaProjection,
    },
}

/// Immutable, Send+Sync cluster snapshot for frozen centroid assignment.
pub struct ClusterSnapshot {
    pub algorithm: SnapshotAlgorithm,
    pub metric: Metric,
    /// If true, assignment maximizes dot product (spherical).
    /// If false, assignment minimizes distance.
    pub spherical: bool,
    /// Flat row-major centroids, shape (k, d). Arc for cheap cloning into threads.
    pub centroids: Arc<Vec<f64>>,
    /// Number of clusters.
    pub k: usize,
    /// Centroid dimensionality (after preprocessing).
    pub d: usize,
    /// Input dimensionality (before preprocessing).
    pub input_dim: usize,
    /// Preprocessing to apply to new data.
    pub preprocessing: Preprocessing,
    /// Per-cluster mean distance at fit time (for drift detection).
    pub fit_mean_distances: Vec<f64>,
    /// Per-cluster sample count from training.
    pub fit_cluster_sizes: Vec<usize>,
    /// Total training samples.
    pub fit_n_samples: usize,
    /// Snapshot format version.
    pub version: u32,
}

// ---- Factory constructors ----

impl ClusterSnapshot {
    /// Create a snapshot from a fitted KMeansState (f64).
    pub fn from_kmeans(
        state: &crate::kmeans::KMeansState<f64>,
        metric: Metric,
    ) -> Self {
        let (k, d) = state.centroids.dim();
        let n_samples = state.labels.len();
        let fit_cluster_sizes = count_labels(&state.labels, k);
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::KMeans,
            metric,
            spherical: false,
            centroids: Arc::clone(&state.centroids_flat),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
        }
    }

    /// Create a snapshot from a fitted KMeansState (f32, converts to f64).
    pub fn from_kmeans_f32(
        state: &crate::kmeans::KMeansState<f32>,
        metric: Metric,
    ) -> Self {
        let (k, d) = state.centroids.dim();
        let n_samples = state.labels.len();
        let fit_cluster_sizes = count_labels(&state.labels, k);
        let centroids_f64: Vec<f64> = state.centroids_flat.iter().map(|&v| v as f64).collect();
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::KMeans,
            metric,
            spherical: false,
            centroids: Arc::new(centroids_f64),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
        }
    }

    /// Create a snapshot from a fitted MiniBatchKMeansState (f64).
    pub fn from_minibatch_kmeans(
        state: &crate::minibatch_kmeans::MiniBatchKMeansState<f64>,
        metric: Metric,
    ) -> Self {
        let (k, d) = state.centroids.dim();
        let n_samples = state.labels.len();
        let fit_cluster_sizes = count_labels(&state.labels, k);
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::MiniBatchKMeans,
            metric,
            spherical: false,
            centroids: Arc::clone(&state.centroids_flat),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
        }
    }

    /// Create a snapshot from a fitted MiniBatchKMeansState (f32).
    pub fn from_minibatch_kmeans_f32(
        state: &crate::minibatch_kmeans::MiniBatchKMeansState<f32>,
        metric: Metric,
    ) -> Self {
        let (k, d) = state.centroids.dim();
        let n_samples = state.labels.len();
        let fit_cluster_sizes = count_labels(&state.labels, k);
        let centroids_f64: Vec<f64> = state.centroids_flat.iter().map(|&v| v as f64).collect();
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::MiniBatchKMeans,
            metric,
            spherical: false,
            centroids: Arc::new(centroids_f64),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
        }
    }

    /// Create a snapshot from a fitted EmbeddingCluster.
    ///
    /// `centroids`: flat unit-norm centroids in reduced space (k * fitted_d).
    /// `pca`: PCA projection if dimensionality reduction was used.
    /// `intra_similarity`: per-cluster mean cosine similarity (used as fit_mean_distances).
    pub fn from_embedding_cluster(
        centroids: &[f64],
        k: usize,
        fitted_d: usize,
        input_dim: usize,
        pca: Option<&crate::embedding::reduction::PcaProjection>,
        labels: &[usize],
        intra_similarity: &[f64],
    ) -> Self {
        let n_samples = labels.len();
        let fit_cluster_sizes = count_labels(labels, k);

        let preprocessing = match pca {
            Some(proj) => Preprocessing::EmbeddingPipeline {
                input_dim,
                pca: proj.clone(),
            },
            None => Preprocessing::L2Normalize,
        };

        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::EmbeddingCluster,
            metric: Metric::Cosine,
            spherical: true,
            centroids: Arc::new(centroids.to_vec()),
            k,
            d: fitted_d,
            input_dim,
            preprocessing,
            fit_mean_distances: intra_similarity.to_vec(),
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
        }
    }
}

/// Count how many labels fall into each cluster [0..k).
fn count_labels(labels: &[usize], k: usize) -> Vec<usize> {
    let mut counts = vec![0usize; k];
    for &l in labels {
        if l < k {
            counts[l] += 1;
        }
    }
    counts
}

// ClusterSnapshot is Send+Sync because:
// - Arc<Vec<f64>> is Send+Sync
// - PcaProjection contains only Vec<f64> and usize (Send+Sync)
// - All other fields are Copy or Vec of Copy types
unsafe impl Send for ClusterSnapshot {}
unsafe impl Sync for ClusterSnapshot {}

/// Result of assigning new points to a snapshot.
#[derive(Debug)]
pub struct AssignmentResult {
    /// Cluster labels (-1 if rejected).
    pub labels: Vec<i64>,
    /// Distance/similarity to nearest centroid.
    pub distances: Vec<f64>,
    /// Distance/similarity to second-nearest centroid.
    pub second_distances: Vec<f64>,
    /// Confidence score in [0, 1). Higher = more decisive assignment.
    pub confidences: Vec<f64>,
    /// Whether each point was rejected.
    pub rejected: Vec<bool>,
}

impl AssignmentResult {
    /// Apply rejection thresholds. Sets labels to -1 for rejected points.
    ///
    /// For standard (min-distance) metrics:
    /// - `distance_threshold`: reject if nearest distance > threshold
    /// - `confidence_threshold`: reject if confidence < threshold
    ///
    /// For spherical (max-dot) metrics:
    /// - `distance_threshold`: reject if best similarity < threshold (i.e., too dissimilar)
    /// - `confidence_threshold`: reject if confidence < threshold
    pub fn apply_rejection(
        &mut self,
        distance_threshold: Option<f64>,
        confidence_threshold: Option<f64>,
        spherical: bool,
    ) {
        for i in 0..self.labels.len() {
            let reject = if spherical {
                // For dot product: higher is better, reject if below threshold
                distance_threshold.map_or(false, |t| self.distances[i] < t)
                    || confidence_threshold.map_or(false, |t| self.confidences[i] < t)
            } else {
                // For distance: lower is better, reject if above threshold
                distance_threshold.map_or(false, |t| self.distances[i] > t)
                    || confidence_threshold.map_or(false, |t| self.confidences[i] < t)
            };
            if reject {
                self.rejected[i] = true;
                self.labels[i] = -1;
            }
        }
    }
}

impl ClusterSnapshot {
    /// Assign a batch of points to the nearest cluster.
    ///
    /// `data`: flat row-major f64, shape (n, input_dim).
    /// Returns assignment result with labels, distances, confidences.
    pub fn assign_batch(&self, data: &[f64], n: usize) -> Result<AssignmentResult, ClusterError> {
        if n == 0 {
            return Ok(AssignmentResult {
                labels: vec![],
                distances: vec![],
                second_distances: vec![],
                confidences: vec![],
                rejected: vec![],
            });
        }

        let expected_len = n * self.input_dim;
        if data.len() != expected_len {
            return Err(ClusterError::DimensionMismatch {
                expected: self.input_dim,
                got: data.len() / n,
            });
        }

        // Preprocess
        let work_data = self.preprocess(data, n)?;
        let work_d = self.d;
        let k = self.k;
        let centroids = &self.centroids[..];

        // Parallel assignment
        let results: Vec<(usize, f64, f64)> = if self.spherical {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &work_data[i * work_d..(i + 1) * work_d];
                    assign_max_dot_two(point, centroids, k, work_d)
                })
                .collect()
        } else {
            match self.metric {
                Metric::Euclidean => self.assign_standard::<SquaredEuclidean>(&work_data, n),
                Metric::Cosine => self.assign_standard::<CosineDistance>(&work_data, n),
                Metric::Manhattan => self.assign_standard::<ManhattanDistance>(&work_data, n),
            }
        };

        // Build result
        let mut labels = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        let mut second_distances = Vec::with_capacity(n);
        let mut confidences = Vec::with_capacity(n);

        for (idx, best, second) in &results {
            labels.push(*idx as i64);
            distances.push(*best);
            second_distances.push(*second);

            let conf = if self.k < 2 {
                // Only one cluster — confidence is meaningless
                0.0
            } else if self.spherical {
                // Dot product: higher is better. best >= second.
                if best.abs() < 1e-30 {
                    0.0
                } else {
                    1.0 - (second / best).clamp(0.0, 1.0)
                }
            } else {
                // Distance: lower is better. best <= second.
                if !second.is_finite() || second.abs() < 1e-30 {
                    0.0
                } else {
                    1.0 - (best / second).clamp(0.0, 1.0)
                }
            };
            confidences.push(conf);
        }

        let rejected = vec![false; n];
        Ok(AssignmentResult {
            labels,
            distances,
            second_distances,
            confidences,
            rejected,
        })
    }

    /// Standard (min-distance) assignment returning (idx, best_dist, second_dist).
    fn assign_standard<D: Distance<f64>>(
        &self,
        work_data: &[f64],
        n: usize,
    ) -> Vec<(usize, f64, f64)> {
        let d = self.d;
        let k = self.k;
        let centroids = &self.centroids[..];

        if k < 2 {
            // Only one cluster — second distance is infinity
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &work_data[i * d..(i + 1) * d];
                    let dist = D::distance(point, &centroids[0..d]).to_f64_lossy();
                    (0, dist, f64::INFINITY)
                })
                .collect()
        } else {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let point = &work_data[i * d..(i + 1) * d];
                    let (idx, best, second) =
                        assign_nearest_two_with::<f64, D>(point, centroids, k, d);
                    (idx, best.to_f64_lossy(), second.to_f64_lossy())
                })
                .collect()
        }
    }

    /// Preprocess data according to the snapshot's preprocessing pipeline.
    fn preprocess(&self, data: &[f64], n: usize) -> Result<Vec<f64>, ClusterError> {
        match &self.preprocessing {
            Preprocessing::None => Ok(data.to_vec()),
            Preprocessing::L2Normalize => {
                let mut buf = data.to_vec();
                normalize::l2_normalize_rows_inplace(&mut buf, n, self.input_dim);
                Ok(buf)
            }
            Preprocessing::EmbeddingPipeline { input_dim, pca } => {
                let mut buf = data.to_vec();
                normalize::l2_normalize_rows_inplace(&mut buf, n, *input_dim);
                let projected = reduction::project_data::<f64>(&buf, n, pca);
                let mut out = projected;
                let out_dim = pca.output_dim;
                normalize::l2_normalize_rows_inplace(&mut out, n, out_dim);
                Ok(out)
            }
        }
    }
}

/// Drift report: how new data compares to the original fit.
#[derive(Debug)]
pub struct DriftReport {
    /// Per-cluster mean distance in the new data.
    pub new_mean_distances: Vec<f64>,
    /// Per-cluster sample count in new data.
    pub new_cluster_sizes: Vec<usize>,
    /// Per-cluster relative drift: (new_mean - fit_mean) / fit_mean.
    /// NaN if fit_mean was 0.
    pub relative_drift: Vec<f64>,
    /// Global mean distance across all new points.
    pub global_mean_distance: f64,
    /// Fraction of points that would be rejected at various distance thresholds.
    pub rejection_rate: f64,
    /// Total points analyzed.
    pub n_samples: usize,
}

impl ClusterSnapshot {
    /// Compute drift statistics for new data against the original fit.
    pub fn drift_report(&self, data: &[f64], n: usize) -> Result<DriftReport, ClusterError> {
        let result = self.assign_batch(data, n)?;
        let k = self.k;

        let mut cluster_dist_sums = vec![0.0f64; k];
        let mut cluster_counts = vec![0usize; k];

        for i in 0..n {
            let label = result.labels[i];
            if label >= 0 && (label as usize) < k {
                let idx = label as usize;
                cluster_dist_sums[idx] += result.distances[i];
                cluster_counts[idx] += 1;
            }
        }

        let new_mean_distances: Vec<f64> = (0..k)
            .map(|c| {
                if cluster_counts[c] > 0 {
                    cluster_dist_sums[c] / cluster_counts[c] as f64
                } else {
                    0.0
                }
            })
            .collect();

        let relative_drift: Vec<f64> = (0..k)
            .map(|c| {
                let fit_mean = self.fit_mean_distances[c];
                let new_mean = new_mean_distances[c];
                if fit_mean.abs() < 1e-30 {
                    if new_mean.abs() < 1e-30 {
                        0.0
                    } else {
                        f64::NAN
                    }
                } else {
                    (new_mean - fit_mean) / fit_mean.abs()
                }
            })
            .collect();

        let total_dist: f64 = result.distances.iter().sum();
        let global_mean_distance = if n > 0 { total_dist / n as f64 } else { 0.0 };

        // Rejection rate: fraction of points with distance > 2 * global mean at fit
        let fit_global_mean = if self.fit_n_samples > 0 {
            self.fit_mean_distances.iter().sum::<f64>() / k as f64
        } else {
            0.0
        };
        let rejection_rate = if fit_global_mean.abs() > 1e-30 && n > 0 {
            let threshold = 2.0 * fit_global_mean;
            let rejected = if self.spherical {
                // For similarity: reject if below threshold
                result.distances.iter().filter(|&&d| d < threshold).count()
            } else {
                // For distance: reject if above threshold
                result.distances.iter().filter(|&&d| d > threshold).count()
            };
            rejected as f64 / n as f64
        } else {
            0.0
        };

        Ok(DriftReport {
            new_mean_distances,
            new_cluster_sizes: cluster_counts,
            relative_drift,
            global_mean_distance,
            rejection_rate,
            n_samples: n,
        })
    }
}

/// Find centroid with maximum dot product, returning best and second-best.
///
/// Used for spherical (cosine) assignment on unit-normalized data.
fn assign_max_dot_two(point: &[f64], centroids: &[f64], k: usize, d: usize) -> (usize, f64, f64) {
    debug_assert!(k >= 1);
    debug_assert_eq!(centroids.len(), k * d);
    debug_assert_eq!(point.len(), d);

    if k == 1 {
        let dot = dot_product(point, &centroids[0..d]);
        return (0, dot, f64::NEG_INFINITY);
    }

    let mut best_idx = 0;
    let mut best_dot = f64::NEG_INFINITY;
    let mut second_dot = f64::NEG_INFINITY;

    for cluster in 0..k {
        let centroid = &centroids[cluster * d..(cluster + 1) * d];
        let dot = dot_product(point, centroid);
        if dot > best_dot {
            second_dot = best_dot;
            best_dot = dot;
            best_idx = cluster;
        } else if dot > second_dot {
            second_dot = dot;
        }
    }

    (best_idx, best_dot, second_dot)
}

/// Dot product between two equal-length slices.
#[inline(always)]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kmeans_snapshot(centroids: Vec<f64>, k: usize, d: usize) -> ClusterSnapshot {
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::KMeans,
            metric: Metric::Euclidean,
            spherical: false,
            centroids: Arc::new(centroids),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::None,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes: vec![50; k],
            fit_n_samples: 100,
            version: 1,
        }
    }

    fn make_spherical_snapshot(centroids: Vec<f64>, k: usize, d: usize) -> ClusterSnapshot {
        ClusterSnapshot {
            algorithm: SnapshotAlgorithm::EmbeddingCluster,
            metric: Metric::Cosine,
            spherical: true,
            centroids: Arc::new(centroids),
            k,
            d,
            input_dim: d,
            preprocessing: Preprocessing::L2Normalize,
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes: vec![50; k],
            fit_n_samples: 100,
            version: 1,
        }
    }

    #[test]
    fn test_kmeans_snapshot_assign() {
        // Two centroids: (0,0) and (10,10)
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);

        // Points near each centroid
        let data = vec![1.0, 1.0, 9.0, 9.0, 0.5, -0.5, 10.5, 9.5];
        let result = snap.assign_batch(&data, 4).unwrap();

        assert_eq!(result.labels[0], result.labels[2]); // both near (0,0)
        assert_eq!(result.labels[1], result.labels[3]); // both near (10,10)
        assert_ne!(result.labels[0], result.labels[1]); // different clusters
    }

    #[test]
    fn test_spherical_snapshot_assign() {
        // Two unit-norm centroids: [1,0] and [0,1]
        let snap = make_spherical_snapshot(vec![1.0, 0.0, 0.0, 1.0], 2, 2);

        // Points that, after L2 normalization, are near each centroid
        let data = vec![5.0, 0.1, 0.1, 5.0];
        let result = snap.assign_batch(&data, 2).unwrap();

        assert_ne!(result.labels[0], result.labels[1]);
        // First point is nearly [1,0] after normalization, should match centroid 0
        assert_eq!(result.labels[0], 0);
        assert_eq!(result.labels[1], 1);
    }

    #[test]
    fn test_confidence_high_for_decisive_assignment() {
        // Two centroids far apart
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 100.0, 100.0], 2, 2);

        // Point very close to first centroid
        let data = vec![0.1, 0.1];
        let result = snap.assign_batch(&data, 1).unwrap();

        assert_eq!(result.labels[0], 0);
        assert!(
            result.confidences[0] > 0.9,
            "confidence={}, expected > 0.9",
            result.confidences[0]
        );
    }

    #[test]
    fn test_confidence_low_for_equidistant() {
        // Two centroids equidistant from midpoint
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 0.0], 2, 2);

        // Point at midpoint (5,0) — equidistant to both
        let data = vec![5.0, 0.0];
        let result = snap.assign_batch(&data, 1).unwrap();

        assert!(
            result.confidences[0] < 0.05,
            "confidence={}, expected ~0",
            result.confidences[0]
        );
    }

    #[test]
    fn test_rejection_by_distance() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);

        let data = vec![1000.0, 1000.0]; // far from all clusters
        let mut result = snap.assign_batch(&data, 1).unwrap();

        // Before rejection
        assert!(!result.rejected[0]);

        // After rejection with tight threshold
        result.apply_rejection(Some(100.0), None, false);
        assert!(result.rejected[0]);
        assert_eq!(result.labels[0], -1);
    }

    #[test]
    fn test_rejection_by_confidence() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 0.0], 2, 2);

        // Equidistant point — low confidence
        let data = vec![5.0, 0.0];
        let mut result = snap.assign_batch(&data, 1).unwrap();

        result.apply_rejection(None, Some(0.5), false);
        assert!(result.rejected[0]);
        assert_eq!(result.labels[0], -1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0], 1, 2);

        // Wrong dimension: 3 features instead of 2
        let data = vec![1.0, 2.0, 3.0];
        let err = snap.assign_batch(&data, 1).unwrap_err();
        assert!(matches!(err, ClusterError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_empty_input() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0], 1, 2);
        let result = snap.assign_batch(&[], 0).unwrap();
        assert!(result.labels.is_empty());
    }

    #[test]
    fn test_single_cluster() {
        let snap = make_kmeans_snapshot(vec![5.0, 5.0], 1, 2);
        let data = vec![1.0, 1.0, 9.0, 9.0];
        let result = snap.assign_batch(&data, 2).unwrap();
        assert_eq!(result.labels, vec![0, 0]);
        assert_eq!(result.confidences, vec![0.0, 0.0]); // no second cluster
    }

    #[test]
    fn test_assign_max_dot_two_basic() {
        // Two unit centroids: [1,0] and [0,1]
        let centroids = vec![1.0, 0.0, 0.0, 1.0];
        let point = [0.9, 0.1];
        let (idx, best, second) = assign_max_dot_two(&point, &centroids, 2, 2);
        assert_eq!(idx, 0);
        assert!(best > second);
    }
}
