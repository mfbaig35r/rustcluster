//! Cluster snapshot: frozen cluster state for incremental assignment.
//!
//! After fitting a clustering model, create a `ClusterSnapshot` to persist
//! the cluster topology and assign new points without re-clustering.
//!
//! Supported algorithms: KMeans, MiniBatchKMeans, EmbeddingCluster.
//!
//! Implementation is split across submodules for readability:
//! - `assign`: `AssignmentResult` + `assign_batch` (the main path).
//! - `drift`: `DriftReport` + `drift_report`.
//! - `calibrate`: `calibrate` (computes confidence quantiles, kappa, variances, distance stats).
//! - `mahalanobis`: `assign_batch_mahalanobis` (calibrated diagonal Mahalanobis assignment).
//!
//! Public surface is unchanged; everything re-exports through this module.

use std::sync::Arc;

use crate::distance::Metric;
use crate::embedding::reduction;
use crate::error::ClusterError;

pub mod assign;
pub mod calibrate;
pub mod drift;
pub mod mahalanobis;

pub use assign::AssignmentResult;
pub use drift::DriftReport;

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
    /// No preprocessing; input must match centroid dimensionality.
    None,
    /// L2-normalize rows (EmbeddingCluster without PCA).
    L2Normalize,
    /// Full embedding pipeline: L2-normalize, PCA project, L2-normalize.
    EmbeddingPipeline {
        input_dim: usize,
        pca: reduction::PcaProjection,
    },
}

/// Per-cluster confidence distribution from calibration.
#[derive(Debug, Clone)]
pub struct ClusterConfidenceStats {
    /// 5th percentile of confidence per cluster.
    pub p5: Vec<f64>,
    /// 10th percentile of confidence per cluster.
    pub p10: Vec<f64>,
    /// 25th percentile of confidence per cluster.
    pub p25: Vec<f64>,
    /// 50th percentile (median) of confidence per cluster.
    pub p50: Vec<f64>,
}

/// Per-cluster per-dimension diagonal variance.
#[derive(Debug, Clone)]
pub struct ClusterVariances {
    /// Flat row-major (k * d): per-cluster per-dimension variance.
    pub variances: Vec<f64>,
}

/// Per-cluster mean and std of the fit-time assignment distance / similarity.
///
/// Populated by `calibrate()`. For non-spherical algorithms the values are
/// Euclidean (or Manhattan) distances; for spherical algorithms they are
/// cosine similarities. Used by `drift_report` to compute `rejection_rate`.
#[derive(Debug, Clone)]
pub struct ClusterDistanceStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

/// Cluster snapshot for frozen centroid assignment.
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
    /// Per-cluster mean assignment quantity at fit time (k entries).
    ///
    /// For non-spherical algorithms (KMeans, MiniBatchKMeans): Euclidean (or
    /// Manhattan) distance. For spherical algorithms (EmbeddingCluster):
    /// cosine similarity. Interpret with the `spherical` field.
    pub fit_mean_distances: Vec<f64>,
    /// Per-cluster sample count from training.
    pub fit_cluster_sizes: Vec<usize>,
    /// Total training samples.
    pub fit_n_samples: usize,
    /// Snapshot format version.
    pub version: u32,

    // ---- v2 calibration fields (None for uncalibrated / v1 snapshots) ----
    /// Per-cluster confidence quantiles from calibrate().
    pub confidence_stats: Option<ClusterConfidenceStats>,
    /// Per-cluster per-dimension variance from calibrate().
    pub cluster_variances: Option<ClusterVariances>,
    /// Per-cluster vMF concentration parameter (spherical only).
    pub fit_kappa: Option<Vec<f64>>,
    /// Per-cluster mean resultant length from fit time (spherical only).
    pub fit_resultant_lengths: Option<Vec<f64>>,
    /// Per-cluster mean+std of fit-time assignment distance (from calibrate()).
    /// Used by `drift_report` for `rejection_rate`. NaN when absent.
    pub fit_distance_stats: Option<ClusterDistanceStats>,
}

// ---- Factory constructors ----

impl ClusterSnapshot {
    /// Create a snapshot from a fitted KMeansState (f64).
    pub fn from_kmeans(state: &crate::kmeans::KMeansState<f64>, metric: Metric) -> Self {
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
            fit_mean_distances: state.fit_mean_distances.clone(),
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
        }
    }

    /// Create a snapshot from a fitted KMeansState (f32, converts to f64).
    pub fn from_kmeans_f32(state: &crate::kmeans::KMeansState<f32>, metric: Metric) -> Self {
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
            fit_mean_distances: state.fit_mean_distances.clone(),
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
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
            fit_mean_distances: state.fit_mean_distances.clone(),
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
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
            fit_mean_distances: state.fit_mean_distances.clone(),
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
        }
    }

    /// Create a snapshot from a fitted EmbeddingCluster.
    ///
    /// `centroids`: flat unit-norm centroids in reduced space (k * fitted_d).
    /// `pca`: PCA projection if dimensionality reduction was used.
    /// `intra_similarity`: per-cluster mean cosine similarity (used as fit_mean_distances).
    /// `resultant_lengths`: per-cluster directional concentration [0, 1].
    pub fn from_embedding_cluster(
        centroids: &[f64],
        k: usize,
        fitted_d: usize,
        input_dim: usize,
        pca: Option<&crate::embedding::reduction::PcaProjection>,
        labels: &[usize],
        intra_similarity: &[f64],
        resultant_lengths: &[f64],
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: Some(resultant_lengths.to_vec()),
            fit_distance_stats: None,
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

#[cfg(test)]
mod tests {
    use super::assign::assign_max_dot_two;
    use super::calibrate::percentile_sorted;
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
            fit_distance_stats: None,
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

        // Point at midpoint (5,0), equidistant to both
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

        // Equidistant point, low confidence
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

    #[test]
    fn test_calibrate_populates_confidence_stats() {
        let mut snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);

        // Training data: 50 points near each centroid
        let mut data = Vec::new();
        for i in 0..50 {
            data.push(0.0 + (i as f64) * 0.01);
            data.push(0.0 + (i as f64) * 0.01);
        }
        for i in 0..50 {
            data.push(10.0 + (i as f64) * 0.01);
            data.push(10.0 + (i as f64) * 0.01);
        }

        snap.calibrate(&data, 100).unwrap();

        let stats = snap.confidence_stats.as_ref().unwrap();
        assert_eq!(stats.p5.len(), 2);
        assert_eq!(stats.p10.len(), 2);
        assert_eq!(stats.p25.len(), 2);
        assert_eq!(stats.p50.len(), 2);

        // Quantiles should be ordered
        for c in 0..2 {
            assert!(
                stats.p5[c] <= stats.p10[c],
                "P5={} > P10={}",
                stats.p5[c],
                stats.p10[c]
            );
            assert!(
                stats.p10[c] <= stats.p25[c],
                "P10={} > P25={}",
                stats.p10[c],
                stats.p25[c]
            );
            assert!(
                stats.p25[c] <= stats.p50[c],
                "P25={} > P50={}",
                stats.p25[c],
                stats.p50[c]
            );
        }

        assert_eq!(snap.version, 2);
    }

    #[test]
    fn test_calibrate_empty_cluster() {
        // k=3 but only 2 clusters get data
        let mut snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0, 1000.0, 1000.0], 3, 2);
        let data = vec![0.1, 0.1, 9.9, 9.9];
        snap.calibrate(&data, 2).unwrap();

        let stats = snap.confidence_stats.as_ref().unwrap();
        // Cluster 2 got no points, quantiles should be 0
        assert_eq!(stats.p10[2], 0.0);
        assert_eq!(stats.p50[2], 0.0);
    }

    #[test]
    fn test_calibrate_new_fields_none_before() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0], 1, 2);
        assert!(snap.confidence_stats.is_none());
        assert!(snap.cluster_variances.is_none());
        assert!(snap.fit_kappa.is_none());
        assert!(snap.fit_resultant_lengths.is_none());
    }

    #[test]
    fn test_adaptive_rejection_per_cluster() {
        let mut snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);

        // Calibrate with well-separated training data
        let mut data = Vec::new();
        for i in 0..100 {
            data.push(0.0 + (i as f64) * 0.01);
            data.push(0.0 + (i as f64) * 0.01);
        }
        for i in 0..100 {
            data.push(10.0 + (i as f64) * 0.01);
            data.push(10.0 + (i as f64) * 0.01);
        }
        snap.calibrate(&data, 200).unwrap();

        // Assign a point with moderate confidence
        let test_data = vec![1.0, 1.0]; // near cluster 0 but not on top
        let mut result = snap.assign_batch(&test_data, 1).unwrap();
        assert!(!result.rejected[0]);

        let stats = snap.confidence_stats.as_ref().unwrap();

        // Try adaptive rejection with a lenient percentile
        result.apply_adaptive_rejection(stats, "p5").unwrap();

        // Verify the method doesn't reject already-rejected points again
        let already_rejected_count = result.rejected.iter().filter(|&&r| r).count();
        result.apply_adaptive_rejection(stats, "p50").unwrap();
        let new_rejected_count = result.rejected.iter().filter(|&&r| r).count();
        assert!(new_rejected_count >= already_rejected_count);
    }

    #[test]
    fn test_adaptive_rejection_invalid_percentile() {
        let stats = ClusterConfidenceStats {
            p5: vec![0.1],
            p10: vec![0.2],
            p25: vec![0.3],
            p50: vec![0.4],
        };
        let mut result = AssignmentResult {
            labels: vec![0],
            distances: vec![1.0],
            second_distances: vec![2.0],
            confidences: vec![0.5],
            rejected: vec![false],
        };
        let err = result.apply_adaptive_rejection(&stats, "p99");
        assert!(err.is_err());
    }

    #[test]
    fn test_percentile_sorted() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_sorted(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile_sorted(&sorted, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile_sorted(&sorted, 100.0) - 5.0).abs() < 1e-10);
        assert!((percentile_sorted(&sorted, 25.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_mahalanobis_prefers_elongated_cluster() {
        // Two centroids at (0,0) and (10,0)
        // Cluster 0 has high variance on y-axis (elongated vertically)
        // Cluster 1 has low variance on both axes (compact)
        let mut snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 0.0], 2, 2);

        // Manually set variances: cluster 0 is elongated on y (var_y=100), cluster 1 is compact
        snap.cluster_variances = Some(ClusterVariances {
            variances: vec![
                1.0, 100.0, // cluster 0: narrow x, wide y
                1.0, 1.0, // cluster 1: compact both
            ],
        });

        // Point at (5, 8); equidistant in Euclidean, but closer to cluster 0 in Mahalanobis
        // because cluster 0 has high y-variance
        let data = vec![5.0, 8.0];

        let _result_eucl = snap.assign_batch(&data, 1).unwrap();
        let result_mahal = snap.assign_batch_mahalanobis(&data, 1).unwrap();

        // Euclidean: (5-0)^2 + (8-0)^2 = 89 vs (5-10)^2 + (8-0)^2 = 89, tie goes to cluster 0
        // Mahalanobis cluster 0: 25/1 + 64/100 = 25.64
        // Mahalanobis cluster 1: 25/1 + 64/1 = 89
        assert_eq!(result_mahal.labels[0], 0);
        assert!(
            result_mahal.distances[0] < result_mahal.second_distances[0],
            "Mahalanobis should clearly prefer cluster 0"
        );
    }

    #[test]
    fn test_mahalanobis_requires_calibration() {
        let snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);
        let data = vec![1.0, 1.0];
        let err = snap.assign_batch_mahalanobis(&data, 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_calibrate_populates_variances() {
        let mut snap = make_kmeans_snapshot(vec![0.0, 0.0, 10.0, 10.0], 2, 2);
        let data = vec![0.1, 0.1, 0.2, 0.2, 9.9, 9.9, 10.1, 10.1];
        snap.calibrate(&data, 4).unwrap();
        assert!(snap.cluster_variances.is_some());
        let cv = snap.cluster_variances.as_ref().unwrap();
        assert_eq!(cv.variances.len(), 2 * 2); // k * d
    }
}
