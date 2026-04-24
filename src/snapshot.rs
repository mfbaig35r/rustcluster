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
    /// Per-cluster mean distance at fit time (for drift detection).
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
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
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
            fit_mean_distances: vec![0.0; k],
            fit_cluster_sizes,
            fit_n_samples: n_samples,
            version: 1,
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
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

    /// Reject points whose confidence falls below the per-cluster adaptive threshold.
    ///
    /// Uses calibration data to set per-cluster thresholds based on the training
    /// confidence distribution. A point is rejected if its confidence is below
    /// the specified percentile of its assigned cluster's training distribution.
    pub fn apply_adaptive_rejection(
        &mut self,
        stats: &ClusterConfidenceStats,
        percentile: &str,
    ) -> Result<(), ClusterError> {
        let thresholds = match percentile {
            "p5" => &stats.p5,
            "p10" => &stats.p10,
            "p25" => &stats.p25,
            "p50" => &stats.p50,
            _ => {
                return Err(ClusterError::SnapshotContract(format!(
                    "unknown percentile '{}', use p5/p10/p25/p50",
                    percentile
                )))
            }
        };

        for i in 0..self.labels.len() {
            if self.rejected[i] {
                continue;
            }
            let label = self.labels[i];
            if label >= 0 && (label as usize) < thresholds.len() {
                let cluster_threshold = thresholds[label as usize];
                if self.confidences[i] < cluster_threshold {
                    self.rejected[i] = true;
                    self.labels[i] = -1;
                }
            }
        }
        Ok(())
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
    /// Per-cluster kappa shift (spherical only, requires calibration).
    /// (new_kappa - fit_kappa) / fit_kappa per cluster.
    pub kappa_drift: Option<Vec<f64>>,
    /// Per-cluster centroid direction shift (spherical only).
    /// 1.0 - dot(old_centroid, new_mean_direction) per cluster. 0 = no shift.
    pub direction_drift: Option<Vec<f64>>,
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

        // vMF drift (spherical + calibrated only)
        let (kappa_drift, direction_drift) = if self.spherical && self.fit_kappa.is_some() {
            let work_data = self.preprocess(data, n)?;
            let d = self.d;
            let fit_kappa = self.fit_kappa.as_ref().unwrap();
            let d_f = d as f64;

            let mut new_kappas = vec![0.0; k];
            let mut new_mean_dirs = vec![0.0f64; k * d];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let label = result.labels[i];
                if label >= 0 && (label as usize) < k {
                    let c = label as usize;
                    counts[c] += 1;
                    for j in 0..d {
                        new_mean_dirs[c * d + j] += work_data[i * d + j];
                    }
                }
            }

            let mut kd = vec![0.0; k];
            let mut dd = vec![0.0; k];

            for c in 0..k {
                if counts[c] > 0 {
                    // Compute mean resultant length
                    let r_bar: f64 = new_mean_dirs[c * d..(c + 1) * d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        / counts[c] as f64;
                    let denom = (1.0 - r_bar * r_bar).max(1e-10);
                    new_kappas[c] = r_bar * (d_f - r_bar * r_bar) / denom;

                    // Kappa drift
                    if fit_kappa[c].abs() > 1e-10 {
                        kd[c] = (new_kappas[c] - fit_kappa[c]) / fit_kappa[c].abs();
                    }

                    // Normalize new mean direction for dot product
                    let norm: f64 = new_mean_dirs[c * d..(c + 1) * d]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt();
                    if norm > 1e-30 {
                        for j in 0..d {
                            new_mean_dirs[c * d + j] /= norm;
                        }
                    }

                    // Direction drift: 1 - dot(old_centroid, new_mean_dir)
                    let centroid = &self.centroids[c * d..(c + 1) * d];
                    let new_dir = &new_mean_dirs[c * d..(c + 1) * d];
                    let dot_val: f64 = centroid.iter().zip(new_dir).map(|(a, b)| a * b).sum();
                    dd[c] = 1.0 - dot_val.clamp(-1.0, 1.0);
                }
            }

            (Some(kd), Some(dd))
        } else {
            (None, None)
        };

        Ok(DriftReport {
            new_mean_distances,
            new_cluster_sizes: cluster_counts,
            relative_drift,
            global_mean_distance,
            rejection_rate,
            n_samples: n,
            kappa_drift,
            direction_drift,
        })
    }

    /// Calibrate per-cluster confidence thresholds from representative data.
    ///
    /// Assigns the calibration data, collects per-cluster confidence scores,
    /// and computes quantiles (P5, P10, P25, P50) for adaptive rejection.
    ///
    /// `data`: flat row-major f64, shape (n, input_dim).
    pub fn calibrate(&mut self, data: &[f64], n: usize) -> Result<(), ClusterError> {
        // Preprocess data (same pipeline as assign_batch)
        let work_data = self.preprocess(data, n)?;
        let result = self.assign_batch(data, n)?;
        let k = self.k;
        let d = self.d;

        // --- Confidence quantiles ---
        let mut per_cluster: Vec<Vec<f64>> = vec![vec![]; k];
        for i in 0..n {
            let label = result.labels[i];
            if label >= 0 && (label as usize) < k {
                per_cluster[label as usize].push(result.confidences[i]);
            }
        }

        let mut p5 = vec![0.0; k];
        let mut p10 = vec![0.0; k];
        let mut p25 = vec![0.0; k];
        let mut p50 = vec![0.0; k];

        for c in 0..k {
            per_cluster[c].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if !per_cluster[c].is_empty() {
                p5[c] = percentile_sorted(&per_cluster[c], 5.0);
                p10[c] = percentile_sorted(&per_cluster[c], 10.0);
                p25[c] = percentile_sorted(&per_cluster[c], 25.0);
                p50[c] = percentile_sorted(&per_cluster[c], 50.0);
            }
        }
        self.confidence_stats = Some(ClusterConfidenceStats { p5, p10, p25, p50 });

        // --- vMF kappa (spherical only) ---
        if self.spherical {
            let d_f = d as f64;
            let mut kappas = vec![0.0; k];

            for c in 0..k {
                let mut sum_vec = vec![0.0; d];
                let mut count = 0usize;
                for i in 0..n {
                    if result.labels[i] == c as i64 {
                        for j in 0..d {
                            sum_vec[j] += work_data[i * d + j];
                        }
                        count += 1;
                    }
                }
                if count > 0 {
                    let r_bar: f64 =
                        sum_vec.iter().map(|v| v * v).sum::<f64>().sqrt() / count as f64;
                    let denom = (1.0 - r_bar * r_bar).max(1e-10);
                    kappas[c] = r_bar * (d_f - r_bar * r_bar) / denom;
                }
            }
            self.fit_kappa = Some(kappas);
        }

        // --- Per-cluster variances (for Mahalanobis, all metrics) ---
        let mut means = vec![0.0f64; k * d];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let label = result.labels[i] as usize;
            if label < k {
                counts[label] += 1;
                for j in 0..d {
                    means[label * d + j] += work_data[i * d + j];
                }
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    means[c * d + j] /= counts[c] as f64;
                }
            }
        }

        let mut var_flat = vec![0.0f64; k * d];
        for i in 0..n {
            let label = result.labels[i] as usize;
            if label < k {
                for j in 0..d {
                    let diff = work_data[i * d + j] - means[label * d + j];
                    var_flat[label * d + j] += diff * diff;
                }
            }
        }
        for c in 0..k {
            if counts[c] > 1 {
                for j in 0..d {
                    var_flat[c * d + j] /= (counts[c] - 1) as f64;
                    if var_flat[c * d + j] < 1e-12 {
                        var_flat[c * d + j] = 1e-12;
                    }
                }
            } else {
                for j in 0..d {
                    var_flat[c * d + j] = 1e-12;
                }
            }
        }
        self.cluster_variances = Some(ClusterVariances {
            variances: var_flat,
        });

        self.version = 2;
        Ok(())
    }
}

/// Linear-interpolation percentile on a pre-sorted slice.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    if hi >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
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

/// Find nearest two centroids by diagonal Mahalanobis distance.
///
/// Mahalanobis distance: sum((x_i - mu_i)^2 / var_i) per dimension.
fn assign_nearest_two_mahalanobis(
    point: &[f64],
    centroids: &[f64],
    variances: &[f64], // flat (k * d), per-cluster per-dimension variance
    k: usize,
    d: usize,
) -> (usize, f64, f64) {
    debug_assert!(k >= 1);

    if k == 1 {
        let mut dist = 0.0;
        for j in 0..d {
            let diff = point[j] - centroids[j];
            dist += diff * diff / variances[j];
        }
        return (0, dist, f64::INFINITY);
    }

    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    let mut second_dist = f64::MAX;

    for c in 0..k {
        let mut dist = 0.0;
        for j in 0..d {
            let diff = point[j] - centroids[c * d + j];
            dist += diff * diff / variances[c * d + j];
        }
        if dist < best_dist {
            second_dist = best_dist;
            best_dist = dist;
            best_idx = c;
        } else if dist < second_dist {
            second_dist = dist;
        }
    }

    (best_idx, best_dist, second_dist)
}

impl ClusterSnapshot {
    /// Assign using diagonal Mahalanobis distance (requires calibration).
    pub fn assign_batch_mahalanobis(
        &self,
        data: &[f64],
        n: usize,
    ) -> Result<AssignmentResult, ClusterError> {
        let cv = self.cluster_variances.as_ref().ok_or_else(|| {
            ClusterError::SnapshotContract("calibrate() required for Mahalanobis mode".to_string())
        })?;

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

        let work_data = self.preprocess(data, n)?;
        let d = self.d;
        let k = self.k;
        let centroids = &self.centroids[..];
        let variances = &cv.variances[..];

        let results: Vec<(usize, f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &work_data[i * d..(i + 1) * d];
                assign_nearest_two_mahalanobis(point, centroids, variances, k, d)
            })
            .collect();

        let mut labels = Vec::with_capacity(n);
        let mut distances = Vec::with_capacity(n);
        let mut second_distances = Vec::with_capacity(n);
        let mut confidences = Vec::with_capacity(n);

        for (idx, best, second) in &results {
            labels.push(*idx as i64);
            distances.push(*best);
            second_distances.push(*second);

            let conf = if k < 2 {
                0.0
            } else if !second.is_finite() || second.abs() < 1e-30 {
                0.0
            } else {
                1.0 - (best / second).clamp(0.0, 1.0)
            };
            confidences.push(conf);
        }

        Ok(AssignmentResult {
            labels,
            distances,
            second_distances,
            confidences,
            rejected: vec![false; n],
        })
    }
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
            confidence_stats: None,
            cluster_variances: None,
            fit_kappa: None,
            fit_resultant_lengths: None,
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
        // Cluster 2 got no points — quantiles should be 0
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
        // Using p50 should reject more aggressively than p5
        let conf = result.confidences[0];

        // Try adaptive rejection with a lenient percentile
        result.apply_adaptive_rejection(stats, "p5").unwrap();
        // Whether it's rejected depends on the cluster's P5 threshold

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

        // Point at (5, 8) — equidistant in Euclidean, but closer to cluster 0 in Mahalanobis
        // because cluster 0 has high y-variance
        let data = vec![5.0, 8.0];

        let result_eucl = snap.assign_batch(&data, 1).unwrap();
        let result_mahal = snap.assign_batch_mahalanobis(&data, 1).unwrap();

        // Euclidean: (5-0)^2 + (8-0)^2 = 89 vs (5-10)^2 + (8-0)^2 = 89 — tie, goes to cluster 0
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
