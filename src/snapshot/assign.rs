//! Batch assignment of new points to a frozen snapshot.

use rayon::prelude::*;

use super::{ClusterConfidenceStats, ClusterSnapshot, Preprocessing};
use crate::distance::{
    CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean,
};
use crate::embedding::{normalize, reduction};
use crate::error::ClusterError;
use crate::utils::assign_nearest_two_with;

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
                // Only one cluster; confidence is meaningless
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
            // Only one cluster; second distance is infinity
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
    pub(super) fn preprocess(&self, data: &[f64], n: usize) -> Result<Vec<f64>, ClusterError> {
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

/// Find centroid with maximum dot product, returning best and second-best.
///
/// Used for spherical (cosine) assignment on unit-normalized data.
pub(super) fn assign_max_dot_two(
    point: &[f64],
    centroids: &[f64],
    k: usize,
    d: usize,
) -> (usize, f64, f64) {
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
