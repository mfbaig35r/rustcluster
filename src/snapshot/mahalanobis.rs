//! Diagonal Mahalanobis assignment using calibration variances.

use rayon::prelude::*;

use super::{AssignmentResult, ClusterSnapshot};
use crate::error::ClusterError;

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
