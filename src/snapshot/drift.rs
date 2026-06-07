//! Drift detection: compare new data against the fit-time distribution.

use super::ClusterSnapshot;
use crate::error::ClusterError;

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
    /// Fraction of points whose assignment distance/similarity falls more than
    /// 3 std outside the per-cluster fit-time distribution. NaN when the
    /// snapshot has not been calibrated (call `calibrate()` first).
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
        debug_assert_eq!(
            self.fit_mean_distances.len(),
            self.k,
            "snapshot factory must populate fit_mean_distances with k entries",
        );
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

        // Rejection rate: fraction of points whose assignment distance falls
        // outside the per-cluster fit-time distribution by more than k_sigma
        // standard deviations. Requires calibration; NaN otherwise.
        //
        // For non-spherical (distance) metrics: outlier = unusually high distance.
        // For spherical (similarity) metrics: outlier = unusually low similarity.
        let rejection_rate = match self.fit_distance_stats.as_ref() {
            Some(stats) if n > 0 => {
                let k_sigma = 3.0_f64;
                let mut rejected = 0usize;
                for i in 0..n {
                    let label = result.labels[i];
                    if label < 0 {
                        continue;
                    }
                    let c = label as usize;
                    if c >= k {
                        continue;
                    }
                    let mean = stats.mean[c];
                    let std = stats.std[c].max(1e-12);
                    let d_val = result.distances[i];
                    let is_outlier = if self.spherical {
                        d_val < mean - k_sigma * std
                    } else {
                        d_val > mean + k_sigma * std
                    };
                    if is_outlier {
                        rejected += 1;
                    }
                }
                rejected as f64 / n as f64
            }
            _ => f64::NAN,
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
}
