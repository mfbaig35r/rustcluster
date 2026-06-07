//! Per-cluster calibration: confidence quantiles, vMF kappa, variances, distance stats.

use super::{ClusterConfidenceStats, ClusterDistanceStats, ClusterSnapshot, ClusterVariances};
use crate::error::ClusterError;

impl ClusterSnapshot {
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

        // --- Per-cluster mean+std of assignment distance (for rejection_rate) ---
        let mut dist_sums = vec![0.0f64; k];
        let mut dist_sq_sums = vec![0.0f64; k];
        let mut dist_counts = vec![0usize; k];
        for i in 0..n {
            let label = result.labels[i];
            if label < 0 {
                continue;
            }
            let c = label as usize;
            if c >= k {
                continue;
            }
            let d_val = result.distances[i];
            dist_sums[c] += d_val;
            dist_sq_sums[c] += d_val * d_val;
            dist_counts[c] += 1;
        }
        let mut dist_mean = vec![0.0f64; k];
        let mut dist_std = vec![0.0f64; k];
        for c in 0..k {
            if dist_counts[c] > 0 {
                let m = dist_sums[c] / dist_counts[c] as f64;
                dist_mean[c] = m;
                if dist_counts[c] > 1 {
                    let denom = (dist_counts[c] - 1) as f64;
                    let var = ((dist_sq_sums[c] - dist_counts[c] as f64 * m * m) / denom).max(0.0);
                    dist_std[c] = var.sqrt();
                }
            }
        }
        self.fit_distance_stats = Some(ClusterDistanceStats {
            mean: dist_mean,
            std: dist_std,
        });

        self.version = 2;
        Ok(())
    }
}

/// Linear-interpolation percentile on a pre-sorted slice.
pub(super) fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
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
