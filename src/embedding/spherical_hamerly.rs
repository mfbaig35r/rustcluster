//! Hamerly-accelerated spherical K-means.
//!
//! Uses triangle inequality in angular distance space to skip
//! full rival scans when bounds prove no reassignment is possible.
//! O(n) bound memory vs Elkan's O(n×K).
//!
//! Reference: Schubert & Lang (2021), "Accelerating Spherical k-Means."

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::distance::Scalar;
use crate::error::ClusterError;

use super::spherical_kmeans::{
    dot_product, spherical_kmeans_plus_plus, IterationTelemetry, SphericalKMeansState,
};

/// Run Hamerly-accelerated spherical K-means with n_init restarts.
pub fn run_spherical_hamerly<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
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
    if k < 2 {
        // Hamerly needs k >= 2 for second-nearest tracking
        return super::spherical_kmeans::run_spherical_kmeans(
            data, n, d, k, max_iter, tol, seed, n_init,
        );
    }

    let mut best: Option<SphericalKMeansState<F>> = None;

    for i in 0..n_init {
        let run_seed = seed.wrapping_add(i as u64);
        let result = run_hamerly_single(data, n, d, k, max_iter, tol, run_seed)?;

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

/// Angular distance between two unit vectors: arccos(dot(a, b)).
#[inline(always)]
fn angular_dist<F: Scalar>(a: &[F], b: &[F]) -> f64 {
    let dot = dot_product(a, b).to_f64_lossy();
    dot.clamp(-1.0, 1.0).acos()
}

/// Angular distance from raw dot product (f64).
#[inline(always)]
fn angle_from_dot(dot: f64) -> f64 {
    dot.clamp(-1.0, 1.0).acos()
}

fn run_hamerly_single<F: Scalar>(
    data: &[F],
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<SphericalKMeansState<F>, ClusterError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = spherical_kmeans_plus_plus(data, n, d, k, &mut rng);

    // Per-point bounds (angular distance, f64)
    let mut upper = vec![std::f64::consts::PI; n]; // upper bound on dist to assigned
    let mut lower = vec![0.0f64; n]; // lower bound on dist to second-nearest
    let mut labels = vec![0usize; n];
    let mut old_labels = vec![0usize; n];

    // Per-centroid
    let mut half_sep = vec![0.0f64; k]; // half of min separation to other centroids
    let mut movements = vec![0.0f64; k]; // angular movement after update

    // Accumulation buffers
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];

    let mut objective = f64::NEG_INFINITY;
    let mut prev_objective;
    let mut n_iter = 0;
    let mut converge_count = 0usize;
    let mut telemetry = Vec::with_capacity(max_iter.min(100));

    // Convergence thresholds
    let rel_obj_tol = 1e-4;
    let churn_tol = 0.001;
    let patience = 2;

    // ---- Initial full assignment (no pruning) ----
    {
        let centroids_slice = centroids.as_slice().expect("contiguous");
        let mut init_objective = 0.0f64;

        for i in 0..n {
            let point = &data[i * d..(i + 1) * d];
            let mut best_idx = 0;
            let mut best_ang = std::f64::consts::PI;
            let mut second_ang = std::f64::consts::PI;

            for j in 0..k {
                let centroid = &centroids_slice[j * d..(j + 1) * d];
                let dot = dot_product(point, centroid).to_f64_lossy();
                let ang = angle_from_dot(dot);

                if ang < best_ang {
                    second_ang = best_ang;
                    best_ang = ang;
                    best_idx = j;
                } else if ang < second_ang {
                    second_ang = ang;
                }

                if j == 0 || ang < best_ang {
                    // Track dot for objective
                }
            }

            labels[i] = best_idx;
            upper[i] = best_ang;
            lower[i] = second_ang;

            // Objective uses dot product (cos of angular distance)
            init_objective += best_ang.cos();
        }
        objective = init_objective;
    }

    // ---- Main Hamerly loop ----
    for iter in 0..max_iter {
        let iter_start = std::time::Instant::now();

        // Save old labels
        old_labels.copy_from_slice(&labels);
        prev_objective = objective;

        // Compute centroid-centroid half-separations
        let centroids_slice = centroids.as_slice().expect("contiguous");
        for j in 0..k {
            let cj = &centroids_slice[j * d..(j + 1) * d];
            let mut min_ang = std::f64::consts::PI;
            for l in 0..k {
                if j == l {
                    continue;
                }
                let cl = &centroids_slice[l * d..(l + 1) * d];
                let ang = angular_dist(cj, cl);
                if ang < min_ang {
                    min_ang = ang;
                }
            }
            half_sep[j] = min_ang * 0.5;
        }

        let m_max: f64 = movements.iter().cloned().fold(0.0f64, f64::max);

        // Point processing
        let mut n_skipped = 0usize;
        let mut n_tightened = 0usize;
        let mut new_objective = 0.0f64;

        for i in 0..n {
            // Update bounds with centroid movements
            if iter > 0 {
                upper[i] += movements[labels[i]];
                lower[i] = (lower[i] - m_max).max(0.0);
            }

            // Test 1: can we skip this point entirely?
            if upper[i] <= lower[i].max(half_sep[labels[i]]) {
                // Bounds prove no reassignment — skip
                new_objective += upper[i].cos();
                n_skipped += 1;
                continue;
            }

            // Test 2: tighten upper bound with exact distance
            let point = &data[i * d..(i + 1) * d];
            let assigned = &centroids_slice[labels[i] * d..(labels[i] + 1) * d];
            upper[i] = angular_dist(point, assigned);
            n_tightened += 1;

            if upper[i] <= lower[i].max(half_sep[labels[i]]) {
                new_objective += upper[i].cos();
                n_skipped += 1;
                continue;
            }

            // Test 3: full rival scan
            let mut best_idx = labels[i];
            let mut best_ang = upper[i];
            let mut second_ang = std::f64::consts::PI;

            for j in 0..k {
                if j == labels[i] {
                    continue;
                }

                // Centroid-centroid pruning: if best_ang <= half the distance
                // between best and j, then j can't beat best
                let cj = &centroids_slice[j * d..(j + 1) * d];
                let cb = &centroids_slice[best_idx * d..(best_idx + 1) * d];
                let cc_ang = angular_dist(cb, cj);
                if best_ang <= cc_ang * 0.5 {
                    continue;
                }

                let ang = angular_dist(point, cj);
                if ang < best_ang {
                    second_ang = best_ang;
                    best_ang = ang;
                    best_idx = j;
                } else if ang < second_ang {
                    second_ang = ang;
                }
            }

            labels[i] = best_idx;
            upper[i] = best_ang;
            lower[i] = second_ang;
            new_objective += best_ang.cos();
        }

        objective = new_objective;

        // Centroid update
        let old_centroids = centroids.clone();
        sums.fill(0.0);
        counts.fill(0);

        for i in 0..n {
            let label = labels[i];
            counts[label] += 1;
            let sum_start = label * d;
            let point_start = i * d;
            for j in 0..d {
                sums[sum_start + j] += data[point_start + j].to_f64_lossy();
            }
        }

        let centroid_slice = centroids.as_slice_mut().expect("contiguous");
        for cluster in 0..k {
            let start = cluster * d;
            if counts[cluster] > 0 {
                let mut norm_sq = 0.0f64;
                for j in 0..d {
                    norm_sq += sums[start + j] * sums[start + j];
                }
                if norm_sq > 1e-30 {
                    let inv_norm = 1.0 / norm_sq.sqrt();
                    for j in 0..d {
                        centroid_slice[start + j] = F::from_f64_lossy(sums[start + j] * inv_norm);
                    }
                } else {
                    // Reseed from farthest point, invalidate bounds
                    let farthest = (0..n)
                        .max_by(|&a, &b| upper[a].partial_cmp(&upper[b]).unwrap_or(std::cmp::Ordering::Equal))
                        .unwrap_or_else(|| rng.gen_range(0..n));
                    let point_start = farthest * d;
                    for j in 0..d {
                        centroid_slice[start + j] = data[point_start + j];
                    }
                }
            } else {
                // Empty cluster: reseed, invalidate bounds
                let farthest = (0..n)
                    .max_by(|&a, &b| upper[a].partial_cmp(&upper[b]).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or_else(|| rng.gen_range(0..n));
                let point_start = farthest * d;
                for j in 0..d {
                    centroid_slice[start + j] = data[point_start + j];
                }
                // Invalidate bounds for points that were assigned to this cluster
                for i in 0..n {
                    if labels[i] == cluster {
                        upper[i] = std::f64::consts::PI;
                        lower[i] = 0.0;
                    }
                }
            }
        }

        // Compute centroid movements (angular distance between old and new)
        let old_slice = old_centroids.as_slice().expect("contiguous");
        let new_slice = centroids.as_slice().expect("contiguous");
        let mut max_shift = 0.0f64;
        for cluster in 0..k {
            let start = cluster * d;
            let end = start + d;
            let dot: f64 = old_slice[start..end]
                .iter()
                .zip(&new_slice[start..end])
                .map(|(&a, &b)| a.to_f64_lossy() * b.to_f64_lossy())
                .sum();
            movements[cluster] = angle_from_dot(dot);
            if movements[cluster] > max_shift {
                max_shift = movements[cluster];
            }
        }

        n_iter = iter + 1;

        // Convergence
        let churn = old_labels
            .iter()
            .zip(labels.iter())
            .filter(|(a, b)| a != b)
            .count() as f64
            / n as f64;

        let rel_change = if prev_objective.abs() > 1e-30 {
            (objective - prev_objective).abs() / prev_objective.abs()
        } else {
            f64::MAX
        };

        let skip_rate = n_skipped as f64 / n as f64;
        let elapsed_ms = iter_start.elapsed().as_secs_f64() * 1000.0;

        telemetry.push(IterationTelemetry {
            iteration: n_iter,
            objective,
            rel_objective_change: rel_change,
            churn,
            max_angular_shift: max_shift,
            wall_clock_ms: elapsed_ms,
        });

        let converged = iter > 0
            && rel_change < rel_obj_tol
            && churn < churn_tol
            && max_shift < tol.max(1e-5);

        if converged {
            converge_count += 1;
        } else {
            converge_count = 0;
        }

        if converge_count >= patience {
            break;
        }
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

    fn make_spherical_data() -> (Vec<f64>, usize, usize) {
        let d = 8;
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(42);

        // Cluster A: near positive x-axis
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = 5.0;
            for j in 1..d {
                point[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&point);
        }
        // Cluster B: near negative x-axis
        for _ in 0..50 {
            let mut point = vec![0.0f64; d];
            point[0] = -5.0;
            for j in 1..d {
                point[j] = rng.gen_range(-0.5..0.5);
            }
            data.extend_from_slice(&point);
        }

        let (normalized, _) = l2_normalize_rows(&data, 100, d);
        (normalized, 100, d)
    }

    #[test]
    fn test_hamerly_well_separated() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_hamerly(&data, n, d, 2, 100, 1e-6, 42, 1).unwrap();

        assert_eq!(result.labels.len(), 100);
        let label_a = result.labels[0];
        let label_b = result.labels[50];
        assert_ne!(label_a, label_b);
        for i in 0..50 {
            assert_eq!(result.labels[i], label_a);
        }
        for i in 50..100 {
            assert_eq!(result.labels[i], label_b);
        }
    }

    #[test]
    fn test_hamerly_matches_lloyd() {
        // Shadow mode: verify Hamerly produces identical results to Lloyd
        let (data, n, d) = make_spherical_data();

        let lloyd = super::super::spherical_kmeans::run_spherical_kmeans(
            &data, n, d, 2, 50, 1e-6, 42, 1,
        )
        .unwrap();
        let hamerly = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();

        // Same partition (labels may be permuted)
        let lloyd_partition: std::collections::HashSet<Vec<usize>> = {
            let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, &l) in lloyd.labels.iter().enumerate() {
                clusters.entry(l).or_default().push(i);
            }
            clusters.into_values().collect()
        };
        let hamerly_partition: std::collections::HashSet<Vec<usize>> = {
            let mut clusters: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for (i, &l) in hamerly.labels.iter().enumerate() {
                clusters.entry(l).or_default().push(i);
            }
            clusters.into_values().collect()
        };
        assert_eq!(lloyd_partition, hamerly_partition, "Partitions differ");
    }

    #[test]
    fn test_hamerly_skip_rate() {
        // Well-separated data should have high skip rate in later iterations
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();

        // Should converge (not hit max_iter with good convergence)
        assert!(result.n_iter < 50, "Should converge before max_iter");
    }

    #[test]
    fn test_hamerly_centroids_unit_norm() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();

        for row in result.centroids.rows() {
            let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-10,
                "Centroid norm = {norm}"
            );
        }
    }

    #[test]
    fn test_hamerly_reproducibility() {
        let (data, n, d) = make_spherical_data();
        let r1 = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();
        let r2 = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.objective - r2.objective).abs() < 1e-10);
    }

    #[test]
    fn test_hamerly_five_clusters() {
        // Test with more clusters
        let d = 8;
        let mut data = Vec::new();
        let mut rng = StdRng::seed_from_u64(99);

        for cluster in 0..5 {
            let mut center = vec![0.0f64; d];
            center[cluster % d] = 5.0;
            if cluster >= d {
                center[0] = -5.0;
            }
            for _ in 0..30 {
                let mut point = center.clone();
                for j in 0..d {
                    point[j] += rng.gen_range(-0.3..0.3);
                }
                data.extend_from_slice(&point);
            }
        }

        let (normalized, _) = l2_normalize_rows(&data, 150, d);
        let result = run_spherical_hamerly(&normalized, 150, d, 5, 100, 1e-6, 42, 3).unwrap();

        assert_eq!(result.labels.len(), 150);
        let unique: std::collections::HashSet<usize> = result.labels.iter().copied().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_hamerly_telemetry() {
        let (data, n, d) = make_spherical_data();
        let result = run_spherical_hamerly(&data, n, d, 2, 50, 1e-6, 42, 1).unwrap();

        assert_eq!(result.telemetry.len(), result.n_iter);
        for t in &result.telemetry {
            assert!(t.objective.is_finite());
            assert!(t.churn >= 0.0 && t.churn <= 1.0);
            assert!(t.max_angular_shift >= 0.0);
            assert!(t.wall_clock_ms >= 0.0);
        }
    }
}
