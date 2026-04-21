//! Hamerly's accelerated exact K-means algorithm.
//!
//! Maintains one upper bound (distance to assigned centroid) and one lower bound
//! (distance to second-closest centroid) per point. When upper <= lower, the
//! point's assignment cannot change — skip the full distance scan.
//!
//! Memory overhead: O(n) vs Elkan's O(n*k).
//! Best for: d < 50, moderate k.
//!
//! Reference: Hamerly, "Making k-means Even Faster" (SDM 2010).

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use rayon::prelude::*;

use crate::distance::{Distance, Scalar};
use crate::error::ClusterError;
use crate::kmeans::{compute_centroid_shifts, recompute_centroids, KMeansState};
use crate::utils::assign_nearest_two_with;

/// Run Hamerly's K-means iteration loop, generic over float type and distance.
///
/// Expects centroids to be already initialized (via kmeans++).
/// Requires k >= 2 (caller must enforce this).
/// Bounds are tracked in f64 for precision regardless of F.
pub fn run_hamerly_iterations<F: Scalar, D: Distance<F>>(
    data_slice: &[F],
    centroids: &mut Array2<F>,
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
) -> Result<KMeansState<F>, ClusterError> {
    debug_assert!(k >= 2, "Hamerly requires k >= 2");

    let mut labels = vec![0usize; n];
    let mut upper = vec![f64::MAX; n]; // bounds in f64 for precision
    let mut lower = vec![0.0f64; n];
    let mut inertia: f64;
    let mut n_iter;

    // --- Initial full assignment ---
    {
        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

        let assignments: Vec<(usize, F, F)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data_slice[i * d..(i + 1) * d];
                assign_nearest_two_with::<F, D>(point, centroids_slice, k, d)
            })
            .collect();

        inertia = 0.0;
        for (i, &(label, best_dist, second_dist)) in assignments.iter().enumerate() {
            labels[i] = label;
            upper[i] = best_dist.to_f64_lossy().sqrt();
            lower[i] = second_dist.to_f64_lossy().sqrt();
            inertia += best_dist.to_f64_lossy();
        }

        let old_centroids = centroids.clone();
        recompute_centroids(data_slice, &labels, centroids, n, d, k);
        handle_empty_clusters_hamerly(data_slice, &labels, &upper, centroids, n, d, k, rng);

        let shifts = compute_centroid_shifts::<F, D>(&old_centroids, centroids, k, d);
        let sqrt_shifts: Vec<f64> = shifts.iter().map(|&s| s.sqrt()).collect();
        let max_shift = sqrt_shifts.iter().cloned().fold(0.0f64, f64::max);

        for i in 0..n {
            upper[i] += sqrt_shifts[labels[i]];
            lower[i] -= max_shift;
        }

        n_iter = 1;

        if max_shift * max_shift < tol {
            let centroids_flat =
                std::sync::Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
            return Ok(KMeansState {
                centroids: centroids.clone(),
                centroids_flat,
                labels,
                inertia,
                n_iter,
            });
        }
    }

    // --- Main Hamerly loop ---
    for iter in 1..max_iter {
        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

        let center_half_dists = compute_center_half_dists::<F, D>(centroids_slice, k, d);

        let point_results: Vec<Option<(usize, usize, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let m = upper[i].max(0.0);
                let lo = lower[i];

                if m <= lo {
                    return None;
                }

                if m <= center_half_dists[labels[i]] {
                    return None;
                }

                let point = &data_slice[i * d..(i + 1) * d];
                let (new_label, best_dist, second_dist) =
                    assign_nearest_two_with::<F, D>(point, centroids_slice, k, d);

                Some((
                    i,
                    new_label,
                    best_dist.to_f64_lossy().sqrt(),
                    second_dist.to_f64_lossy().sqrt(),
                ))
            })
            .collect();

        for result in &point_results {
            if let Some((i, new_label, new_upper, new_lower)) = result {
                labels[*i] = *new_label;
                upper[*i] = *new_upper;
                lower[*i] = *new_lower;
            }
        }

        let old_centroids = centroids.clone();
        recompute_centroids(data_slice, &labels, centroids, n, d, k);
        handle_empty_clusters_hamerly(data_slice, &labels, &upper, centroids, n, d, k, rng);

        let shifts = compute_centroid_shifts::<F, D>(&old_centroids, centroids, k, d);
        let sqrt_shifts: Vec<f64> = shifts.iter().map(|&s| s.sqrt()).collect();
        let max_shift = sqrt_shifts.iter().cloned().fold(0.0f64, f64::max);

        for i in 0..n {
            upper[i] += sqrt_shifts[labels[i]];
            lower[i] -= max_shift;
            if lower[i] < 0.0 {
                lower[i] = 0.0;
            }
        }

        n_iter = iter + 1;

        let max_sq_shift = shifts.iter().cloned().fold(0.0f64, f64::max);
        if max_sq_shift < tol {
            break;
        }
    }

    // Final inertia computation
    let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");
    inertia = (0..n)
        .into_par_iter()
        .map(|i| {
            let point = &data_slice[i * d..(i + 1) * d];
            let centroid = &centroids_slice[labels[i] * d..(labels[i] + 1) * d];
            D::distance(point, centroid).to_f64_lossy()
        })
        .sum();

    let centroids_flat = std::sync::Arc::new(centroids.as_slice().expect("C-contiguous").to_vec());
    Ok(KMeansState {
        centroids: centroids.clone(),
        centroids_flat,
        labels,
        inertia,
        n_iter,
    })
}

/// Compute half the minimum inter-centroid distance for each centroid.
fn compute_center_half_dists<F: Scalar, D: Distance<F>>(
    centroids_slice: &[F],
    k: usize,
    d: usize,
) -> Vec<f64> {
    let mut half_dists = vec![f64::MAX; k];

    for j in 0..k {
        let cj = &centroids_slice[j * d..(j + 1) * d];
        for l in 0..k {
            if j == l {
                continue;
            }
            let cl = &centroids_slice[l * d..(l + 1) * d];
            let dist = D::distance(cj, cl).to_f64_lossy().sqrt();
            let half = dist * 0.5;
            if half < half_dists[j] {
                half_dists[j] = half;
            }
        }
    }

    half_dists
}

/// Re-seed empty clusters for Hamerly.
fn handle_empty_clusters_hamerly<F: Scalar>(
    data_slice: &[F],
    labels: &[usize],
    upper_bounds: &[f64],
    centroids: &mut Array2<F>,
    n: usize,
    d: usize,
    k: usize,
    rng: &mut StdRng,
) {
    let mut counts = vec![0usize; k];
    for &label in labels.iter() {
        counts[label] += 1;
    }

    let centroid_slice = centroids
        .as_slice_mut()
        .expect("centroids are C-contiguous");

    for cluster in 0..k {
        if counts[cluster] == 0 {
            let farthest = upper_bounds
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or_else(|| rng.gen_range(0..n));

            let start = cluster * d;
            let point_start = farthest * d;
            for j in 0..d {
                centroid_slice[start + j] = data_slice[point_start + j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distance::SquaredEuclidean;
    use crate::kmeans::{kmeans_plus_plus_init, Algorithm};
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

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
    fn test_hamerly_converges_on_separated_clusters() {
        let data = well_separated_data();
        let data_slice = data.as_slice().unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let mut centroids =
            kmeans_plus_plus_init::<f64, SquaredEuclidean>(&data.view(), 2, &mut rng);

        let result = run_hamerly_iterations::<f64, SquaredEuclidean>(
            data_slice,
            &mut centroids,
            8,
            2,
            2,
            100,
            1e-4,
            &mut rng,
        )
        .unwrap();

        assert_eq!(result.labels.len(), 8);
        let label_a = result.labels[0];
        let label_b = result.labels[4];
        assert_ne!(label_a, label_b);
        for i in 0..4 {
            assert_eq!(result.labels[i], label_a);
        }
        for i in 4..8 {
            assert_eq!(result.labels[i], label_b);
        }
    }

    #[test]
    fn test_hamerly_matches_lloyd() {
        let data = well_separated_data();
        let view = data.view();

        let lloyd =
            crate::kmeans::run_kmeans_n_init(&view, 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        let hamerly =
            crate::kmeans::run_kmeans_n_init(&view, 2, 100, 1e-4, 42, 1, Algorithm::Hamerly)
                .unwrap();

        assert_eq!(lloyd.labels, hamerly.labels);
        assert!((lloyd.inertia - hamerly.inertia).abs() < 1e-6);
    }

    #[test]
    fn test_hamerly_reproducibility() {
        let data = well_separated_data();
        let view = data.view();

        let r1 = crate::kmeans::run_kmeans_n_init(&view, 2, 100, 1e-4, 42, 1, Algorithm::Hamerly)
            .unwrap();
        let r2 = crate::kmeans::run_kmeans_n_init(&view, 2, 100, 1e-4, 42, 1, Algorithm::Hamerly)
            .unwrap();

        assert_eq!(r1.labels, r2.labels);
        assert!((r1.inertia - r2.inertia).abs() < 1e-10);
    }

    #[test]
    fn test_hamerly_larger_dataset() {
        let mut data_vec = Vec::new();
        for i in 0..50 {
            let f = i as f64 * 0.01;
            data_vec.extend_from_slice(&[f, f, f]);
        }
        for i in 0..50 {
            let f = 50.0 + i as f64 * 0.01;
            data_vec.extend_from_slice(&[f, f, f]);
        }
        for i in 0..50 {
            let f = 100.0 + i as f64 * 0.01;
            data_vec.extend_from_slice(&[f, f, f]);
        }
        let data = Array2::from_shape_vec((150, 3), data_vec).unwrap();
        let view = data.view();

        let result =
            crate::kmeans::run_kmeans_n_init(&view, 3, 100, 1e-4, 42, 1, Algorithm::Hamerly)
                .unwrap();
        assert_eq!(result.labels.len(), 150);
        assert!(result.inertia >= 0.0);
        let unique: std::collections::HashSet<_> = result.labels.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn test_center_half_dists() {
        let centroids = [0.0f64, 0.0, 10.0, 0.0];
        let half_dists = compute_center_half_dists::<f64, SquaredEuclidean>(&centroids, 2, 2);
        assert!((half_dists[0] - 5.0).abs() < 1e-10);
        assert!((half_dists[1] - 5.0).abs() < 1e-10);
    }
}
