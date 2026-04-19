use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::error::KMeansError;
use crate::utils::{assign_nearest, squared_euclidean, validate_data};

/// Result of a fitted K-means model.
pub struct KMeansState {
    pub centroids: Array2<f64>,
    pub labels: Vec<usize>,
    pub inertia: f64,
    pub n_iter: usize,
}

/// Run K-means with `n_init` random initializations, returning the best result.
pub fn run_kmeans_n_init(
    data: &ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
) -> Result<KMeansState, KMeansError> {
    let mut best: Option<KMeansState> = None;

    for i in 0..n_init {
        // Derive a deterministic seed for each run
        let run_seed = seed.wrapping_add(i as u64);
        let result = run_kmeans(data, k, max_iter, tol, run_seed)?;

        let is_better = match &best {
            None => true,
            Some(prev) => result.inertia < prev.inertia,
        };

        if is_better {
            best = Some(result);
        }
    }

    // n_init >= 1 guaranteed by caller, so best is always Some
    Ok(best.unwrap())
}

/// Run a single K-means fit: initialize centroids, then iterate Lloyd's algorithm.
fn run_kmeans(
    data: &ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<KMeansState, KMeansError> {
    validate_data(data)?;

    let (n, d) = data.dim();
    if k == 0 || k > n {
        return Err(KMeansError::InvalidClusters { k, n });
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize centroids via kmeans++
    let mut centroids = kmeans_plus_plus_init(data, k, &mut rng);

    // Data as flat slice — safe because caller guarantees C-contiguous
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let mut labels = vec![0usize; n];
    let mut inertia = f64::MAX;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

        // Assignment step (parallel)
        let assignments: Vec<(usize, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data_slice[i * d..(i + 1) * d];
                assign_nearest(point, centroids_slice, k, d)
            })
            .collect();

        // Unpack labels and compute inertia
        let mut new_inertia = 0.0;
        for (i, &(label, dist)) in assignments.iter().enumerate() {
            labels[i] = label;
            new_inertia += dist;
        }
        inertia = new_inertia;

        // Update step: recompute centroids as cluster means
        let old_centroids = centroids.clone();
        recompute_centroids(data_slice, &labels, &mut centroids, n, d, k);

        // Handle empty clusters by re-seeding from the farthest point
        handle_empty_clusters(data_slice, &labels, &mut centroids, &assignments, n, d, k, &mut rng);

        n_iter = iter + 1;

        // Convergence check: max centroid shift
        let max_shift = compute_max_shift(&old_centroids, &centroids, k, d);
        if max_shift < tol {
            break;
        }
    }

    Ok(KMeansState {
        centroids,
        labels,
        inertia,
        n_iter,
    })
}

/// K-means++ initialization: greedily pick centroids proportional to squared distance.
fn kmeans_plus_plus_init(
    data: &ArrayView2<f64>,
    k: usize,
    rng: &mut StdRng,
) -> Array2<f64> {
    let (n, d) = data.dim();
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let mut centroids = Array2::<f64>::zeros((k, d));

    // Pick first centroid uniformly at random
    let first = rng.gen_range(0..n);
    let first_row = &data_slice[first * d..(first + 1) * d];
    centroids.row_mut(0).assign(&ndarray::ArrayView1::from(first_row));

    // Min distance from each point to any chosen centroid
    let mut min_dists = vec![f64::MAX; n];

    for c in 1..k {
        // Update min distances with the last-added centroid
        let last_centroid = centroids.row(c - 1);
        let last_slice = last_centroid.as_slice().expect("centroid row is contiguous");

        for i in 0..n {
            let point = &data_slice[i * d..(i + 1) * d];
            let dist = squared_euclidean(point, last_slice);
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

        // Sample next centroid proportional to squared distances
        // If all distances are zero (duplicate points), fall back to uniform
        let total: f64 = min_dists.iter().sum();
        let next = if total <= 0.0 {
            rng.gen_range(0..n)
        } else {
            let dist = WeightedIndex::new(&min_dists).expect("weights are non-negative");
            dist.sample(rng)
        };

        let next_row = &data_slice[next * d..(next + 1) * d];
        centroids.row_mut(c).assign(&ndarray::ArrayView1::from(next_row));
    }

    centroids
}

/// Recompute centroids as the mean of assigned points.
fn recompute_centroids(
    data_slice: &[f64],
    labels: &[usize],
    centroids: &mut Array2<f64>,
    n: usize,
    d: usize,
    k: usize,
) {
    let centroid_slice = centroids.as_slice_mut().expect("centroids are C-contiguous");

    // Zero out sums
    let mut sums = vec![0.0; k * d];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let label = labels[i];
        counts[label] += 1;
        let point_start = i * d;
        let sum_start = label * d;
        for j in 0..d {
            sums[sum_start + j] += data_slice[point_start + j];
        }
    }

    for cluster in 0..k {
        if counts[cluster] > 0 {
            let count = counts[cluster] as f64;
            let start = cluster * d;
            for j in 0..d {
                centroid_slice[start + j] = sums[start + j] / count;
            }
        }
        // Empty clusters are handled separately in handle_empty_clusters
    }
}

/// Re-seed empty clusters with the point farthest from its assigned centroid.
fn handle_empty_clusters(
    data_slice: &[f64],
    labels: &[usize],
    centroids: &mut Array2<f64>,
    assignments: &[(usize, f64)],
    n: usize,
    d: usize,
    k: usize,
    rng: &mut StdRng,
) {
    let mut counts = vec![0usize; k];
    for &label in labels.iter() {
        counts[label] += 1;
    }

    let centroid_slice = centroids.as_slice_mut().expect("centroids are C-contiguous");

    for cluster in 0..k {
        if counts[cluster] == 0 {
            // Find the point with the largest distance to its assigned centroid
            let farthest = assignments
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
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

/// Compute the maximum squared shift across all centroids between iterations.
fn compute_max_shift(
    old: &Array2<f64>,
    new: &Array2<f64>,
    k: usize,
    d: usize,
) -> f64 {
    let old_slice = old.as_slice().expect("old centroids are contiguous");
    let new_slice = new.as_slice().expect("new centroids are contiguous");

    let mut max_shift = 0.0f64;
    for cluster in 0..k {
        let start = cluster * d;
        let end = start + d;
        let shift = squared_euclidean(&old_slice[start..end], &new_slice[start..end]);
        if shift > max_shift {
            max_shift = shift;
        }
    }
    max_shift
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn well_separated_data() -> Array2<f64> {
        // Two clear clusters: around (0,0) and (100,100)
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
    fn test_kmeans_plus_plus_returns_k_centroids() {
        let data = well_separated_data();
        let mut rng = StdRng::seed_from_u64(42);
        let centroids = kmeans_plus_plus_init(&data.view(), 2, &mut rng);
        assert_eq!(centroids.shape(), &[2, 2]);
    }

    #[test]
    fn test_kmeans_plus_plus_picks_from_data() {
        let data = well_separated_data();
        let mut rng = StdRng::seed_from_u64(42);
        let centroids = kmeans_plus_plus_init(&data.view(), 2, &mut rng);

        // Each centroid should be one of the data points
        for c in 0..2 {
            let centroid = centroids.row(c);
            let found = data.rows().into_iter().any(|row| {
                row.iter()
                    .zip(centroid.iter())
                    .all(|(a, b)| (a - b).abs() < 1e-10)
            });
            assert!(found, "Centroid {} not found in input data", c);
        }
    }

    #[test]
    fn test_lloyd_converges_on_separated_clusters() {
        let data = well_separated_data();
        let result = run_kmeans(&data.view(), 2, 100, 1e-4, 42).unwrap();

        assert_eq!(result.labels.len(), 8);
        assert_eq!(result.centroids.shape(), &[2, 2]);
        assert!(result.inertia >= 0.0);
        assert!(result.n_iter > 0);
        assert!(result.n_iter <= 100);

        // First 4 points should share one label, last 4 another
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
    fn test_reproducibility() {
        let data = well_separated_data();
        let r1 = run_kmeans(&data.view(), 2, 100, 1e-4, 42).unwrap();
        let r2 = run_kmeans(&data.view(), 2, 100, 1e-4, 42).unwrap();

        assert_eq!(r1.labels, r2.labels);
        assert!((r1.inertia - r2.inertia).abs() < 1e-10);
        assert_eq!(r1.n_iter, r2.n_iter);
    }

    #[test]
    fn test_n_init_selects_best() {
        let data = well_separated_data();
        let single = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 1).unwrap();
        let multi = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 10).unwrap();

        // n_init=10 should find equal or better inertia
        assert!(multi.inertia <= single.inertia + 1e-10);
    }

    #[test]
    fn test_k_greater_than_n_fails() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = run_kmeans(&data.view(), 5, 100, 1e-4, 42);
        assert!(matches!(result, Err(KMeansError::InvalidClusters { .. })));
    }

    #[test]
    fn test_empty_input_fails() {
        let data = Array2::<f64>::zeros((0, 2));
        let result = run_kmeans(&data.view(), 1, 100, 1e-4, 42);
        assert!(matches!(result, Err(KMeansError::EmptyInput)));
    }

    #[test]
    fn test_single_cluster() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = run_kmeans(&data.view(), 1, 100, 1e-4, 42).unwrap();

        // All points assigned to cluster 0
        assert!(result.labels.iter().all(|&l| l == 0));
        // Centroid should be the mean: (3, 4)
        let center = result.centroids.row(0);
        assert!((center[0] - 3.0).abs() < 1e-10);
        assert!((center[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_k_equals_n() {
        let data = array![[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]];
        let result = run_kmeans(&data.view(), 3, 100, 1e-4, 42).unwrap();

        // Each point should be its own cluster
        let mut sorted_labels = result.labels.clone();
        sorted_labels.sort();
        assert_eq!(sorted_labels, vec![0, 1, 2]);

        // Inertia should be 0 (each point is exactly its centroid)
        assert!(result.inertia < 1e-10);
    }

    #[test]
    fn test_identical_points() {
        let data = array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
        let result = run_kmeans(&data.view(), 1, 100, 1e-4, 42).unwrap();

        assert!(result.labels.iter().all(|&l| l == 0));
        assert!(result.inertia < 1e-10);
        // Should converge immediately
        assert_eq!(result.n_iter, 1);
    }
}
