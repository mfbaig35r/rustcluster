use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::WeightedIndex;
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::distance::{CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean};
use crate::error::ClusterError;
use crate::utils::{assign_nearest_with, validate_data_generic};

/// Result of a fitted K-means model, generic over float type.
pub struct KMeansState<F: Scalar> {
    pub centroids: Array2<F>,
    pub labels: Vec<usize>,
    pub inertia: f64, // always f64 for consistency at the Python boundary
    pub n_iter: usize,
}

/// Algorithm selection for K-means.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    Auto,
    Lloyd,
    Hamerly,
}

impl Algorithm {
    pub fn from_str(s: &str) -> Result<Self, ClusterError> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Algorithm::Auto),
            "lloyd" => Ok(Algorithm::Lloyd),
            "hamerly" => Ok(Algorithm::Hamerly),
            _ => Err(ClusterError::InvalidAlgorithm(s.to_string())),
        }
    }

    pub fn resolve(self, d: usize, k: usize) -> Algorithm {
        match self {
            Algorithm::Auto => {
                if k < 2 {
                    Algorithm::Lloyd
                } else if d <= 16 && k <= 32 {
                    Algorithm::Lloyd
                } else {
                    Algorithm::Hamerly
                }
            }
            other => other,
        }
    }
}

// ---- Public entry points (f64, SquaredEuclidean — for Python API) ----

/// Run K-means with f64 and squared Euclidean distance.
pub fn run_kmeans_n_init(
    data: &ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
) -> Result<KMeansState<f64>, ClusterError> {
    run_kmeans_n_init_generic::<f64, SquaredEuclidean>(data, k, max_iter, tol, seed, n_init, algo)
}

/// Run K-means with f32 and squared Euclidean distance.
pub fn run_kmeans_n_init_f32(
    data: &ArrayView2<f32>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
) -> Result<KMeansState<f32>, ClusterError> {
    run_kmeans_n_init_generic::<f32, SquaredEuclidean>(data, k, max_iter, tol, seed, n_init, algo)
}

/// Run K-means with runtime metric selection (f64).
/// Cosine distance forces Lloyd (Hamerly assumes Euclidean bounds).
pub fn run_kmeans_with_metric(
    data: &ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
    metric: Metric,
) -> Result<KMeansState<f64>, ClusterError> {
    match metric {
        Metric::Euclidean => run_kmeans_n_init_generic::<f64, SquaredEuclidean>(data, k, max_iter, tol, seed, n_init, algo),
        Metric::Cosine => run_kmeans_n_init_generic::<f64, CosineDistance>(data, k, max_iter, tol, seed, n_init, Algorithm::Lloyd),
        Metric::Manhattan => run_kmeans_n_init_generic::<f64, ManhattanDistance>(data, k, max_iter, tol, seed, n_init, Algorithm::Lloyd),
    }
}

/// Run K-means with runtime metric selection (f32).
pub fn run_kmeans_with_metric_f32(
    data: &ArrayView2<f32>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
    metric: Metric,
) -> Result<KMeansState<f32>, ClusterError> {
    match metric {
        Metric::Euclidean => run_kmeans_n_init_generic::<f32, SquaredEuclidean>(data, k, max_iter, tol, seed, n_init, algo),
        Metric::Cosine => run_kmeans_n_init_generic::<f32, CosineDistance>(data, k, max_iter, tol, seed, n_init, Algorithm::Lloyd),
        Metric::Manhattan => run_kmeans_n_init_generic::<f32, ManhattanDistance>(data, k, max_iter, tol, seed, n_init, Algorithm::Lloyd),
    }
}

// ---- Generic implementation ----

/// Run K-means with `n_init` initializations, generic over float type and distance.
pub fn run_kmeans_n_init_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
) -> Result<KMeansState<F>, ClusterError> {
    let mut best: Option<KMeansState<F>> = None;

    for i in 0..n_init {
        let run_seed = seed.wrapping_add(i as u64);
        let result = run_kmeans_single::<F, D>(data, k, max_iter, tol, run_seed, algo)?;

        let is_better = match &best {
            None => true,
            Some(prev) => result.inertia < prev.inertia,
        };

        if is_better {
            best = Some(result);
        }
    }

    Ok(best.unwrap())
}

fn run_kmeans_single<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    algo: Algorithm,
) -> Result<KMeansState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();
    if k == 0 || k > n {
        return Err(ClusterError::InvalidClusters { k, n });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeans_plus_plus_init::<F, D>(data, k, &mut rng);
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let resolved = algo.resolve(d, k);
    let resolved = if resolved == Algorithm::Hamerly && k < 2 {
        Algorithm::Lloyd
    } else {
        resolved
    };

    match resolved {
        Algorithm::Lloyd => run_lloyd_iterations::<F, D>(data_slice, &mut centroids, n, d, k, max_iter, tol, &mut rng),
        Algorithm::Hamerly => crate::hamerly::run_hamerly_iterations::<F, D>(data_slice, &mut centroids, n, d, k, max_iter, tol, &mut rng),
        Algorithm::Auto => unreachable!(),
    }
}

/// Lloyd's iteration loop, generic over float type and distance.
fn run_lloyd_iterations<F: Scalar, D: Distance<F>>(
    data_slice: &[F],
    centroids: &mut Array2<F>,
    n: usize,
    d: usize,
    k: usize,
    max_iter: usize,
    tol: f64,
    rng: &mut StdRng,
) -> Result<KMeansState<F>, ClusterError> {
    let mut labels = vec![0usize; n];
    let mut inertia = f64::MAX;
    let mut n_iter = 0;

    let mut old_centroid_buf = vec![F::zero(); k * d];

    for iter in 0..max_iter {
        let centroids_slice = centroids.as_slice().expect("centroids are C-contiguous");

        let assignments: Vec<(usize, F)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let point = &data_slice[i * d..(i + 1) * d];
                assign_nearest_with::<F, D>(point, centroids_slice, k, d)
            })
            .collect();

        let mut new_inertia = 0.0f64;
        for (i, &(label, dist)) in assignments.iter().enumerate() {
            labels[i] = label;
            new_inertia += dist.to_f64_lossy();
        }
        inertia = new_inertia;

        old_centroid_buf.copy_from_slice(centroids_slice);

        recompute_centroids(data_slice, &labels, centroids, n, d, k);
        handle_empty_clusters(data_slice, &labels, centroids, &assignments, n, d, k, rng);

        n_iter = iter + 1;

        // Convergence check
        let new_slice = centroids.as_slice().expect("centroids are C-contiguous");
        let mut max_shift = 0.0f64;
        for cluster in 0..k {
            let start = cluster * d;
            let end = start + d;
            let shift = D::distance(&old_centroid_buf[start..end], &new_slice[start..end]).to_f64_lossy();
            if shift > max_shift {
                max_shift = shift;
            }
        }
        if max_shift < tol {
            break;
        }
    }

    Ok(KMeansState {
        centroids: centroids.clone(),
        labels,
        inertia,
        n_iter,
    })
}

// ---- Shared helpers ----

/// K-means++ initialization, generic.
pub fn kmeans_plus_plus_init<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    k: usize,
    rng: &mut StdRng,
) -> Array2<F> {
    let (n, d) = data.dim();
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let mut centroids = Array2::<F>::zeros((k, d));

    let first = rng.gen_range(0..n);
    let first_row = &data_slice[first * d..(first + 1) * d];
    centroids.row_mut(0).assign(&ndarray::ArrayView1::from(first_row));

    let mut min_dists = vec![f64::MAX; n]; // sampling uses f64 for precision

    for c in 1..k {
        let last_centroid = centroids.row(c - 1);
        let last_slice = last_centroid.as_slice().expect("centroid row is contiguous");

        for i in 0..n {
            let point = &data_slice[i * d..(i + 1) * d];
            let dist = D::distance(point, last_slice).to_f64_lossy();
            if dist < min_dists[i] {
                min_dists[i] = dist;
            }
        }

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

/// Recompute centroids as mean of assigned points.
pub(crate) fn recompute_centroids<F: Scalar>(
    data_slice: &[F],
    labels: &[usize],
    centroids: &mut Array2<F>,
    n: usize,
    d: usize,
    k: usize,
) {
    // Accumulate in f64 for precision, then convert back
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let label = labels[i];
        counts[label] += 1;
        let point_start = i * d;
        let sum_start = label * d;
        for j in 0..d {
            sums[sum_start + j] += data_slice[point_start + j].to_f64_lossy();
        }
    }

    let centroid_slice = centroids.as_slice_mut().expect("centroids are C-contiguous");
    for cluster in 0..k {
        if counts[cluster] > 0 {
            let count = counts[cluster] as f64;
            let start = cluster * d;
            for j in 0..d {
                centroid_slice[start + j] = F::from_f64_lossy(sums[start + j] / count);
            }
        }
    }
}

/// Re-seed empty clusters with the point farthest from its assigned centroid.
pub(crate) fn handle_empty_clusters<F: Scalar>(
    data_slice: &[F],
    labels: &[usize],
    centroids: &mut Array2<F>,
    assignments: &[(usize, F)],
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

/// Compute per-centroid shifts, generic.
pub(crate) fn compute_centroid_shifts<F: Scalar, D: Distance<F>>(
    old: &Array2<F>,
    new: &Array2<F>,
    k: usize,
    d: usize,
) -> Vec<f64> {
    let old_slice = old.as_slice().expect("old centroids are contiguous");
    let new_slice = new.as_slice().expect("new centroids are contiguous");

    let mut shifts = vec![0.0f64; k];
    for cluster in 0..k {
        let start = cluster * d;
        let end = start + d;
        shifts[cluster] = D::distance(&old_slice[start..end], &new_slice[start..end]).to_f64_lossy();
    }
    shifts
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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

    fn well_separated_data_f32() -> Array2<f32> {
        array![
            [0.0f32, 0.0],
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
        let centroids = kmeans_plus_plus_init::<f64, SquaredEuclidean>(&data.view(), 2, &mut rng);
        assert_eq!(centroids.shape(), &[2, 2]);
    }

    #[test]
    fn test_kmeans_plus_plus_picks_from_data() {
        let data = well_separated_data();
        let mut rng = StdRng::seed_from_u64(42);
        let centroids = kmeans_plus_plus_init::<f64, SquaredEuclidean>(&data.view(), 2, &mut rng);

        for c in 0..2 {
            let centroid = centroids.row(c);
            let found = data.rows().into_iter().any(|row| {
                row.iter().zip(centroid.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
            });
            assert!(found, "Centroid {} not found in input data", c);
        }
    }

    #[test]
    fn test_lloyd_converges_on_separated_clusters() {
        let data = well_separated_data();
        let result = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();

        assert_eq!(result.labels.len(), 8);
        assert_eq!(result.centroids.shape(), &[2, 2]);
        assert!(result.inertia >= 0.0);
        assert!(result.n_iter > 0);

        let label_a = result.labels[0];
        let label_b = result.labels[4];
        assert_ne!(label_a, label_b);
        for i in 0..4 { assert_eq!(result.labels[i], label_a); }
        for i in 4..8 { assert_eq!(result.labels[i], label_b); }
    }

    #[test]
    fn test_reproducibility() {
        let data = well_separated_data();
        let r1 = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        let r2 = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert!((r1.inertia - r2.inertia).abs() < 1e-10);
    }

    #[test]
    fn test_n_init_selects_best() {
        let data = well_separated_data();
        let single = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        let multi = run_kmeans_n_init(&data.view(), 2, 100, 1e-4, 42, 10, Algorithm::Lloyd).unwrap();
        assert!(multi.inertia <= single.inertia + 1e-10);
    }

    #[test]
    fn test_k_greater_than_n_fails() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let result = run_kmeans_n_init(&data.view(), 5, 100, 1e-4, 42, 1, Algorithm::Lloyd);
        assert!(matches!(result, Err(ClusterError::InvalidClusters { .. })));
    }

    #[test]
    fn test_empty_input_fails() {
        let data = Array2::<f64>::zeros((0, 2));
        let result = run_kmeans_n_init(&data.view(), 1, 100, 1e-4, 42, 1, Algorithm::Lloyd);
        assert!(matches!(result, Err(ClusterError::EmptyInput)));
    }

    #[test]
    fn test_single_cluster() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = run_kmeans_n_init(&data.view(), 1, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
        let center = result.centroids.row(0);
        assert!((center[0] - 3.0).abs() < 1e-10);
        assert!((center[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_k_equals_n() {
        let data = array![[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]];
        let result = run_kmeans_n_init(&data.view(), 3, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        let mut sorted = result.labels.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
        assert!(result.inertia < 1e-10);
    }

    #[test]
    fn test_identical_points() {
        let data = array![[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]];
        let result = run_kmeans_n_init(&data.view(), 1, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
        assert!(result.inertia < 1e-10);
        assert_eq!(result.n_iter, 1);
    }

    #[test]
    fn test_algorithm_resolve() {
        assert_eq!(Algorithm::Auto.resolve(8, 16), Algorithm::Lloyd);
        assert_eq!(Algorithm::Auto.resolve(32, 64), Algorithm::Hamerly);
        assert_eq!(Algorithm::Lloyd.resolve(32, 64), Algorithm::Lloyd);
        assert_eq!(Algorithm::Hamerly.resolve(2, 2), Algorithm::Hamerly);
        assert_eq!(Algorithm::Auto.resolve(8, 1), Algorithm::Lloyd);
    }

    #[test]
    fn test_compute_centroid_shifts() {
        let old = array![[0.0, 0.0], [10.0, 10.0]];
        let new = array![[1.0, 0.0], [10.0, 10.0]];
        let shifts = compute_centroid_shifts::<f64, SquaredEuclidean>(&old, &new, 2, 2);
        assert!((shifts[0] - 1.0).abs() < 1e-10);
        assert!((shifts[1] - 0.0).abs() < 1e-10);
    }

    // ---- f32 tests ----

    #[test]
    fn test_f32_lloyd_converges() {
        let data = well_separated_data_f32();
        let result = run_kmeans_n_init_f32(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        assert_eq!(result.labels.len(), 8);
        let label_a = result.labels[0];
        let label_b = result.labels[4];
        assert_ne!(label_a, label_b);
    }

    #[test]
    fn test_f32_hamerly_converges() {
        let data = well_separated_data_f32();
        let result = run_kmeans_n_init_f32(&data.view(), 2, 100, 1e-4, 42, 1, Algorithm::Hamerly).unwrap();
        assert_eq!(result.labels.len(), 8);
        let label_a = result.labels[0];
        let label_b = result.labels[4];
        assert_ne!(label_a, label_b);
    }

    #[test]
    fn test_f32_matches_f64_partition() {
        let data_f64 = well_separated_data();
        let data_f32 = well_separated_data_f32();
        let r64 = run_kmeans_n_init(&data_f64.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        let r32 = run_kmeans_n_init_f32(&data_f32.view(), 2, 100, 1e-4, 42, 1, Algorithm::Lloyd).unwrap();
        // Same partitioning on well-separated data
        assert_eq!(r64.labels, r32.labels);
    }
}
