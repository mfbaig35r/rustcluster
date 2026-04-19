//! Evaluation metrics for embedding clusters.
//!
//! All metrics operate on unit-normalized data and use dot products
//! (equivalent to cosine similarity for unit-norm vectors).

use crate::distance::Scalar;
use crate::embedding::spherical_kmeans::dot_product;
use rayon::prelude::*;

/// Per-cluster resultant length: ||mean(cluster_vectors)||.
/// Range [0, 1]: 0 = uniformly scattered, 1 = all identical direction.
/// Directly related to vMF concentration parameter κ.
pub fn resultant_lengths<F: Scalar>(
    data: &[F],
    labels: &[usize],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f64> {
    let mut sums = vec![0.0f64; k * d];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let label = labels[i];
        counts[label] += 1;
        for j in 0..d {
            sums[label * d + j] += data[i * d + j].to_f64_lossy();
        }
    }

    (0..k)
        .map(|c| {
            if counts[c] == 0 {
                return 0.0;
            }
            let norm_sq: f64 = (0..d).map(|j| {
                let mean = sums[c * d + j] / counts[c] as f64;
                mean * mean
            }).sum();
            norm_sq.sqrt()
        })
        .collect()
}

/// Per-cluster average cosine similarity to centroid.
/// For unit-norm data and centroids, this is just the average dot product.
pub fn intra_cluster_similarity<F: Scalar>(
    data: &[F],
    labels: &[usize],
    centroids: &[F],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f64> {
    let mut sums = vec![0.0f64; k];
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let label = labels[i];
        let point = &data[i * d..(i + 1) * d];
        let centroid = &centroids[label * d..(label + 1) * d];
        sums[label] += dot_product(point, centroid).to_f64_lossy();
        counts[label] += 1;
    }

    (0..k)
        .map(|c| if counts[c] > 0 { sums[c] / counts[c] as f64 } else { 0.0 })
        .collect()
}

/// Find the representative (nearest-to-centroid) point for each cluster.
/// Returns a Vec of original point indices.
pub fn find_representatives<F: Scalar>(
    data: &[F],
    labels: &[usize],
    centroids: &[F],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<usize> {
    let mut best_idx = vec![0usize; k];
    let mut best_sim = vec![f64::NEG_INFINITY; k];

    for i in 0..n {
        let label = labels[i];
        let point = &data[i * d..(i + 1) * d];
        let centroid = &centroids[label * d..(label + 1) * d];
        let sim = dot_product(point, centroid).to_f64_lossy();
        if sim > best_sim[label] {
            best_sim[label] = sim;
            best_idx[label] = i;
        }
    }

    best_idx
}

/// Cosine silhouette score for unit-normalized data.
/// Uses cosine distance (1 - dot) instead of Euclidean.
/// Optionally samples for large datasets.
pub fn cosine_silhouette<F: Scalar>(
    data: &[F],
    labels: &[usize],
    n: usize,
    d: usize,
    k: usize,
    sample_size: Option<usize>,
    seed: u64,
) -> f64 {
    if k < 2 || n < 2 {
        return 0.0;
    }

    // Determine which indices to evaluate
    let indices: Vec<usize> = match sample_size {
        Some(s) if s < n => {
            use rand::rngs::StdRng;
            use rand::seq::index::sample;
            use rand::SeedableRng;
            let mut rng = StdRng::seed_from_u64(seed);
            let sampled = sample(&mut rng, n, s);
            sampled.into_iter().collect()
        }
        _ => (0..n).collect(),
    };

    let silhouette_sum: f64 = indices
        .par_iter()
        .map(|&i| {
            let point = &data[i * d..(i + 1) * d];
            let my_label = labels[i];

            // Compute mean cosine distance to each cluster
            let mut cluster_dist_sum = vec![0.0f64; k];
            let mut cluster_count = vec![0usize; k];

            for j in 0..n {
                if i == j { continue; }
                let other = &data[j * d..(j + 1) * d];
                let cos_dist = 1.0 - dot_product(point, other).to_f64_lossy();
                cluster_dist_sum[labels[j]] += cos_dist;
                cluster_count[labels[j]] += 1;
            }

            // a = mean intra-cluster distance
            let a = if cluster_count[my_label] > 0 {
                cluster_dist_sum[my_label] / cluster_count[my_label] as f64
            } else {
                0.0
            };

            // b = min mean inter-cluster distance
            let b = (0..k)
                .filter(|&c| c != my_label && cluster_count[c] > 0)
                .map(|c| cluster_dist_sum[c] / cluster_count[c] as f64)
                .fold(f64::MAX, f64::min);

            if b == f64::MAX || a.max(b) == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            }
        })
        .sum();

    silhouette_sum / indices.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::normalize::l2_normalize_rows;

    #[test]
    fn test_resultant_length_tight_cluster() {
        // All points in same direction → resultant ≈ 1
        let data = vec![1.0f64, 0.0, 0.99, 0.14, 0.98, 0.20]; // roughly same direction
        let (norm_data, _) = l2_normalize_rows(&data, 3, 2);
        let labels = vec![0, 0, 0];
        let rl = resultant_lengths(&norm_data, &labels, 3, 2, 1);
        assert!(rl[0] > 0.95, "resultant = {}", rl[0]);
    }

    #[test]
    fn test_resultant_length_scattered() {
        // Opposite directions → resultant ≈ 0
        let data = vec![1.0f64, 0.0, -1.0, 0.0];
        let labels = vec![0, 0];
        let rl = resultant_lengths(&data, &labels, 2, 2, 1);
        assert!(rl[0] < 0.01, "resultant = {}", rl[0]);
    }

    #[test]
    fn test_representatives() {
        // 3 points, centroid is [1, 0]. Point 0 is closest.
        let data = vec![0.99f64, 0.14, 0.7, 0.7, 0.0, 1.0];
        let (norm_data, _) = l2_normalize_rows(&data, 3, 2);
        let centroids = vec![1.0f64, 0.0];
        let labels = vec![0, 0, 0];
        let reps = find_representatives(&norm_data, &labels, &centroids, 3, 2, 1);
        assert_eq!(reps[0], 0); // point [0.99, 0.14] normalized is closest to [1, 0]
    }

    #[test]
    fn test_intra_similarity_tight() {
        let data = vec![1.0f64, 0.0, 0.99, 0.14, 0.98, 0.20];
        let (norm_data, _) = l2_normalize_rows(&data, 3, 2);
        let centroids = vec![1.0f64, 0.0]; // unit-norm centroid
        let labels = vec![0, 0, 0];
        let sims = intra_cluster_similarity(&norm_data, &labels, &centroids, 3, 2, 1);
        assert!(sims[0] > 0.9, "sim = {}", sims[0]);
    }

    #[test]
    fn test_cosine_silhouette_well_separated() {
        // Two clusters: [1,0,0] direction and [0,1,0] direction
        let d = 3;
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(&[1.0, 0.0, 0.0]);
            labels.push(0);
        }
        for _ in 0..20 {
            data.extend_from_slice(&[0.0, 1.0, 0.0]);
            labels.push(1);
        }
        let score = cosine_silhouette(&data, &labels, 40, d, 2, None, 42);
        assert!(score > 0.9, "silhouette = {score}");
    }

    #[test]
    fn test_cosine_silhouette_single_cluster() {
        let data = vec![1.0f64, 0.0, 0.0, 1.0];
        let labels = vec![0, 0];
        let score = cosine_silhouette(&data, &labels, 2, 2, 1, None, 42);
        assert_eq!(score, 0.0);
    }
}
