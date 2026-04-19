//! Clustering evaluation metrics.
//!
//! All metrics operate on (data, labels) pairs and use Euclidean distance.
//! Noise points (label = -1) are excluded from computation.

use ndarray::ArrayView2;
use rayon::prelude::*;

use crate::distance::Scalar;
use crate::error::ClusterError;
use crate::utils::squared_euclidean_generic;

/// Silhouette score: mean silhouette coefficient across all samples.
///
/// For each point, computes:
///   a = mean distance to points in same cluster
///   b = mean distance to points in nearest other cluster
///   s = (b - a) / max(a, b)
///
/// Returns the mean of s across all points. Range: [-1, 1]. Higher is better.
/// Noise points (label < 0) are excluded.
/// Requires at least 2 clusters.
pub fn silhouette_score<F: Scalar>(
    data: &ArrayView2<F>,
    labels: &[i64],
) -> Result<f64, ClusterError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }
    if labels.len() != n {
        return Err(ClusterError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }

    // Filter out noise
    let valid_indices: Vec<usize> = (0..n).filter(|&i| labels[i] >= 0).collect();
    if valid_indices.is_empty() {
        return Err(ClusterError::EmptyInput);
    }

    let unique_labels: std::collections::HashSet<i64> =
        valid_indices.iter().map(|&i| labels[i]).collect();
    let n_clusters = unique_labels.len();
    if n_clusters < 2 {
        return Ok(0.0); // silhouette is undefined for 1 cluster, return 0
    }

    let data_slice = data.as_slice().expect("data must be C-contiguous");

    let silhouette_sum: f64 = valid_indices
        .par_iter()
        .map(|&i| {
            let point = &data_slice[i * d..(i + 1) * d];
            let my_label = labels[i];

            // Compute mean distance to each cluster
            let mut cluster_sums: std::collections::HashMap<i64, (f64, usize)> =
                std::collections::HashMap::new();

            for &j in &valid_indices {
                if i == j {
                    continue;
                }
                let other = &data_slice[j * d..(j + 1) * d];
                let dist = squared_euclidean_generic(point, other)
                    .to_f64_lossy()
                    .sqrt();
                let entry = cluster_sums.entry(labels[j]).or_insert((0.0, 0));
                entry.0 += dist;
                entry.1 += 1;
            }

            // a = mean intra-cluster distance
            let a = match cluster_sums.get(&my_label) {
                Some(&(sum, count)) if count > 0 => sum / count as f64,
                _ => 0.0,
            };

            // b = min mean inter-cluster distance
            let b = cluster_sums
                .iter()
                .filter(|(&label, _)| label != my_label)
                .map(|(_, &(sum, count))| {
                    if count > 0 {
                        sum / count as f64
                    } else {
                        f64::MAX
                    }
                })
                .fold(f64::MAX, f64::min);

            if a.max(b) == 0.0 {
                0.0
            } else {
                (b - a) / a.max(b)
            }
        })
        .sum();

    Ok(silhouette_sum / valid_indices.len() as f64)
}

/// Calinski-Harabasz index (Variance Ratio Criterion).
///
/// Ratio of between-cluster dispersion to within-cluster dispersion.
/// Higher is better. Requires at least 2 clusters.
pub fn calinski_harabasz_score<F: Scalar>(
    data: &ArrayView2<F>,
    labels: &[i64],
) -> Result<f64, ClusterError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }
    if labels.len() != n {
        return Err(ClusterError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }

    let valid_indices: Vec<usize> = (0..n).filter(|&i| labels[i] >= 0).collect();
    let n_valid = valid_indices.len();
    if n_valid == 0 {
        return Err(ClusterError::EmptyInput);
    }

    let data_slice = data.as_slice().expect("data must be C-contiguous");

    // Compute overall centroid
    let mut global_centroid = vec![0.0f64; d];
    for &i in &valid_indices {
        for j in 0..d {
            global_centroid[j] += data_slice[i * d + j].to_f64_lossy();
        }
    }
    for j in 0..d {
        global_centroid[j] /= n_valid as f64;
    }

    // Compute per-cluster centroids and counts
    let unique_labels: std::collections::BTreeSet<i64> =
        valid_indices.iter().map(|&i| labels[i]).collect();
    let n_clusters = unique_labels.len();
    if n_clusters < 2 {
        return Ok(0.0);
    }

    let mut cluster_centroids: std::collections::HashMap<i64, Vec<f64>> =
        std::collections::HashMap::new();
    let mut cluster_counts: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();

    for &label in &unique_labels {
        cluster_centroids.insert(label, vec![0.0; d]);
        cluster_counts.insert(label, 0);
    }

    for &i in &valid_indices {
        let label = labels[i];
        let centroid = cluster_centroids.get_mut(&label).unwrap();
        *cluster_counts.get_mut(&label).unwrap() += 1;
        for j in 0..d {
            centroid[j] += data_slice[i * d + j].to_f64_lossy();
        }
    }

    for (&label, centroid) in cluster_centroids.iter_mut() {
        let count = *cluster_counts.get(&label).unwrap() as f64;
        for j in 0..d {
            centroid[j] /= count;
        }
    }

    // Between-cluster dispersion: sum of n_k * ||c_k - c_global||^2
    let mut bg = 0.0f64;
    for (&label, centroid) in &cluster_centroids {
        let count = *cluster_counts.get(&label).unwrap() as f64;
        let mut sq_dist = 0.0;
        for j in 0..d {
            let diff = centroid[j] - global_centroid[j];
            sq_dist += diff * diff;
        }
        bg += count * sq_dist;
    }

    // Within-cluster dispersion: sum of ||x_i - c_k||^2
    let mut wg = 0.0f64;
    for &i in &valid_indices {
        let label = labels[i];
        let centroid = cluster_centroids.get(&label).unwrap();
        for j in 0..d {
            let diff = data_slice[i * d + j].to_f64_lossy() - centroid[j];
            wg += diff * diff;
        }
    }

    if wg == 0.0 {
        return Ok(f64::MAX); // perfect clustering
    }

    let k = n_clusters as f64;
    let n_f = n_valid as f64;
    Ok((bg / (k - 1.0)) / (wg / (n_f - k)))
}

/// Davies-Bouldin index.
///
/// Average similarity between each cluster and its most similar cluster.
/// Lower is better. Requires at least 2 clusters.
pub fn davies_bouldin_score<F: Scalar>(
    data: &ArrayView2<F>,
    labels: &[i64],
) -> Result<f64, ClusterError> {
    let (n, d) = data.dim();
    if n == 0 || d == 0 {
        return Err(ClusterError::EmptyInput);
    }
    if labels.len() != n {
        return Err(ClusterError::DimensionMismatch {
            expected: n,
            got: labels.len(),
        });
    }

    let valid_indices: Vec<usize> = (0..n).filter(|&i| labels[i] >= 0).collect();
    let data_slice = data.as_slice().expect("data must be C-contiguous");

    // Compute per-cluster centroids
    let unique_labels: Vec<i64> = {
        let set: std::collections::BTreeSet<i64> =
            valid_indices.iter().map(|&i| labels[i]).collect();
        set.into_iter().collect()
    };
    let n_clusters = unique_labels.len();
    if n_clusters < 2 {
        return Ok(0.0);
    }

    let mut cluster_centroids: std::collections::HashMap<i64, Vec<f64>> =
        std::collections::HashMap::new();
    let mut cluster_counts: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::new();

    for &label in &unique_labels {
        cluster_centroids.insert(label, vec![0.0; d]);
        cluster_counts.insert(label, 0);
    }

    for &i in &valid_indices {
        let label = labels[i];
        let centroid = cluster_centroids.get_mut(&label).unwrap();
        *cluster_counts.get_mut(&label).unwrap() += 1;
        for j in 0..d {
            centroid[j] += data_slice[i * d + j].to_f64_lossy();
        }
    }

    for (&label, centroid) in cluster_centroids.iter_mut() {
        let count = *cluster_counts.get(&label).unwrap() as f64;
        for j in 0..d {
            centroid[j] /= count;
        }
    }

    // Compute mean intra-cluster distance (scatter) for each cluster
    let mut scatters: std::collections::HashMap<i64, f64> = std::collections::HashMap::new();
    for &label in &unique_labels {
        let centroid = cluster_centroids.get(&label).unwrap();
        let count = *cluster_counts.get(&label).unwrap();
        let mut total = 0.0f64;
        for &i in &valid_indices {
            if labels[i] != label {
                continue;
            }
            let mut sq_dist = 0.0;
            for j in 0..d {
                let diff = data_slice[i * d + j].to_f64_lossy() - centroid[j];
                sq_dist += diff * diff;
            }
            total += sq_dist.sqrt();
        }
        scatters.insert(label, if count > 0 { total / count as f64 } else { 0.0 });
    }

    // For each cluster, find the maximum R_ij = (s_i + s_j) / d(c_i, c_j)
    let mut db_sum = 0.0f64;
    for (idx_i, &label_i) in unique_labels.iter().enumerate() {
        let s_i = *scatters.get(&label_i).unwrap();
        let c_i = cluster_centroids.get(&label_i).unwrap();

        let mut max_r = 0.0f64;
        for (idx_j, &label_j) in unique_labels.iter().enumerate() {
            if idx_i == idx_j {
                continue;
            }
            let s_j = *scatters.get(&label_j).unwrap();
            let c_j = cluster_centroids.get(&label_j).unwrap();

            let mut sq_dist = 0.0f64;
            for k in 0..d {
                let diff = c_i[k] - c_j[k];
                sq_dist += diff * diff;
            }
            let dist = sq_dist.sqrt();
            if dist > 0.0 {
                let r = (s_i + s_j) / dist;
                if r > max_r {
                    max_r = r;
                }
            }
        }
        db_sum += max_r;
    }

    Ok(db_sum / n_clusters as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn simple_clusters() -> (ndarray::Array2<f64>, Vec<i64>) {
        // Two tight clusters
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        (data, labels)
    }

    #[test]
    fn test_silhouette_well_separated() {
        let (data, labels) = simple_clusters();
        let score = silhouette_score(&data.view(), &labels).unwrap();
        assert!(score > 0.9); // should be close to 1 for well-separated clusters
    }

    #[test]
    fn test_silhouette_range() {
        let (data, labels) = simple_clusters();
        let score = silhouette_score(&data.view(), &labels).unwrap();
        assert!((-1.0..=1.0).contains(&score));
    }

    #[test]
    fn test_silhouette_single_cluster() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let labels = vec![0, 0];
        let score = silhouette_score(&data.view(), &labels).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_silhouette_excludes_noise() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [50.0, 50.0], // noise
        ];
        let labels = vec![0, 0, 1, 1, -1];
        let score = silhouette_score(&data.view(), &labels).unwrap();
        assert!(score > 0.9);
    }

    #[test]
    fn test_calinski_harabasz_well_separated() {
        let (data, labels) = simple_clusters();
        let score = calinski_harabasz_score(&data.view(), &labels).unwrap();
        assert!(score > 100.0); // high ratio for well-separated clusters
    }

    #[test]
    fn test_calinski_harabasz_single_cluster() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let labels = vec![0, 0];
        let score = calinski_harabasz_score(&data.view(), &labels).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_davies_bouldin_well_separated() {
        let (data, labels) = simple_clusters();
        let score = davies_bouldin_score(&data.view(), &labels).unwrap();
        assert!(score < 0.1); // low = good separation
    }

    #[test]
    fn test_davies_bouldin_single_cluster() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let labels = vec![0, 0];
        let score = davies_bouldin_score(&data.view(), &labels).unwrap();
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_empty_labels() {
        let data = ndarray::Array2::<f64>::zeros((0, 2));
        let labels: Vec<i64> = vec![];
        assert!(silhouette_score(&data.view(), &labels).is_err());
        assert!(calinski_harabasz_score(&data.view(), &labels).is_err());
        assert!(davies_bouldin_score(&data.view(), &labels).is_err());
    }

    #[test]
    fn test_f32_silhouette() {
        let data = array![[0.0f32, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0],];
        let labels = vec![0, 0, 1, 1];
        let score = silhouette_score(&data.view(), &labels).unwrap();
        assert!(score > 0.9);
    }
}
