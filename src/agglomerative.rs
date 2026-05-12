//! Agglomerative (hierarchical) clustering.
//!
//! Builds a dendrogram bottom-up by merging the two nearest clusters at each step.
//! Supports Ward, complete, average, and single linkage.
//!
//! Complexity: O(n^2 log n) with priority queue.
//!
//! Memory: the pairwise distance matrix is stored condensed (upper triangle,
//! n*(n-1)/2 entries) in the input dtype F. A priority queue of all initial
//! pairs (O(n^2)) drives merge selection; a future NN-chain rewrite (Müllner
//! 2011) will replace it with O(n) extra memory.

use ndarray::{Array2, ArrayView2};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::distance::{
    CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean,
};
use crate::error::ClusterError;
use crate::utils::validate_data_generic;

/// Index into a condensed upper-triangular distance matrix for the pair (i, j), i != j.
///
/// The condensed layout stores n*(n-1)/2 entries enumerated as
/// (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1) — matching scipy's
/// `scipy.spatial.distance.pdist` ordering.
#[inline(always)]
fn cd_idx(i: usize, j: usize, n: usize) -> usize {
    debug_assert!(i != j);
    debug_assert!(i < n && j < n);
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * n - i * (i + 1) / 2 + (j - i - 1)
}

/// Result of a fitted agglomerative model.
pub struct AgglomerativeState<F: Scalar> {
    pub labels: Vec<i64>,
    pub n_clusters: usize,
    pub children: Vec<(usize, usize)>, // merge history: (n-1) entries
    pub distances: Vec<f64>,           // distance at each merge
    pub _phantom: std::marker::PhantomData<F>,
}

/// Linkage method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Linkage {
    Ward,
    Complete,
    Average,
    Single,
}

impl std::str::FromStr for Linkage {
    type Err = ClusterError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ward" => Ok(Linkage::Ward),
            "complete" => Ok(Linkage::Complete),
            "average" => Ok(Linkage::Average),
            "single" => Ok(Linkage::Single),
            _ => Err(ClusterError::InvalidLinkage(s.to_string())),
        }
    }
}

// ---- Public entry points ----

pub fn run_agglomerative_with_metric(
    data: &ArrayView2<f64>,
    n_clusters: usize,
    linkage: Linkage,
    metric: Metric,
) -> Result<AgglomerativeState<f64>, ClusterError> {
    if linkage == Linkage::Ward && metric != Metric::Euclidean {
        return Err(ClusterError::WardRequiresEuclidean);
    }
    match metric {
        Metric::Euclidean => {
            run_agglomerative_generic::<f64, SquaredEuclidean>(data, n_clusters, linkage)
        }
        Metric::Cosine => {
            run_agglomerative_generic::<f64, CosineDistance>(data, n_clusters, linkage)
        }
        Metric::Manhattan => {
            run_agglomerative_generic::<f64, ManhattanDistance>(data, n_clusters, linkage)
        }
    }
}

pub fn run_agglomerative_with_metric_f32(
    data: &ArrayView2<f32>,
    n_clusters: usize,
    linkage: Linkage,
    metric: Metric,
) -> Result<AgglomerativeState<f32>, ClusterError> {
    if linkage == Linkage::Ward && metric != Metric::Euclidean {
        return Err(ClusterError::WardRequiresEuclidean);
    }
    match metric {
        Metric::Euclidean => {
            run_agglomerative_generic::<f32, SquaredEuclidean>(data, n_clusters, linkage)
        }
        Metric::Cosine => {
            run_agglomerative_generic::<f32, CosineDistance>(data, n_clusters, linkage)
        }
        Metric::Manhattan => {
            run_agglomerative_generic::<f32, ManhattanDistance>(data, n_clusters, linkage)
        }
    }
}

// ---- Generic implementation ----

fn run_agglomerative_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    target_n_clusters: usize,
    linkage: Linkage,
) -> Result<AgglomerativeState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();
    if target_n_clusters == 0 || target_n_clusters > n {
        return Err(ClusterError::InvalidClusters {
            k: target_n_clusters,
            n,
        });
    }

    let data_slice = data.as_slice().expect("data must be C-contiguous");

    // Pairwise distance matrix in condensed upper-triangular form
    // (n*(n-1)/2 entries), stored in the input dtype F.
    //
    // For Ward, store squared Euclidean; for others, store the metric distance.
    // The f64 round trip preserves sqrt() precision when F = f32.
    let cm_len = n.checked_mul(n - 1).expect("n*(n-1) overflow in dist matrix size") / 2;
    let mut dist_matrix: Vec<F> = vec![F::zero(); cm_len];
    for i in 0..n {
        let pi = &data_slice[i * d..(i + 1) * d];
        for j in (i + 1)..n {
            let pj = &data_slice[j * d..(j + 1) * d];
            let raw = D::distance(pi, pj).to_f64_lossy();
            let dist = if matches!(linkage, Linkage::Ward) {
                raw // SquaredEuclidean — Lance-Williams expects squared distances
            } else {
                D::to_metric(raw)
            };
            dist_matrix[cd_idx(i, j, n)] = F::from_f64_lossy(dist);
        }
    }

    // Cluster tracking
    let mut cluster_size = vec![1usize; n];
    let mut active = vec![true; n]; // which clusters are still active
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n - 1);
    let mut merge_distances: Vec<f64> = Vec::with_capacity(n - 1);

    // Label mapping: original cluster i -> current cluster id
    // After merges, new clusters get IDs n, n+1, ...
    let mut cluster_id = vec![0usize; n];
    for i in 0..n {
        cluster_id[i] = i;
    }
    let mut next_id = n;

    // Priority queue: (distance, gen_i, gen_j, i, j) — skip stale entries by generation.
    // Phase 2 (NN-chain) will replace this with O(n) extra memory.
    let mut generation = vec![0u32; n];
    let mut heap: BinaryHeap<Reverse<(FloatOrd, u32, u32, usize, usize)>> = BinaryHeap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            heap.push(Reverse((
                FloatOrd(dist_matrix[cd_idx(i, j, n)].to_f64_lossy()),
                0,
                0,
                i,
                j,
            )));
        }
    }

    // Track which slot each point belongs to. When cj merges into ci,
    // all points in cj now belong to ci.
    let mut membership: Vec<usize> = (0..n).collect();

    let n_merges_needed = n - target_n_clusters;

    for _ in 0..n_merges_needed {
        // Pop the minimum-distance pair that is active and current-generation
        let (ci, cj, merge_dist) = loop {
            let Reverse((FloatOrd(dist), gi, gj, i, j)) =
                heap.pop().expect("heap empty before done");
            if active[i] && active[j] && gi == generation[i] && gj == generation[j] {
                break (i, j, dist);
            }
        };

        // Record merge
        children.push((cluster_id[ci], cluster_id[cj]));
        let report_dist = if matches!(linkage, Linkage::Ward) {
            merge_dist.sqrt() // report actual Euclidean distance
        } else {
            merge_dist
        };
        merge_distances.push(report_dist);

        // Merge cj into ci
        active[cj] = false;
        // Forward cj's membership to ci
        membership[cj] = ci;
        let new_size = cluster_size[ci] + cluster_size[cj];

        // Update distances from the merged cluster to all other active clusters.
        // Lance-Williams arithmetic in f64 for numerical stability (matches the
        // k-means centroid-accumulation convention); result cast back to F.
        for k in 0..n {
            if !active[k] || k == ci {
                continue;
            }
            let d_ci_k = dist_matrix[cd_idx(ci, k, n)].to_f64_lossy();
            let d_cj_k = dist_matrix[cd_idx(cj, k, n)].to_f64_lossy();
            let n_i = cluster_size[ci] as f64;
            let n_j = cluster_size[cj] as f64;
            let n_k = cluster_size[k] as f64;

            let new_dist = match linkage {
                Linkage::Ward => {
                    // Lance-Williams for Ward (on squared distances)
                    let n_total = n_i + n_j + n_k;
                    ((n_i + n_k) * d_ci_k + (n_j + n_k) * d_cj_k - n_k * merge_dist) / n_total
                }
                Linkage::Complete => d_ci_k.max(d_cj_k),
                Linkage::Single => d_ci_k.min(d_cj_k),
                Linkage::Average => (n_i * d_ci_k + n_j * d_cj_k) / (n_i + n_j),
            };

            dist_matrix[cd_idx(ci, k, n)] = F::from_f64_lossy(new_dist);

            // Push updated distance with current generations
            heap.push(Reverse((
                FloatOrd(new_dist),
                generation[ci] + 1,
                generation[k],
                ci,
                k,
            )));
        }

        cluster_size[ci] = new_size;
        cluster_id[ci] = next_id;
        next_id += 1;
        generation[ci] += 1; // increment so old entries for ci are skipped
    }

    // Assign labels using the membership tracker built during merges.
    // membership[i] = the active slot that point i currently belongs to.
    let mut slot_label: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
    let mut next_label = 0i64;
    for i in 0..n {
        if active[i] {
            slot_label.insert(i, next_label);
            next_label += 1;
        }
    }

    let mut labels = vec![0i64; n];
    for i in 0..n {
        // Follow the membership chain to find the active slot
        let mut slot = membership[i];
        while !active[slot] {
            slot = membership[slot];
        }
        labels[i] = *slot_label.get(&slot).unwrap();
    }

    Ok(AgglomerativeState {
        labels,
        n_clusters: target_n_clusters,
        children,
        distances: merge_distances,
        _phantom: std::marker::PhantomData,
    })
}

/// Wrapper for f64 to implement Ord for BinaryHeap.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FloatOrd(f64);

impl Eq for FloatOrd {}

impl PartialOrd for FloatOrd {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FloatOrd {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_two_clusters() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels.len(), 6);
        assert_eq!(result.n_clusters, 2);
        // First 3 should share a label, last 3 another
        let c1 = result.labels[0];
        let c2 = result.labels[3];
        assert_ne!(c1, c2);
        for i in 0..3 {
            assert_eq!(result.labels[i], c1);
        }
        for i in 3..6 {
            assert_eq!(result.labels[i], c2);
        }
    }

    #[test]
    fn test_single_cluster() {
        let data = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 1, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_n_equals_n_clusters() {
        let data = array![[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 3, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        let mut sorted = result.labels.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_children_length() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        // n=4, target=2 → 2 merges
        assert_eq!(result.children.len(), 2);
        assert_eq!(result.distances.len(), 2);
    }

    #[test]
    fn test_distances_non_decreasing() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0], [20.0, 0.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 1, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        for w in result.distances.windows(2) {
            assert!(w[1] >= w[0] - 1e-10);
        }
    }

    #[test]
    fn test_complete_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Complete, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_average_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Average, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_single_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Single, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_ward_requires_euclidean() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), 1, Linkage::Ward, Metric::Cosine),
            Err(ClusterError::WardRequiresEuclidean)
        ));
    }

    #[test]
    fn test_manhattan_metric() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), 2, Linkage::Complete, Metric::Manhattan)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
    }

    #[test]
    fn test_f32() {
        let data = array![[0.0f32, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0],];
        let result =
            run_agglomerative_with_metric_f32(&data.view(), 2, Linkage::Ward, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_invalid_n_clusters() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), 5, Linkage::Ward, Metric::Euclidean),
            Err(ClusterError::InvalidClusters { .. })
        ));
    }

    #[test]
    fn test_empty_input() {
        let data = Array2::<f64>::zeros((0, 2));
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), 1, Linkage::Ward, Metric::Euclidean),
            Err(ClusterError::EmptyInput)
        ));
    }

    #[test]
    fn test_deterministic() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
        let r1 = run_agglomerative_with_metric(&data.view(), 2, Linkage::Ward, Metric::Euclidean)
            .unwrap();
        let r2 = run_agglomerative_with_metric(&data.view(), 2, Linkage::Ward, Metric::Euclidean)
            .unwrap();
        assert_eq!(r1.labels, r2.labels);
    }

    #[test]
    fn test_cd_idx_scipy_pdist_order() {
        // scipy.spatial.distance.pdist enumerates pairs in row-major upper-triangle
        // order: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1). The condensed
        // index function must match this exactly so callers can interop with
        // scipy's linkage matrix format in later phases.
        let n = 5;
        let mut idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(cd_idx(i, j, n), idx, "(i={i}, j={j})");
                // Symmetric: same index regardless of argument order
                assert_eq!(cd_idx(j, i, n), idx, "(j={j}, i={i}) symmetric");
                idx += 1;
            }
        }
        assert_eq!(idx, n * (n - 1) / 2);
    }

    #[test]
    fn test_larger_n_average_cosine_f32() {
        // Exercise the condensed matrix + f32 storage on a non-trivial n with
        // cosine distance — the path that the supplier-clustering workload hits.
        // Three well-separated clusters of 10 points each on the unit circle.
        let n_per = 10;
        let mut rows: Vec<[f32; 2]> = Vec::with_capacity(3 * n_per);
        let centers = [(0.0f32, 1.0f32), (1.0, 0.0), (-0.7, -0.7)];
        for (cx, cy) in centers {
            for k in 0..n_per {
                let jitter = (k as f32) * 1e-4;
                rows.push([cx + jitter, cy - jitter]);
            }
        }
        let data = ndarray::Array2::from_shape_vec((rows.len(), 2), rows.concat()).unwrap();
        let result = run_agglomerative_with_metric_f32(
            &data.view(),
            3,
            Linkage::Average,
            Metric::Cosine,
        )
        .unwrap();
        assert_eq!(result.n_clusters, 3);
        // Each block of n_per consecutive rows should share a label
        for block in 0..3 {
            let base = result.labels[block * n_per];
            for k in 1..n_per {
                assert_eq!(
                    result.labels[block * n_per + k],
                    base,
                    "block {block} row {k} mislabeled"
                );
            }
        }
        // Distances must be non-decreasing (reducibility property)
        for w in result.distances.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-5,
                "non-monotonic merge distances: {:?}",
                result.distances
            );
        }
    }
}
