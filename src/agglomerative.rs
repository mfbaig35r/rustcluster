//! Agglomerative (hierarchical) clustering.
//!
//! Builds a dendrogram bottom-up by merging the two nearest clusters at each
//! step using the nearest-neighbor chain algorithm (Müllner 2011, *Modern
//! hierarchical, agglomerative clustering algorithms*, arXiv:1109.2378).
//! Supports Ward, complete, average, and single linkage — all reducible
//! linkages for which NN-chain produces the same dendrogram as the naive
//! priority-queue algorithm.
//!
//! Complexity: O(n^2) time, O(n^2 / 2) memory for the condensed pairwise
//! distance matrix in the input dtype F, plus O(n) auxiliary state for the
//! chain stack, active set, and cluster sizes. No priority queue.

use ndarray::{Array2, ArrayView2};

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

/// How to cut the dendrogram to produce flat clusters.
///
/// `NClusters(k)` stops at exactly k clusters (applies n-k lowest-distance
/// merges). `DistanceThreshold(t)` applies every merge whose reported
/// distance is strictly less than `t` — matching sklearn's
/// `AgglomerativeClustering(distance_threshold=t, n_clusters=None)` semantics.
#[derive(Debug, Clone, Copy)]
pub enum Cut {
    NClusters(usize),
    DistanceThreshold(f64),
}

// ---- Public entry points ----

pub fn run_agglomerative_with_metric(
    data: &ArrayView2<f64>,
    cut: Cut,
    linkage: Linkage,
    metric: Metric,
) -> Result<AgglomerativeState<f64>, ClusterError> {
    if linkage == Linkage::Ward && metric != Metric::Euclidean {
        return Err(ClusterError::WardRequiresEuclidean);
    }
    match metric {
        Metric::Euclidean => {
            run_agglomerative_generic::<f64, SquaredEuclidean>(data, cut, linkage)
        }
        Metric::Cosine => run_agglomerative_generic::<f64, CosineDistance>(data, cut, linkage),
        Metric::Manhattan => {
            run_agglomerative_generic::<f64, ManhattanDistance>(data, cut, linkage)
        }
    }
}

pub fn run_agglomerative_with_metric_f32(
    data: &ArrayView2<f32>,
    cut: Cut,
    linkage: Linkage,
    metric: Metric,
) -> Result<AgglomerativeState<f32>, ClusterError> {
    if linkage == Linkage::Ward && metric != Metric::Euclidean {
        return Err(ClusterError::WardRequiresEuclidean);
    }
    match metric {
        Metric::Euclidean => {
            run_agglomerative_generic::<f32, SquaredEuclidean>(data, cut, linkage)
        }
        Metric::Cosine => run_agglomerative_generic::<f32, CosineDistance>(data, cut, linkage),
        Metric::Manhattan => {
            run_agglomerative_generic::<f32, ManhattanDistance>(data, cut, linkage)
        }
    }
}

// ---- Generic implementation ----

fn run_agglomerative_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    cut: Cut,
    linkage: Linkage,
) -> Result<AgglomerativeState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();

    // Validate the cut spec before doing any work.
    match cut {
        Cut::NClusters(k) => {
            if k == 0 || k > n {
                return Err(ClusterError::InvalidClusters { k, n });
            }
        }
        Cut::DistanceThreshold(t) => {
            if !t.is_finite() {
                return Err(ClusterError::InvalidDistanceThreshold(t));
            }
        }
    }

    // Edge case: single point, no merges possible. The cut spec is moot.
    if n == 1 {
        return Ok(AgglomerativeState {
            labels: vec![0i64],
            n_clusters: 1,
            children: Vec::new(),
            distances: Vec::new(),
            _phantom: std::marker::PhantomData,
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

    // ---- NN-chain phase: build the full dendrogram (n-1 merges) ----
    //
    // Each "slot" 0..n holds an active cluster. Merges always keep the lower-
    // index slot and deactivate the higher one, so every cluster is reachable
    // by a stable index in [0, n) without slot reallocation. The chain stack
    // walks reciprocal-nearest-neighbor pairs (Müllner 2011, §3.2).
    //
    // Merges are produced in CHAIN ORDER, which is monotonic in distance
    // WITHIN a single chain run but may jump when the chain empties and
    // restarts on a different cluster. The next phase sorts by distance.
    let mut size = vec![1usize; n];
    let mut active = vec![true; n];
    let mut chain: Vec<usize> = Vec::with_capacity(n);
    let mut raw_merges: Vec<(usize, usize, f64)> = Vec::with_capacity(n - 1); // (lo, hi, dist)

    for _ in 0..(n - 1) {
        if chain.is_empty() {
            let start = (0..n)
                .find(|&i| active[i])
                .expect("active clusters remain but none found");
            chain.push(start);
        }

        // Extend the chain until we find a reciprocal nearest neighbor pair.
        loop {
            let a = *chain.last().unwrap();
            let prev = if chain.len() >= 2 {
                Some(chain[chain.len() - 2])
            } else {
                None
            };

            // Find nearest active cluster to a, with deterministic tie-breaking:
            // (1) smallest distance wins; (2) on ties, smallest slot index wins;
            // (3) on a tie with `prev` (the cluster that previously chose a),
            //     prefer `prev` so an RNN pair closes instead of cycling
            //     through tied neighbors. This is the standard NN-chain tie-
            //     break rule from Müllner §3.2.
            let mut best_k = usize::MAX;
            let mut best_d = f64::INFINITY;
            for k in 0..n {
                if k == a || !active[k] {
                    continue;
                }
                let dk = dist_matrix[cd_idx(a, k, n)].to_f64_lossy();
                if dk < best_d || (dk == best_d && k < best_k) {
                    best_d = dk;
                    best_k = k;
                }
            }
            if let Some(p) = prev {
                if active[p] && p != a {
                    let dp = dist_matrix[cd_idx(a, p, n)].to_f64_lossy();
                    if dp <= best_d {
                        best_d = dp;
                        best_k = p;
                    }
                }
            }
            let b = best_k;

            // RNN closure: if `b` is the cluster that put `a` on the chain,
            // we have a reciprocal nearest neighbor pair and can merge.
            if Some(b) == prev {
                chain.pop(); // a
                chain.pop(); // b

                // Lower index survives; deterministic and removes the slot-pick
                // ambiguity the old priority-queue version inherited from heap
                // pop order.
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                let merge_dist = best_d;
                raw_merges.push((lo, hi, merge_dist));

                // Lance-Williams update: d(merged, k) for all other active k.
                // Arithmetic in f64 for numerical stability (matches k-means
                // centroid-accumulation convention); result cast back to F.
                let n_lo = size[lo] as f64;
                let n_hi = size[hi] as f64;
                for k in 0..n {
                    if !active[k] || k == lo || k == hi {
                        continue;
                    }
                    let d_lo_k = dist_matrix[cd_idx(lo, k, n)].to_f64_lossy();
                    let d_hi_k = dist_matrix[cd_idx(hi, k, n)].to_f64_lossy();
                    let n_k = size[k] as f64;

                    let new_dist = match linkage {
                        Linkage::Ward => {
                            let n_total = n_lo + n_hi + n_k;
                            ((n_lo + n_k) * d_lo_k + (n_hi + n_k) * d_hi_k
                                - n_k * merge_dist)
                                / n_total
                        }
                        Linkage::Complete => d_lo_k.max(d_hi_k),
                        Linkage::Single => d_lo_k.min(d_hi_k),
                        Linkage::Average => (n_lo * d_lo_k + n_hi * d_hi_k) / (n_lo + n_hi),
                    };

                    dist_matrix[cd_idx(lo, k, n)] = F::from_f64_lossy(new_dist);
                }

                active[hi] = false;
                size[lo] += size[hi];
                break; // back to the outer "n-1 merges" loop
            }

            chain.push(b);
        }
    }

    // ---- Sort + cut phase: produce scipy-compatible output ----
    //
    // NN-chain's chain-order may not be distance-monotonic across chain
    // restarts. Sorting (stable, so chain order breaks ties) gives the
    // canonical scipy linkage matrix ordering. We then take the first
    // n_to_apply sorted merges for both labels and the children/distances
    // output. n_to_apply depends on the cut spec:
    //   - NClusters(k):      n_to_apply = n - k.
    //   - DistanceThreshold: number of merges whose reported distance is
    //                        strictly below the threshold (matches sklearn).
    let mut order: Vec<usize> = (0..raw_merges.len()).collect();
    order.sort_by(|&i, &j| {
        raw_merges[i]
            .2
            .partial_cmp(&raw_merges[j].2)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let n_to_apply: usize = match cut {
        Cut::NClusters(k) => n - k,
        Cut::DistanceThreshold(t) => {
            // Walk the sorted prefix until we hit a merge at or above the
            // threshold. For Ward, distances are stored squared but reported
            // as the actual Euclidean distance — compare against the reported
            // value so sklearn-style thresholds (which are in Euclidean units
            // for Ward) work as users expect.
            let mut count = 0usize;
            for &ord_idx in &order {
                let raw_d = raw_merges[ord_idx].2;
                let report_d = if matches!(linkage, Linkage::Ward) {
                    raw_d.sqrt()
                } else {
                    raw_d
                };
                if report_d < t {
                    count += 1;
                } else {
                    break;
                }
            }
            count
        }
    };
    let result_n_clusters = n - n_to_apply;

    // Union-find on slots for label assignment. Smaller-index becomes the
    // root, mirroring the lo-survives convention used during NN-chain so
    // that final cluster labels are deterministic and stable across reruns.
    let mut parent: Vec<usize> = (0..n).collect();
    for &ord_idx in &order[..n_to_apply] {
        let (lo, hi, _) = raw_merges[ord_idx];
        let rlo = uf_find(&mut parent, lo);
        let rhi = uf_find(&mut parent, hi);
        if rlo != rhi {
            if rlo < rhi {
                parent[rhi] = rlo;
            } else {
                parent[rlo] = rhi;
            }
        }
    }

    // Assign labels: walk slot indices in order, assign sequential ints to
    // each distinct root.
    let mut slot_label: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
    let mut next_label = 0i64;
    let mut labels = vec![0i64; n];
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        let lbl = *slot_label.entry(root).or_insert_with(|| {
            let l = next_label;
            next_label += 1;
            l
        });
        labels[i] = lbl;
    }

    // Build scipy-format children + distances from the sorted merge prefix.
    // cluster_id[slot] tracks the current cluster ID for each slot: initially
    // its slot index (i.e. a leaf), updated to (n + j) when it survives a merge.
    let mut cluster_id: Vec<usize> = (0..n).collect();
    let mut children: Vec<(usize, usize)> = Vec::with_capacity(n_to_apply);
    let mut merge_distances: Vec<f64> = Vec::with_capacity(n_to_apply);
    for (j, &ord_idx) in order[..n_to_apply].iter().enumerate() {
        let (lo, hi, dist) = raw_merges[ord_idx];
        let report_dist = if matches!(linkage, Linkage::Ward) {
            dist.sqrt() // report actual Euclidean distance for Ward
        } else {
            dist
        };
        children.push((cluster_id[lo], cluster_id[hi]));
        merge_distances.push(report_dist);
        cluster_id[lo] = n + j;
    }

    Ok(AgglomerativeState {
        labels,
        n_clusters: result_n_clusters,
        children,
        distances: merge_distances,
        _phantom: std::marker::PhantomData,
    })
}

/// Path-compressing union-find lookup. Used to compute final cluster labels
/// from the sorted merge prefix without rebuilding a membership chain.
#[inline]
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]]; // halving — sufficient and branch-free
        x = parent[x];
    }
    x
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
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Ward, Metric::Euclidean)
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
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(1), Linkage::Ward, Metric::Euclidean)
                .unwrap();
        assert!(result.labels.iter().all(|&l| l == 0));
    }

    #[test]
    fn test_n_equals_n_clusters() {
        let data = array![[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(3), Linkage::Ward, Metric::Euclidean)
                .unwrap();
        let mut sorted = result.labels.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_children_length() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Ward, Metric::Euclidean)
                .unwrap();
        // n=4, target=2 → 2 merges
        assert_eq!(result.children.len(), 2);
        assert_eq!(result.distances.len(), 2);
    }

    #[test]
    fn test_distances_non_decreasing() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0], [20.0, 0.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(1), Linkage::Ward, Metric::Euclidean)
                .unwrap();
        for w in result.distances.windows(2) {
            assert!(w[1] >= w[0] - 1e-10);
        }
    }

    #[test]
    fn test_complete_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Complete, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[2], result.labels[3]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_average_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Average, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_single_linkage() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Single, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_ward_requires_euclidean() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(1), Linkage::Ward, Metric::Cosine),
            Err(ClusterError::WardRequiresEuclidean)
        ));
    }

    #[test]
    fn test_manhattan_metric() {
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]];
        let result =
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Complete, Metric::Manhattan)
                .unwrap();
        assert_eq!(result.labels[0], result.labels[1]);
    }

    #[test]
    fn test_f32() {
        let data = array![[0.0f32, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0],];
        let result =
            run_agglomerative_with_metric_f32(&data.view(), Cut::NClusters(2), Linkage::Ward, Metric::Euclidean)
                .unwrap();
        assert_eq!(result.labels.len(), 4);
        assert_eq!(result.n_clusters, 2);
    }

    #[test]
    fn test_invalid_n_clusters() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(5), Linkage::Ward, Metric::Euclidean),
            Err(ClusterError::InvalidClusters { .. })
        ));
    }

    #[test]
    fn test_empty_input() {
        let data = Array2::<f64>::zeros((0, 2));
        assert!(matches!(
            run_agglomerative_with_metric(&data.view(), Cut::NClusters(1), Linkage::Ward, Metric::Euclidean),
            Err(ClusterError::EmptyInput)
        ));
    }

    #[test]
    fn test_deterministic() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [6.0, 0.0]];
        let r1 = run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Ward, Metric::Euclidean)
            .unwrap();
        let r2 = run_agglomerative_with_metric(&data.view(), Cut::NClusters(2), Linkage::Ward, Metric::Euclidean)
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
            Cut::NClusters(3),
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

    // ---- distance_threshold mode (Phase 3) ----

    #[test]
    fn test_distance_threshold_typical() {
        // Two tight pairs separated by a wide gap. With Euclidean + complete
        // linkage, intra-pair distances are 1.0 and the inter-pair distance is
        // 100. A threshold of 5.0 should produce two clusters; a threshold of
        // 200.0 should produce one.
        let data = array![[0.0, 0.0], [1.0, 0.0], [100.0, 0.0], [101.0, 0.0]];

        let tight = run_agglomerative_with_metric(
            &data.view(),
            Cut::DistanceThreshold(5.0),
            Linkage::Complete,
            Metric::Euclidean,
        )
        .unwrap();
        assert_eq!(tight.n_clusters, 2);
        assert_eq!(tight.labels[0], tight.labels[1]);
        assert_eq!(tight.labels[2], tight.labels[3]);
        assert_ne!(tight.labels[0], tight.labels[2]);

        let loose = run_agglomerative_with_metric(
            &data.view(),
            Cut::DistanceThreshold(200.0),
            Linkage::Complete,
            Metric::Euclidean,
        )
        .unwrap();
        assert_eq!(loose.n_clusters, 1);
        for &l in &loose.labels {
            assert_eq!(l, loose.labels[0]);
        }
    }

    #[test]
    fn test_distance_threshold_too_low_keeps_singletons() {
        // Threshold below every pairwise distance → no merges, n singletons.
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]];
        let result = run_agglomerative_with_metric(
            &data.view(),
            Cut::DistanceThreshold(0.5),
            Linkage::Complete,
            Metric::Euclidean,
        )
        .unwrap();
        assert_eq!(result.n_clusters, 3);
        assert_eq!(result.children.len(), 0);
        assert_eq!(result.distances.len(), 0);
        let mut sorted = result.labels.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    #[test]
    fn test_distance_threshold_sklearn_semantics_strict_lt() {
        // sklearn: "linkage distance threshold at or above which clusters will
        // not be merged" — exact-equality merges are NOT applied. Verify this
        // strict-less-than semantics with an exactly-on-threshold case.
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]];
        let result = run_agglomerative_with_metric(
            &data.view(),
            Cut::DistanceThreshold(1.0), // d(0,1) is exactly 1.0
            Linkage::Complete,
            Metric::Euclidean,
        )
        .unwrap();
        // Strict <: the 1.0 merge is not applied, so we keep three singletons.
        assert_eq!(result.n_clusters, 3);
    }

    #[test]
    fn test_distance_threshold_ward_uses_reported_distance() {
        // For Ward the matrix stores squared Euclidean internally but the
        // reported distance is sqrt. Threshold comparisons should match the
        // reported value (sklearn's convention), so a threshold of 1.5
        // applied to two points 1 unit apart should merge them (sqrt(1.0) < 1.5).
        let data = array![[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]];
        let result = run_agglomerative_with_metric(
            &data.view(),
            Cut::DistanceThreshold(1.5),
            Linkage::Ward,
            Metric::Euclidean,
        )
        .unwrap();
        // d(0,1)=1.0 below threshold → merged. d(merged, 2) is ≫ 1.5.
        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_ne!(result.labels[0], result.labels[2]);
    }

    #[test]
    fn test_distance_threshold_cosine_supplier_workload() {
        // Mirror the supplier-clustering notebook: average linkage, cosine
        // distance, threshold around 0.2. Three tight cosine clusters should
        // separate cleanly.
        let n_per = 8;
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
            Cut::DistanceThreshold(0.2),
            Linkage::Average,
            Metric::Cosine,
        )
        .unwrap();
        assert_eq!(result.n_clusters, 3);
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
    }

    #[test]
    fn test_distance_threshold_invalid_nan() {
        let data = array![[0.0, 0.0], [1.0, 0.0]];
        assert!(matches!(
            run_agglomerative_with_metric(
                &data.view(),
                Cut::DistanceThreshold(f64::NAN),
                Linkage::Ward,
                Metric::Euclidean,
            ),
            Err(ClusterError::InvalidDistanceThreshold(_))
        ));
        assert!(matches!(
            run_agglomerative_with_metric(
                &data.view(),
                Cut::DistanceThreshold(f64::INFINITY),
                Linkage::Ward,
                Metric::Euclidean,
            ),
            Err(ClusterError::InvalidDistanceThreshold(_))
        ));
    }
}
