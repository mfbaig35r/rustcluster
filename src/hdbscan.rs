//! HDBSCAN (Hierarchical Density-Based Spatial Clustering).
//!
//! Builds a cluster hierarchy via mutual reachability distances and extracts
//! flat clusters using stability-based selection.
//!
//! Six stages:
//! 1. Core distances (k-th nearest neighbor per point)
//! 2. MST on mutual reachability graph (Prim's algorithm)
//! 3. Build hierarchy (Union-Find on sorted MST edges)
//! 4. Condense tree (collapse clusters smaller than min_cluster_size)
//! 5. Compute stability
//! 6. Extract flat clusters (EOM or leaf selection)
//!
//! Complexity: O(n²) for stages 1-2 (naive). Spatial indexing is future work.
//!
//! Reference: Campello, Moulavi, Sander (2013).

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use crate::distance::{
    CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean,
};
use crate::error::ClusterError;
use crate::kdtree::{BBoxDistance, KdTree};
use crate::utils::validate_data_generic;

/// Result of a fitted HDBSCAN model.
pub struct HdbscanState<F: Scalar> {
    pub labels: Vec<i64>,
    pub probabilities: Vec<f64>,
    pub cluster_persistence: Vec<f64>,
    pub n_clusters: usize,
    pub _phantom: std::marker::PhantomData<F>,
}

/// Cluster selection method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterSelectionMethod {
    Eom,
    Leaf,
}

impl ClusterSelectionMethod {
    pub fn from_str(s: &str) -> Result<Self, ClusterError> {
        match s.to_lowercase().as_str() {
            "eom" => Ok(ClusterSelectionMethod::Eom),
            "leaf" => Ok(ClusterSelectionMethod::Leaf),
            _ => Err(ClusterError::InvalidClusterSelectionMethod(s.to_string())),
        }
    }
}

// ---- Public entry points ----

pub fn run_hdbscan_with_metric(
    data: &ArrayView2<f64>,
    min_cluster_size: usize,
    min_samples: usize,
    metric: Metric,
    selection: ClusterSelectionMethod,
) -> Result<HdbscanState<f64>, ClusterError> {
    match metric {
        Metric::Euclidean => run_hdbscan_accelerated::<f64, SquaredEuclidean, SquaredEuclidean>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
        Metric::Cosine => run_hdbscan_generic::<f64, CosineDistance>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
        Metric::Manhattan => run_hdbscan_accelerated::<f64, ManhattanDistance, ManhattanDistance>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
    }
}

pub fn run_hdbscan_with_metric_f32(
    data: &ArrayView2<f32>,
    min_cluster_size: usize,
    min_samples: usize,
    metric: Metric,
    selection: ClusterSelectionMethod,
) -> Result<HdbscanState<f32>, ClusterError> {
    match metric {
        Metric::Euclidean => run_hdbscan_accelerated::<f32, SquaredEuclidean, SquaredEuclidean>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
        Metric::Cosine => run_hdbscan_generic::<f32, CosineDistance>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
        Metric::Manhattan => run_hdbscan_accelerated::<f32, ManhattanDistance, ManhattanDistance>(
            data,
            min_cluster_size,
            min_samples,
            selection,
        ),
    }
}

// ---- Accelerated implementation (KD-tree for core distances) ----

fn run_hdbscan_accelerated<F: Scalar, D: Distance<F>, B: BBoxDistance>(
    data: &ArrayView2<F>,
    min_cluster_size: usize,
    min_samples: usize,
    selection: ClusterSelectionMethod,
) -> Result<HdbscanState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();
    if min_cluster_size < 2 {
        return Err(ClusterError::InvalidMinClusterSize(min_cluster_size));
    }
    if min_samples == 0 {
        return Err(ClusterError::InvalidMinSamples(0));
    }

    let data_slice = data.as_slice().expect("data must be C-contiguous");

    if n < min_cluster_size {
        return Ok(HdbscanState {
            labels: vec![-1; n],
            probabilities: vec![0.0; n],
            cluster_persistence: vec![],
            n_clusters: 0,
            _phantom: std::marker::PhantomData,
        });
    }

    // Try KD-tree for core distances
    let data_f64: Vec<f64> = data_slice.iter().map(|v| v.to_f64_lossy()).collect();
    let tree = KdTree::build_v2(data_slice, n, d);

    let core_dists = match tree {
        Some(ref tree) => {
            // KD-tree accelerated k-NN
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let query = &data_f64[i * d..(i + 1) * d];
                    let knn = tree.query_knn::<B>(query, min_samples, Some(i));
                    match knn.last() {
                        Some(&(_, dist)) => D::to_metric(dist),
                        None => 0.0,
                    }
                })
                .collect::<Vec<f64>>()
        }
        None => {
            // Brute force fallback
            compute_core_distances::<F, D>(data_slice, n, d, min_samples)
        }
    };

    // MST and remaining stages use existing code (still O(n²))
    let mst = build_mst::<F, D>(data_slice, n, d, &core_dists);
    let hierarchy = build_hierarchy(&mst, n);
    let (labels, probabilities, persistence) =
        condense_and_extract(&hierarchy, n, min_cluster_size, selection);

    let n_clusters = {
        let mut unique = std::collections::HashSet::new();
        for &l in &labels {
            if l >= 0 {
                unique.insert(l);
            }
        }
        unique.len()
    };

    Ok(HdbscanState {
        labels,
        probabilities,
        cluster_persistence: persistence,
        n_clusters,
        _phantom: std::marker::PhantomData,
    })
}

// ---- Generic implementation (brute force, for Cosine) ----

fn run_hdbscan_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    min_cluster_size: usize,
    min_samples: usize,
    selection: ClusterSelectionMethod,
) -> Result<HdbscanState<F>, ClusterError> {
    validate_data_generic(data)?;

    let (n, d) = data.dim();
    if min_cluster_size < 2 {
        return Err(ClusterError::InvalidMinClusterSize(min_cluster_size));
    }
    if min_samples == 0 {
        return Err(ClusterError::InvalidMinSamples(0));
    }

    let data_slice = data.as_slice().expect("data must be C-contiguous");

    // Handle trivial cases
    if n < min_cluster_size {
        return Ok(HdbscanState {
            labels: vec![-1; n],
            probabilities: vec![0.0; n],
            cluster_persistence: vec![],
            n_clusters: 0,
            _phantom: std::marker::PhantomData,
        });
    }

    // Stage 1: Core distances
    let core_dists = compute_core_distances::<F, D>(data_slice, n, d, min_samples);

    // Stage 2: MST via Prim's on mutual reachability graph
    let mst = build_mst::<F, D>(data_slice, n, d, &core_dists);

    // Stage 3: Build hierarchy via Union-Find
    let hierarchy = build_hierarchy(&mst, n);

    // Stage 4-6: Condense tree and extract clusters
    let (labels, probabilities, persistence) =
        condense_and_extract(&hierarchy, n, min_cluster_size, selection);

    let n_clusters = {
        let mut unique = std::collections::HashSet::new();
        for &l in &labels {
            if l >= 0 {
                unique.insert(l);
            }
        }
        unique.len()
    };

    Ok(HdbscanState {
        labels,
        probabilities,
        cluster_persistence: persistence,
        n_clusters,
        _phantom: std::marker::PhantomData,
    })
}

// ---- Stage 1: Core distances ----

/// For each point, compute the distance to its k-th nearest neighbor.
/// Uses sqrt of squared distance for Euclidean compatibility.
fn compute_core_distances<F: Scalar, D: Distance<F>>(
    data_slice: &[F],
    n: usize,
    d: usize,
    k: usize,
) -> Vec<f64> {
    let k = k.min(n); // clamp to n

    (0..n)
        .into_par_iter()
        .map(|i| {
            let point_i = &data_slice[i * d..(i + 1) * d];
            let mut dists: Vec<f64> = (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        D::to_metric(
                            D::distance(point_i, &data_slice[j * d..(j + 1) * d]).to_f64_lossy(),
                        )
                    }
                })
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // k-th nearest neighbor (index k-1 since index 0 is self with distance 0)
            if k < dists.len() {
                dists[k]
            } else {
                *dists.last().unwrap_or(&0.0)
            }
        })
        .collect()
}

// ---- Stage 2: MST via Prim's ----

/// Edge in the MST: (from, to, weight).
type MstEdge = (usize, usize, f64);

/// Build MST on the mutual reachability graph using Prim's algorithm.
fn build_mst<F: Scalar, D: Distance<F>>(
    data_slice: &[F],
    n: usize,
    d: usize,
    core_dists: &[f64],
) -> Vec<MstEdge> {
    if n <= 1 {
        return vec![];
    }

    let mut in_tree = vec![false; n];
    let mut min_weight = vec![f64::MAX; n]; // min mutual reachability to any tree node
    let mut min_from = vec![0usize; n]; // which tree node gives the min weight
    let mut mst = Vec::with_capacity(n - 1);

    // Start from node 0
    in_tree[0] = true;

    // Initialize distances from node 0 to all others
    let point_0 = &data_slice[0..d];
    for j in 1..n {
        let point_j = &data_slice[j * d..(j + 1) * d];
        let raw_dist = D::to_metric(D::distance(point_0, point_j).to_f64_lossy());
        let mrd = raw_dist.max(core_dists[0]).max(core_dists[j]);
        min_weight[j] = mrd;
        min_from[j] = 0;
    }

    for _ in 0..n - 1 {
        // Find the node not in tree with minimum edge weight
        let mut best_node = 0;
        let mut best_weight = f64::MAX;
        for j in 0..n {
            if !in_tree[j] && min_weight[j] < best_weight {
                best_weight = min_weight[j];
                best_node = j;
            }
        }

        in_tree[best_node] = true;
        mst.push((min_from[best_node], best_node, best_weight));

        // Update distances for the newly added node
        let point_best = &data_slice[best_node * d..(best_node + 1) * d];
        for j in 0..n {
            if !in_tree[j] {
                let point_j = &data_slice[j * d..(j + 1) * d];
                let raw_dist = D::to_metric(D::distance(point_best, point_j).to_f64_lossy());
                let mrd = raw_dist.max(core_dists[best_node]).max(core_dists[j]);
                if mrd < min_weight[j] {
                    min_weight[j] = mrd;
                    min_from[j] = best_node;
                }
            }
        }
    }

    // Sort by weight for hierarchy construction
    mst.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    mst
}

// ---- Stage 3: Build hierarchy ----

/// Dendrogram entry: (child_a, child_b, distance, merged_size).
type DendrogramEntry = (usize, usize, f64, usize);

fn build_hierarchy(mst: &[MstEdge], n: usize) -> Vec<DendrogramEntry> {
    let mut uf = UnionFind::new(n);
    let mut hierarchy = Vec::with_capacity(n.saturating_sub(1));

    for &(a, b, weight) in mst {
        let root_a = uf.find(a);
        let root_b = uf.find(b);
        if root_a != root_b {
            let size_a = uf.size[root_a];
            let size_b = uf.size[root_b];
            uf.union(root_a, root_b);
            hierarchy.push((root_a, root_b, weight, size_a + size_b));
        }
    }

    hierarchy
}

// ---- Stages 4-6: Condense tree and extract clusters ----

fn condense_and_extract(
    hierarchy: &[DendrogramEntry],
    n: usize,
    min_cluster_size: usize,
    selection: ClusterSelectionMethod,
) -> (Vec<i64>, Vec<f64>, Vec<f64>) {
    if hierarchy.is_empty() {
        return (vec![-1; n], vec![0.0; n], vec![]);
    }

    // Build a parent-child representation of the dendrogram.
    // Internal nodes are numbered n, n+1, n+2, ...
    // Leaves are 0..n.

    let n_merges = hierarchy.len();
    let n_nodes = n + n_merges; // total nodes in the tree

    // For each internal node, record its children and the lambda (1/distance) at which it formed
    let mut children: Vec<(usize, usize)> = vec![(0, 0); n_nodes];
    let mut node_size: Vec<usize> = vec![1; n_nodes]; // leaves have size 1
    let mut lambda_birth: Vec<f64> = vec![0.0; n_nodes];

    // Map original UF root IDs to internal node IDs
    let mut root_to_node: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for i in 0..n {
        root_to_node.insert(i, i); // initially each point is its own node
    }

    for (idx, &(child_a, child_b, distance, size)) in hierarchy.iter().enumerate() {
        let internal_id = n + idx;
        let node_a = *root_to_node.get(&child_a).unwrap_or(&child_a);
        let node_b = *root_to_node.get(&child_b).unwrap_or(&child_b);

        children[internal_id] = (node_a, node_b);
        node_size[internal_id] = size;

        let lambda = if distance > 0.0 {
            1.0 / distance
        } else {
            f64::MAX
        };
        lambda_birth[internal_id] = lambda;

        // The merged root now maps to this internal node
        // Find which root survived the union in the original UF
        // Since we process in order, update both roots to point to new node
        root_to_node.insert(child_a, internal_id);
        root_to_node.insert(child_b, internal_id);
    }

    let root = n + n_merges - 1; // the final merge is the root

    // Condense: walk top-down, collecting "condensed clusters"
    // A condensed cluster is one where both children have size >= min_cluster_size (a real split)
    // or a leaf-level cluster.

    // For each point, track which condensed cluster it belongs to and its lambda_p (when it "fell out")
    let mut point_cluster: Vec<usize> = vec![root; n]; // which condensed cluster each point belongs to
    let mut point_lambda: Vec<f64> = vec![0.0; n]; // lambda at which each point was last assigned

    // Condensed cluster info
    let mut condensed_clusters: Vec<usize> = vec![root]; // IDs of condensed clusters
    let mut cluster_lambda_birth: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    let mut cluster_lambda_death: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    let mut cluster_stability: std::collections::HashMap<usize, f64> =
        std::collections::HashMap::new();
    let mut cluster_children: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();

    cluster_lambda_birth.insert(root, lambda_birth[root]);

    // BFS/DFS to condense the tree
    let mut stack: Vec<(usize, usize)> = vec![(root, root)]; // (node, current_condensed_cluster)

    while let Some((node, current_cluster)) = stack.pop() {
        if node < n {
            // Leaf node (data point)
            point_cluster[node] = current_cluster;
            point_lambda[node] = *cluster_lambda_birth.get(&current_cluster).unwrap_or(&0.0);
            continue;
        }

        let (left, right) = children[node];
        let left_size = if left < n { 1 } else { node_size[left] };
        let right_size = if right < n { 1 } else { node_size[right] };
        let lambda = lambda_birth[node];

        let left_big = left_size >= min_cluster_size;
        let right_big = right_size >= min_cluster_size;

        if left_big && right_big {
            // Real split: both children become new condensed clusters
            cluster_lambda_death.insert(current_cluster, lambda);

            // Left child becomes a new condensed cluster
            condensed_clusters.push(left);
            cluster_lambda_birth.insert(left, lambda);
            cluster_children
                .entry(current_cluster)
                .or_default()
                .push(left);
            stack.push((left, left));

            // Right child becomes a new condensed cluster
            condensed_clusters.push(right);
            cluster_lambda_birth.insert(right, lambda);
            cluster_children
                .entry(current_cluster)
                .or_default()
                .push(right);
            stack.push((right, right));
        } else if left_big {
            // Only left is big enough — right's points "fall out" as noise candidates
            // Assign right subtree points with their lambda
            assign_subtree_points(
                right,
                n,
                &children,
                &node_size,
                current_cluster,
                lambda,
                &mut point_cluster,
                &mut point_lambda,
            );
            stack.push((left, current_cluster));
        } else if right_big {
            assign_subtree_points(
                left,
                n,
                &children,
                &node_size,
                current_cluster,
                lambda,
                &mut point_cluster,
                &mut point_lambda,
            );
            stack.push((right, current_cluster));
        } else {
            // Neither child is big enough — all points fall out
            assign_subtree_points(
                left,
                n,
                &children,
                &node_size,
                current_cluster,
                lambda,
                &mut point_cluster,
                &mut point_lambda,
            );
            assign_subtree_points(
                right,
                n,
                &children,
                &node_size,
                current_cluster,
                lambda,
                &mut point_cluster,
                &mut point_lambda,
            );
        }
    }

    // Set lambda_death for leaf condensed clusters (those that were never split)
    for &c in &condensed_clusters {
        if !cluster_lambda_death.contains_key(&c) {
            // Find max lambda among its points
            let max_lambda = (0..n)
                .filter(|&i| point_cluster[i] == c)
                .map(|i| point_lambda[i])
                .fold(0.0f64, f64::max);
            let birth = *cluster_lambda_birth.get(&c).unwrap_or(&0.0);
            cluster_lambda_death.insert(c, max_lambda.max(birth));
        }
    }

    // Compute stability for each condensed cluster
    for &c in &condensed_clusters {
        let birth = *cluster_lambda_birth.get(&c).unwrap_or(&0.0);
        let stability: f64 = (0..n)
            .filter(|&i| point_cluster[i] == c)
            .map(|i| point_lambda[i] - birth)
            .filter(|&v| v > 0.0)
            .sum();
        cluster_stability.insert(c, stability.max(0.0));
    }

    // Stage 6: Extract flat clusters
    let selected = match selection {
        ClusterSelectionMethod::Eom => {
            select_eom(&condensed_clusters, &cluster_stability, &cluster_children)
        }
        ClusterSelectionMethod::Leaf => select_leaf(&condensed_clusters, &cluster_children),
    };

    // Assign labels
    let mut labels = vec![-1i64; n];
    let mut probabilities = vec![0.0f64; n];
    let mut persistence = Vec::new();

    for (cluster_label, &cluster_id) in selected.iter().enumerate() {
        let birth = *cluster_lambda_birth.get(&cluster_id).unwrap_or(&0.0);
        let death = *cluster_lambda_death.get(&cluster_id).unwrap_or(&0.0);
        let span = (death - birth).max(1e-15);
        let stab = *cluster_stability.get(&cluster_id).unwrap_or(&0.0);
        persistence.push(stab);

        for i in 0..n {
            if point_cluster[i] == cluster_id {
                labels[i] = cluster_label as i64;
                let prob = ((point_lambda[i] - birth) / span).clamp(0.0, 1.0);
                probabilities[i] = prob;
            }
        }
    }

    (labels, probabilities, persistence)
}

/// Assign all points in a subtree to a condensed cluster with a given lambda.
fn assign_subtree_points(
    node: usize,
    n: usize,
    children: &[(usize, usize)],
    _node_size: &[usize],
    cluster: usize,
    lambda: f64,
    point_cluster: &mut [usize],
    point_lambda: &mut [f64],
) {
    let mut stack = vec![node];
    while let Some(nd) = stack.pop() {
        if nd < n {
            point_cluster[nd] = cluster;
            point_lambda[nd] = lambda;
        } else {
            let (left, right) = children[nd];
            stack.push(left);
            stack.push(right);
        }
    }
}

/// EOM (Excess of Mass) cluster selection: bottom-up stability propagation.
fn select_eom(
    clusters: &[usize],
    stability: &std::collections::HashMap<usize, f64>,
    children_map: &std::collections::HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    // Find leaf clusters (no children in the condensed tree)
    let mut selected: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut propagated_stability: std::collections::HashMap<usize, f64> = stability.clone();

    // Process bottom-up: iterate clusters in reverse order (leaves first)
    let mut ordered = clusters.to_vec();
    ordered.reverse();

    for &c in &ordered {
        let child_list = children_map.get(&c);
        let is_leaf = match child_list {
            None => true,
            Some(v) => v.is_empty(),
        };
        if is_leaf {
            // Leaf condensed cluster — always selected initially
            selected.insert(c);
        } else {
            let child_ids = child_list.unwrap();
            {
                let children_total: f64 = child_ids
                    .iter()
                    .map(|ch| propagated_stability.get(ch).unwrap_or(&0.0))
                    .sum();
                let own_stability = *stability.get(&c).unwrap_or(&0.0);

                if own_stability >= children_total {
                    // This cluster is better than its children combined
                    selected.insert(c);
                    // Deselect all descendants
                    for ch in child_ids {
                        deselect_subtree(*ch, children_map, &mut selected);
                    }
                    propagated_stability.insert(c, own_stability);
                } else {
                    // Children are better — propagate their stability up
                    propagated_stability.insert(c, children_total);
                }
            }
        }
    }

    // Filter to only actually selected clusters and sort for deterministic label order
    let mut result: Vec<usize> = selected.into_iter().collect();
    result.sort();
    result
}

fn deselect_subtree(
    node: usize,
    children_map: &std::collections::HashMap<usize, Vec<usize>>,
    selected: &mut std::collections::HashSet<usize>,
) {
    selected.remove(&node);
    if let Some(children) = children_map.get(&node) {
        for &ch in children {
            deselect_subtree(ch, children_map, selected);
        }
    }
}

/// Leaf cluster selection: select all condensed clusters with no children.
fn select_leaf(
    clusters: &[usize],
    children_map: &std::collections::HashMap<usize, Vec<usize>>,
) -> Vec<usize> {
    let mut result: Vec<usize> = clusters
        .iter()
        .filter(|&&c| children_map.get(&c).map_or(true, |v| v.is_empty()))
        .copied()
        .collect();
    result.sort();
    result
}

// ---- Union-Find ----

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            rank: vec![0; n],
            size: vec![1; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]); // path compression
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        // Union by rank
        if self.rank[ra] < self.rank[rb] {
            self.parent[ra] = rb;
            self.size[rb] += self.size[ra];
        } else if self.rank[ra] > self.rank[rb] {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
        } else {
            self.parent[rb] = ra;
            self.size[ra] += self.size[rb];
            self.rank[ra] += 1;
        }
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
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            3,
            3,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert_eq!(result.labels.len(), 10);
        assert_eq!(result.n_clusters, 2);
        let c1 = result.labels[0];
        let c2 = result.labels[5];
        assert!(c1 >= 0);
        assert!(c2 >= 0);
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_noise_detection() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
            [50.0, 50.0], // outlier
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            3,
            3,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert_eq!(result.labels[10], -1); // outlier is noise
    }

    #[test]
    fn test_all_noise() {
        let data = array![[0.0, 0.0], [10.0, 10.0]];
        // min_cluster_size=3 but only 2 points
        let result = run_hdbscan_with_metric(
            &data.view(),
            3,
            2,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert!(result.labels.iter().all(|&l| l == -1));
        assert_eq!(result.n_clusters, 0);
    }

    #[test]
    fn test_probabilities_range() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            3,
            3,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        for &p in &result.probabilities {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn test_deterministic() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];
        let r1 = run_hdbscan_with_metric(
            &data.view(),
            2,
            2,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        let r2 = run_hdbscan_with_metric(
            &data.view(),
            2,
            2,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert_eq!(r1.labels, r2.labels);
        assert_eq!(r1.probabilities, r2.probabilities);
    }

    #[test]
    fn test_leaf_selection() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            2,
            2,
            Metric::Euclidean,
            ClusterSelectionMethod::Leaf,
        )
        .unwrap();
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_invalid_min_cluster_size() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_hdbscan_with_metric(
                &data.view(),
                1,
                1,
                Metric::Euclidean,
                ClusterSelectionMethod::Eom
            ),
            Err(ClusterError::InvalidMinClusterSize(_))
        ));
    }

    #[test]
    fn test_invalid_min_samples() {
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        assert!(matches!(
            run_hdbscan_with_metric(
                &data.view(),
                2,
                0,
                Metric::Euclidean,
                ClusterSelectionMethod::Eom
            ),
            Err(ClusterError::InvalidMinSamples(_))
        ));
    }

    #[test]
    fn test_empty_input() {
        let data = Array2::<f64>::zeros((0, 2));
        assert!(matches!(
            run_hdbscan_with_metric(
                &data.view(),
                2,
                2,
                Metric::Euclidean,
                ClusterSelectionMethod::Eom
            ),
            Err(ClusterError::EmptyInput)
        ));
    }

    #[test]
    fn test_f32() {
        let data = array![
            [0.0f32, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
        ];
        let result = run_hdbscan_with_metric_f32(
            &data.view(),
            2,
            2,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert_eq!(result.labels.len(), 8);
        assert!(result.n_clusters >= 1);
    }

    #[test]
    fn test_cosine_metric() {
        let data = array![
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],
            [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
            [0.03, 0.97],
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            2,
            2,
            Metric::Cosine,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        assert_eq!(result.labels.len(), 8);
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_ne!(uf.find(0), uf.find(2));
        uf.union(0, 2);
        assert_eq!(uf.find(0), uf.find(3));
        let root = uf.find(0);
        assert_eq!(uf.size[root], 4);
    }

    #[test]
    fn test_cluster_persistence_non_negative() {
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [0.05, 0.05],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
            [10.1, 10.1],
            [10.05, 10.05],
        ];
        let result = run_hdbscan_with_metric(
            &data.view(),
            3,
            3,
            Metric::Euclidean,
            ClusterSelectionMethod::Eom,
        )
        .unwrap();
        for &p in &result.cluster_persistence {
            assert!(p >= 0.0);
        }
    }
}
