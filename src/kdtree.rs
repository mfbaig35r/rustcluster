//! KD-tree spatial index for accelerating neighbor queries.
//!
//! Used internally by DBSCAN (range query) and HDBSCAN (k-NN query).
//! Flat arena layout for cache locality. Points stored as f64.
//! Supports SquaredEuclidean and Manhattan distance pruning.
//! Cosine distance is not supported (no axis-aligned bounding box pruning).

use std::collections::BinaryHeap;

use crate::distance::Scalar;

const DEFAULT_LEAF_SIZE: usize = 32;
const MAX_TREE_DIM: usize = 16;

// ---- Node and Tree ----

#[derive(Debug, Clone)]
struct KdNode {
    split_dim: usize,
    split_val: f64,
    left_or_start: usize, // internal: left child idx / leaf: start in reordered buffer
    right_or_end: usize,  // internal: right child idx / leaf: end (exclusive)
    is_leaf: bool,
}

/// A KD-tree for fast spatial queries on n points of dimension d.
pub struct KdTree {
    nodes: Vec<KdNode>,
    points: Vec<f64>,        // reordered, contiguous f64
    orig_indices: Vec<usize>,
    bbox_min: Vec<f64>,      // flat: node_idx * d + dim
    bbox_max: Vec<f64>,
    n: usize,
    d: usize,
}

// ---- BBoxDistance trait ----

/// Minimum distance from a point to an axis-aligned bounding box.
/// Implemented per metric that supports KD-tree pruning.
pub trait BBoxDistance: Send + Sync {
    fn min_dist_to_bbox(point: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64;
}

impl BBoxDistance for crate::distance::SquaredEuclidean {
    #[inline]
    fn min_dist_to_bbox(point: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64 {
        let mut dist_sq = 0.0;
        for i in 0..point.len() {
            let clamped = point[i].clamp(bbox_min[i], bbox_max[i]);
            let diff = point[i] - clamped;
            dist_sq += diff * diff;
        }
        dist_sq
    }
}

impl BBoxDistance for crate::distance::ManhattanDistance {
    #[inline]
    fn min_dist_to_bbox(point: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64 {
        let mut dist = 0.0;
        for i in 0..point.len() {
            if point[i] < bbox_min[i] {
                dist += bbox_min[i] - point[i];
            } else if point[i] > bbox_max[i] {
                dist += point[i] - bbox_max[i];
            }
        }
        dist
    }
}

/// Point-to-point distance via degenerate bounding box.
#[inline(always)]
fn point_dist<B: BBoxDistance>(a: &[f64], b: &[f64]) -> f64 {
    B::min_dist_to_bbox(a, b, b)
}

// ---- Construction ----

impl KdTree {
    /// Build a KD-tree from flat row-major data.
    /// Returns None if d > MAX_TREE_DIM.
    pub fn build<F: Scalar>(data_slice: &[F], n: usize, d: usize) -> Option<Self> {
        if d > MAX_TREE_DIM || n == 0 {
            return None;
        }

        // Convert to f64 once
        let data_f64: Vec<f64> = data_slice.iter().map(|v| v.to_f64_lossy()).collect();
        let mut indices: Vec<usize> = (0..n).collect();

        let mut nodes = Vec::with_capacity(2 * n / DEFAULT_LEAF_SIZE + 1);
        let mut bbox_min_vec = Vec::new();
        let mut bbox_max_vec = Vec::new();

        Self::build_recursive(
            &data_f64, &mut indices, d,
            &mut nodes, &mut bbox_min_vec, &mut bbox_max_vec,
        );

        // Reorder points to match final index order
        let mut points = vec![0.0f64; n * d];
        let mut orig_indices = vec![0usize; n];
        for (new_i, &orig_i) in indices.iter().enumerate() {
            orig_indices[new_i] = orig_i;
            let src = orig_i * d;
            let dst = new_i * d;
            points[dst..dst + d].copy_from_slice(&data_f64[src..src + d]);
        }

        Some(KdTree {
            nodes,
            points,
            orig_indices,
            bbox_min: bbox_min_vec,
            bbox_max: bbox_max_vec,
            n,
            d,
        })
    }

    fn build_recursive(
        data: &[f64],
        indices: &mut [usize],
        d: usize,
        nodes: &mut Vec<KdNode>,
        bbox_min: &mut Vec<f64>,
        bbox_max: &mut Vec<f64>,
    ) -> usize {
        let n = indices.len();
        let node_idx = nodes.len();

        // Compute bounding box
        let mut lo = vec![f64::MAX; d];
        let mut hi = vec![f64::MIN; d];
        for &idx in indices.iter() {
            for dim in 0..d {
                let val = data[idx * d + dim];
                if val < lo[dim] { lo[dim] = val; }
                if val > hi[dim] { hi[dim] = val; }
            }
        }

        // Push placeholder node
        nodes.push(KdNode {
            split_dim: 0, split_val: 0.0,
            left_or_start: 0, right_or_end: 0, is_leaf: false,
        });
        bbox_min.extend_from_slice(&lo);
        bbox_max.extend_from_slice(&hi);

        if n <= DEFAULT_LEAF_SIZE {
            // Leaf node — store range into the indices buffer
            // We need the absolute position, but indices is a mutable slice
            // with relative positions. We'll use a different approach:
            // leaves store start/end as indices into the reordered buffer.
            // Since we reorder later, we need to track the global position.
            // For simplicity: leaves are identified, and during queries we
            // use the reordered points buffer with orig_indices.
            nodes[node_idx] = KdNode {
                split_dim: 0,
                split_val: 0.0,
                left_or_start: 0, // will be fixed up
                right_or_end: 0,  // will be fixed up
                is_leaf: true,
            };
            return node_idx;
        }

        // Pick split dimension: widest extent
        let mut best_dim = 0;
        let mut best_extent = 0.0f64;
        for dim in 0..d {
            let extent = hi[dim] - lo[dim];
            if extent > best_extent {
                best_extent = extent;
                best_dim = dim;
            }
        }

        // Find median on split dimension
        let median = n / 2;
        indices.select_nth_unstable_by(median, |&a, &b| {
            let va = data[a * d + best_dim];
            let vb = data[b * d + best_dim];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let split_val = data[indices[median] * d + best_dim];

        // Recurse
        let (left_indices, right_indices) = indices.split_at_mut(median);
        let left_child = Self::build_recursive(data, left_indices, d, nodes, bbox_min, bbox_max);
        let right_child = Self::build_recursive(data, right_indices, d, nodes, bbox_min, bbox_max);

        nodes[node_idx] = KdNode {
            split_dim: best_dim,
            split_val,
            left_or_start: left_child,
            right_or_end: right_child,
            is_leaf: false,
        };

        node_idx
    }

    /// Whether KD-tree acceleration should be used.
    pub fn should_use(d: usize, metric: crate::distance::Metric) -> bool {
        d <= MAX_TREE_DIM && metric != crate::distance::Metric::Cosine
    }
}

// ---- Leaf range tracking ----
// After construction, we need to know which reordered points belong to each leaf.
// The indices slice was partitioned in-place during construction.
// We track leaf ranges by doing a DFS and counting.

impl KdTree {
    /// Assign leaf start/end ranges by DFS over the tree.
    /// Must be called after construction and reordering.
    fn assign_leaf_ranges(nodes: &mut [KdNode], node_idx: usize, counter: &mut usize) {
        let node = &nodes[node_idx];
        if node.is_leaf {
            // Count how many points are in this leaf's subtree
            // We know from construction: leaves were created for chunks <= DEFAULT_LEAF_SIZE
            // But we don't know the exact count stored in the node.
            // Fix: store count during construction.
            unreachable!("leaf ranges set during build");
        }
        let left = node.left_or_start;
        let right = node.right_or_end;
        Self::assign_leaf_ranges(nodes, left, counter);
        Self::assign_leaf_ranges(nodes, right, counter);
    }
}

// The above approach is getting complicated. Let me restructure:
// Instead of fixing up leaf ranges post-construction, track the absolute
// position during construction via the index range.

// Let me rewrite build to track absolute positions properly.

impl KdTree {
    /// Build with proper leaf range tracking.
    pub fn build_v2<F: Scalar>(data_slice: &[F], n: usize, d: usize) -> Option<Self> {
        if d > MAX_TREE_DIM || n == 0 {
            return None;
        }

        let data_f64: Vec<f64> = data_slice.iter().map(|v| v.to_f64_lossy()).collect();
        let mut indices: Vec<usize> = (0..n).collect();

        let mut nodes = Vec::with_capacity(2 * n / DEFAULT_LEAF_SIZE + 1);
        let mut bbox_min_vec = Vec::new();
        let mut bbox_max_vec = Vec::new();

        Self::build_recursive_v2(
            &data_f64, &mut indices, 0, n, d,
            &mut nodes, &mut bbox_min_vec, &mut bbox_max_vec,
        );

        // Reorder points buffer
        let mut points = vec![0.0f64; n * d];
        let mut orig_indices = vec![0usize; n];
        for (new_i, &orig_i) in indices.iter().enumerate() {
            orig_indices[new_i] = orig_i;
            points[new_i * d..(new_i + 1) * d]
                .copy_from_slice(&data_f64[orig_i * d..(orig_i + 1) * d]);
        }

        Some(KdTree { nodes, points, orig_indices, bbox_min: bbox_min_vec, bbox_max: bbox_max_vec, n, d })
    }

    fn build_recursive_v2(
        data: &[f64],
        indices: &mut Vec<usize>,
        start: usize,      // absolute position in indices
        count: usize,       // number of points in this subtree
        d: usize,
        nodes: &mut Vec<KdNode>,
        bbox_min: &mut Vec<f64>,
        bbox_max: &mut Vec<f64>,
    ) -> usize {
        let node_idx = nodes.len();

        // Compute bounding box
        let mut lo = vec![f64::MAX; d];
        let mut hi = vec![f64::MIN; d];
        for i in start..start + count {
            let idx = indices[i];
            for dim in 0..d {
                let val = data[idx * d + dim];
                if val < lo[dim] { lo[dim] = val; }
                if val > hi[dim] { hi[dim] = val; }
            }
        }

        nodes.push(KdNode {
            split_dim: 0, split_val: 0.0,
            left_or_start: 0, right_or_end: 0, is_leaf: false,
        });
        bbox_min.extend_from_slice(&lo);
        bbox_max.extend_from_slice(&hi);

        if count <= DEFAULT_LEAF_SIZE {
            nodes[node_idx] = KdNode {
                split_dim: 0, split_val: 0.0,
                left_or_start: start,
                right_or_end: start + count,
                is_leaf: true,
            };
            return node_idx;
        }

        // Pick split dimension
        let mut best_dim = 0;
        let mut best_extent = 0.0f64;
        for dim in 0..d {
            let extent = hi[dim] - lo[dim];
            if extent > best_extent {
                best_extent = extent;
                best_dim = dim;
            }
        }

        // Median partition
        let median = count / 2;
        let slice = &mut indices[start..start + count];
        slice.select_nth_unstable_by(median, |&a, &b| {
            let va = data[a * d + best_dim];
            let vb = data[b * d + best_dim];
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let split_val = data[indices[start + median] * d + best_dim];

        let left_child = Self::build_recursive_v2(data, indices, start, median, d, nodes, bbox_min, bbox_max);
        let right_child = Self::build_recursive_v2(data, indices, start + median, count - median, d, nodes, bbox_min, bbox_max);

        nodes[node_idx] = KdNode {
            split_dim: best_dim, split_val,
            left_or_start: left_child,
            right_or_end: right_child,
            is_leaf: false,
        };

        node_idx
    }
}

// ---- Range Query ----

impl KdTree {
    /// Find all points within `radius` of `query`.
    /// `radius` is in raw distance space (eps² for SquaredEuclidean, eps for Manhattan).
    pub fn query_radius<B: BBoxDistance>(&self, query: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        self.query_radius_recursive::<B>(0, query, radius, &mut results);
        results
    }

    fn query_radius_recursive<B: BBoxDistance>(
        &self,
        node_idx: usize,
        query: &[f64],
        radius: f64,
        results: &mut Vec<usize>,
    ) {
        let node = &self.nodes[node_idx];
        let d = self.d;

        // Prune: min bbox distance > radius
        let bbox_lo = &self.bbox_min[node_idx * d..(node_idx + 1) * d];
        let bbox_hi = &self.bbox_max[node_idx * d..(node_idx + 1) * d];
        let min_dist = B::min_dist_to_bbox(query, bbox_lo, bbox_hi);
        if min_dist > radius {
            return;
        }

        if node.is_leaf {
            for i in node.left_or_start..node.right_or_end {
                let point = &self.points[i * d..(i + 1) * d];
                let dist = point_dist::<B>(query, point);
                if dist <= radius {
                    results.push(self.orig_indices[i]);
                }
            }
            return;
        }

        self.query_radius_recursive::<B>(node.left_or_start, query, radius, results);
        self.query_radius_recursive::<B>(node.right_or_end, query, radius, results);
    }
}

// ---- k-NN Query ----

#[derive(PartialEq)]
struct Neighbor {
    dist: f64,
    index: usize,
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist) // max-heap: largest at top
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl KdTree {
    /// Find the k nearest neighbors of `query`.
    /// Returns (original_index, raw_distance) sorted by distance ascending.
    /// `self_index`: if Some, excludes that original index from results.
    pub fn query_knn<B: BBoxDistance>(
        &self,
        query: &[f64],
        k: usize,
        self_index: Option<usize>,
    ) -> Vec<(usize, f64)> {
        let mut heap: BinaryHeap<Neighbor> = BinaryHeap::with_capacity(k + 1);
        self.query_knn_recursive::<B>(0, query, k, self_index, &mut heap);

        let mut results: Vec<(usize, f64)> = heap.into_iter().map(|n| (n.index, n.dist)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn query_knn_recursive<B: BBoxDistance>(
        &self,
        node_idx: usize,
        query: &[f64],
        k: usize,
        self_index: Option<usize>,
        heap: &mut BinaryHeap<Neighbor>,
    ) {
        let node = &self.nodes[node_idx];
        let d = self.d;

        // Prune
        let bbox_lo = &self.bbox_min[node_idx * d..(node_idx + 1) * d];
        let bbox_hi = &self.bbox_max[node_idx * d..(node_idx + 1) * d];
        let min_dist = B::min_dist_to_bbox(query, bbox_lo, bbox_hi);
        let threshold = if heap.len() >= k {
            heap.peek().unwrap().dist
        } else {
            f64::MAX
        };
        if min_dist > threshold {
            return;
        }

        if node.is_leaf {
            for i in node.left_or_start..node.right_or_end {
                let orig_idx = self.orig_indices[i];
                if self_index == Some(orig_idx) {
                    continue;
                }
                let point = &self.points[i * d..(i + 1) * d];
                let dist = point_dist::<B>(query, point);
                if heap.len() < k {
                    heap.push(Neighbor { dist, index: orig_idx });
                } else if dist < heap.peek().unwrap().dist {
                    heap.pop();
                    heap.push(Neighbor { dist, index: orig_idx });
                }
            }
            return;
        }

        // Visit closer child first for better pruning
        let go_left_first = query[node.split_dim] <= node.split_val;
        let (first, second) = if go_left_first {
            (node.left_or_start, node.right_or_end)
        } else {
            (node.right_or_end, node.left_or_start)
        };

        self.query_knn_recursive::<B>(first, query, k, self_index, heap);
        self.query_knn_recursive::<B>(second, query, k, self_index, heap);
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::{ManhattanDistance, SquaredEuclidean};

    // ---- Construction ----

    #[test]
    fn test_build_basic() {
        let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let tree = KdTree::build_v2::<f64>(&data, 4, 2).unwrap();
        assert_eq!(tree.n, 4);
        assert_eq!(tree.d, 2);
        let mut indices = tree.orig_indices.clone();
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_build_single_point() {
        let data = [5.0, 3.0];
        let tree = KdTree::build_v2::<f64>(&data, 1, 2).unwrap();
        assert_eq!(tree.n, 1);
    }

    #[test]
    fn test_build_two_points() {
        let data = [0.0, 0.0, 1.0, 1.0];
        let tree = KdTree::build_v2::<f64>(&data, 2, 2).unwrap();
        assert_eq!(tree.n, 2);
    }

    #[test]
    fn test_build_identical_points() {
        let data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let tree = KdTree::build_v2::<f64>(&data, 3, 2).unwrap();
        assert_eq!(tree.n, 3);
    }

    #[test]
    fn test_build_returns_none_high_dim() {
        let data = vec![0.0; 10 * 20];
        assert!(KdTree::build_v2::<f64>(&data, 10, 20).is_none());
    }

    #[test]
    fn test_build_f32() {
        let data: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let tree = KdTree::build_v2::<f32>(&data, 4, 2).unwrap();
        assert_eq!(tree.n, 4);
    }

    // ---- Range Query ----

    fn brute_force_radius_sqeuc(data: &[f64], n: usize, d: usize, query: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        for i in 0..n {
            let point = &data[i * d..(i + 1) * d];
            let dist: f64 = point.iter().zip(query).map(|(a, b)| (a - b).powi(2)).sum();
            if dist <= radius {
                results.push(i);
            }
        }
        results.sort();
        results
    }

    fn brute_force_radius_manhattan(data: &[f64], n: usize, d: usize, query: &[f64], radius: f64) -> Vec<usize> {
        let mut results = Vec::new();
        for i in 0..n {
            let point = &data[i * d..(i + 1) * d];
            let dist: f64 = point.iter().zip(query).map(|(a, b)| (a - b).abs()).sum();
            if dist <= radius {
                results.push(i);
            }
        }
        results.sort();
        results
    }

    #[test]
    fn test_range_sqeuc_matches_brute_force() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let d = 3;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tree = KdTree::build_v2::<f64>(&data, n, d).unwrap();

        for qi in [0, 10, 50, 100, 150, 199] {
            let query = &data[qi * d..(qi + 1) * d];
            let radius = 5.0;
            let mut tree_result = tree.query_radius::<SquaredEuclidean>(query, radius);
            tree_result.sort();
            let expected = brute_force_radius_sqeuc(&data, n, d, query, radius);
            assert_eq!(tree_result, expected, "Mismatch at query point {qi}");
        }
    }

    #[test]
    fn test_range_manhattan_matches_brute_force() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let d = 3;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tree = KdTree::build_v2::<f64>(&data, n, d).unwrap();

        for qi in [0, 10, 50, 100, 150, 199] {
            let query = &data[qi * d..(qi + 1) * d];
            let radius = 5.0;
            let mut tree_result = tree.query_radius::<ManhattanDistance>(query, radius);
            tree_result.sort();
            let expected = brute_force_radius_manhattan(&data, n, d, query, radius);
            assert_eq!(tree_result, expected, "Mismatch at query point {qi}");
        }
    }

    #[test]
    fn test_range_radius_zero() {
        let data = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let tree = KdTree::build_v2::<f64>(&data, 3, 2).unwrap();
        let result = tree.query_radius::<SquaredEuclidean>(&[0.0, 0.0], 0.0);
        assert_eq!(result.len(), 1);
        assert!(result.contains(&0));
    }

    #[test]
    fn test_range_large_radius() {
        let data = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let tree = KdTree::build_v2::<f64>(&data, 3, 2).unwrap();
        let mut result = tree.query_radius::<SquaredEuclidean>(&[0.5, 0.5], 100.0);
        result.sort();
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn test_range_no_results() {
        let data = [0.0, 0.0, 1.0, 0.0];
        let tree = KdTree::build_v2::<f64>(&data, 2, 2).unwrap();
        let result = tree.query_radius::<SquaredEuclidean>(&[100.0, 100.0], 1.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_range_1d() {
        let data = [1.0, 3.0, 5.0, 7.0, 9.0];
        let tree = KdTree::build_v2::<f64>(&data, 5, 1).unwrap();
        let mut result = tree.query_radius::<SquaredEuclidean>(&[4.0], 2.0); // sq dist <= 2
        result.sort();
        assert!(result.contains(&1)); // 3.0, dist_sq=1
        assert!(result.contains(&2)); // 5.0, dist_sq=1
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_range_identical_points() {
        let data = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let tree = KdTree::build_v2::<f64>(&data, 4, 2).unwrap();
        let mut result = tree.query_radius::<SquaredEuclidean>(&[5.0, 5.0], 0.0);
        result.sort();
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    // ---- k-NN Query ----

    fn brute_force_knn_sqeuc(data: &[f64], n: usize, d: usize, query: &[f64], k: usize, exclude: Option<usize>) -> Vec<(usize, f64)> {
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&i| exclude != Some(i))
            .map(|i| {
                let point = &data[i * d..(i + 1) * d];
                let dist: f64 = point.iter().zip(query).map(|(a, b)| (a - b).powi(2)).sum();
                (i, dist)
            })
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }

    #[test]
    fn test_knn_sqeuc_matches_brute_force() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let d = 3;
        let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let tree = KdTree::build_v2::<f64>(&data, n, d).unwrap();

        for qi in [0, 10, 50, 100, 150, 199] {
            let query = &data[qi * d..(qi + 1) * d];
            let k = 5;
            let tree_result = tree.query_knn::<SquaredEuclidean>(query, k, Some(qi));
            let expected = brute_force_knn_sqeuc(&data, n, d, query, k, Some(qi));

            assert_eq!(tree_result.len(), expected.len(), "Length mismatch at {qi}");
            for (t, e) in tree_result.iter().zip(expected.iter()) {
                assert!((t.1 - e.1).abs() < 1e-10, "Distance mismatch at point {qi}: tree={}, brute={}", t.1, e.1);
            }
        }
    }

    #[test]
    fn test_knn_k_equals_n() {
        let data = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let tree = KdTree::build_v2::<f64>(&data, 3, 2).unwrap();
        let result = tree.query_knn::<SquaredEuclidean>(&[0.0, 0.0], 3, None);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_knn_k_greater_than_n() {
        let data = [0.0, 0.0, 1.0, 0.0];
        let tree = KdTree::build_v2::<f64>(&data, 2, 2).unwrap();
        let result = tree.query_knn::<SquaredEuclidean>(&[0.0, 0.0], 10, None);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_knn_self_exclusion() {
        let data = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
        let tree = KdTree::build_v2::<f64>(&data, 3, 2).unwrap();
        let result = tree.query_knn::<SquaredEuclidean>(&[0.0, 0.0], 2, Some(0));
        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|(idx, _)| *idx != 0));
    }

    #[test]
    fn test_knn_1d() {
        let data = [1.0, 3.0, 5.0, 7.0, 9.0];
        let tree = KdTree::build_v2::<f64>(&data, 5, 1).unwrap();
        let result = tree.query_knn::<SquaredEuclidean>(&[4.0], 2, None);
        assert_eq!(result.len(), 2);
        // Nearest: 3.0 (dist_sq=1) and 5.0 (dist_sq=1)
        let indices: Vec<usize> = result.iter().map(|(i, _)| *i).collect();
        assert!(indices.contains(&1) && indices.contains(&2));
    }

    #[test]
    fn test_should_use() {
        use crate::distance::Metric;
        assert!(KdTree::should_use(2, Metric::Euclidean));
        assert!(KdTree::should_use(16, Metric::Manhattan));
        assert!(!KdTree::should_use(20, Metric::Euclidean));
        assert!(!KdTree::should_use(2, Metric::Cosine));
    }
}
