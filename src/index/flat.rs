//! Flat exact indexes: `IndexFlatL2` and `IndexFlatIP`.
//!
//! Owns the database vectors as a `(ntotal, d) f32` matrix. Search is
//! computed as one big GEMM `Q @ X^T` followed by per-query top-k selection.
//!
//! For L2, squared norms `||x||²` are cached on `add` so the L2 conversion
//! `||q-x||² = ||q||² + ||x||² - 2 q·x` is a single fused pass over the
//! score matrix.

use faer::Par;
use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::error::ClusterError;

use super::ids::IdMap;
use super::kernel::{ip_batch, ip_to_l2_sq, row_norms_sq};
use super::similarity_graph::{similarity_graph_ip, similarity_graph_l2, EdgeList};
use super::topk::{topk, Direction};
use super::{Metric, RangeResult, SearchOpts, SearchResult, VectorIndex};

/// Flat exact L2 (squared Euclidean) index.
///
/// Distances returned by `search`/`range_search` are **squared** L2 distances,
/// matching FAISS `IndexFlatL2`. Take `sqrt` at the boundary if needed.
#[derive(Debug, Clone)]
pub struct IndexFlatL2 {
    dim: usize,
    vectors: Array2<f32>,
    norms_sq: Vec<f32>,
    ids: IdMap,
}

impl IndexFlatL2 {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Array2::<f32>::zeros((0, dim)),
            norms_sq: Vec::new(),
            ids: IdMap::new(),
        }
    }

    pub fn vectors(&self) -> ArrayView2<'_, f32> {
        self.vectors.view()
    }

    pub fn ids(&self) -> &IdMap {
        &self.ids
    }

    /// Emit every pair `(i, j)` with `i != j` whose squared L2 distance is
    /// `<= threshold`. Returns three parallel vectors `(src_id, dst_id, score)`.
    ///
    /// `unique_pairs = true` emits only `(i, j)` with `i < j` (one row per
    /// undirected pair). Default emits both directions.
    pub fn similarity_graph(&self, threshold: f32, unique_pairs: bool) -> EdgeList {
        similarity_graph_l2(
            self.vectors.view(),
            &self.norms_sq,
            &self.ids,
            threshold,
            unique_pairs,
        )
    }

    /// Persistence accessors (crate-private — used by `index::persistence`).
    pub(crate) fn dim_internal(&self) -> usize {
        self.dim
    }
    pub(crate) fn vectors_internal(&self) -> &Array2<f32> {
        &self.vectors
    }
    pub(crate) fn ids_internal(&self) -> &IdMap {
        &self.ids
    }

    /// Reconstruct from raw parts (used by `index::persistence::load_flat_l2`).
    pub(crate) fn from_parts(dim: usize, vectors: Array2<f32>, ids: IdMap) -> Self {
        let norms_sq = row_norms_sq(vectors.view());
        Self {
            dim,
            vectors,
            norms_sq,
            ids,
        }
    }
}

impl VectorIndex for IndexFlatL2 {
    fn dim(&self) -> usize {
        self.dim
    }
    fn ntotal(&self) -> usize {
        self.vectors.nrows()
    }
    fn metric(&self) -> Metric {
        Metric::L2
    }

    fn add(&mut self, vectors: ArrayView2<f32>) -> Result<(), ClusterError> {
        validate_for_add(self.dim, &vectors)?;
        self.ids.extend_sequential(vectors.nrows())?;
        append_rows(&mut self.vectors, vectors);
        self.norms_sq.extend(row_norms_sq(vectors));
        Ok(())
    }

    fn add_with_ids(&mut self, vectors: ArrayView2<f32>, ids: &[u64]) -> Result<(), ClusterError> {
        validate_for_add(self.dim, &vectors)?;
        if ids.len() != vectors.nrows() {
            return Err(ClusterError::IndexIdModeConflict(format!(
                "ids length {} != vectors {}",
                ids.len(),
                vectors.nrows()
            )));
        }
        self.ids.extend_explicit(ids)?;
        append_rows(&mut self.vectors, vectors);
        self.norms_sq.extend(row_norms_sq(vectors));
        Ok(())
    }

    fn search(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        opts: SearchOpts,
    ) -> Result<SearchResult, ClusterError> {
        validate_for_query(self.dim, &queries, k, self.ntotal())?;
        let mut scores = ip_batch(queries, self.vectors.view(), Par::rayon(0));
        let q_norms = row_norms_sq(queries);
        ip_to_l2_sq(&mut scores, &q_norms, &self.norms_sq);
        Ok(materialize_topk(
            scores,
            k,
            Direction::Smallest,
            &self.ids,
            opts,
            queries,
            &self.vectors,
        ))
    }

    fn range_search(
        &self,
        queries: ArrayView2<f32>,
        threshold: f32,
        opts: SearchOpts,
    ) -> Result<RangeResult, ClusterError> {
        validate_for_query(self.dim, &queries, 1, self.ntotal())?;
        let mut scores = ip_batch(queries, self.vectors.view(), Par::rayon(0));
        let q_norms = row_norms_sq(queries);
        ip_to_l2_sq(&mut scores, &q_norms, &self.norms_sq);
        Ok(materialize_range(
            scores,
            threshold,
            Direction::Smallest,
            &self.ids,
            opts,
            queries,
            &self.vectors,
        ))
    }
}

/// Flat exact inner-product index. Higher score = closer.
///
/// For cosine similarity, L2-normalize vectors before `add` and queries
/// before `search`/`range_search`.
#[derive(Debug, Clone)]
pub struct IndexFlatIP {
    dim: usize,
    vectors: Array2<f32>,
    ids: IdMap,
}

impl IndexFlatIP {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            vectors: Array2::<f32>::zeros((0, dim)),
            ids: IdMap::new(),
        }
    }

    pub fn vectors(&self) -> ArrayView2<'_, f32> {
        self.vectors.view()
    }

    pub fn ids(&self) -> &IdMap {
        &self.ids
    }

    /// Emit every pair `(i, j)` with `i != j` whose inner product is
    /// `>= threshold`. Returns three parallel vectors `(src_id, dst_id, score)`.
    ///
    /// For cosine similarity, ensure vectors were L2-normalized before `add`.
    /// `unique_pairs = true` emits only `(i, j)` with `i < j`.
    pub fn similarity_graph(&self, threshold: f32, unique_pairs: bool) -> EdgeList {
        similarity_graph_ip(self.vectors.view(), &self.ids, threshold, unique_pairs)
    }

    pub(crate) fn dim_internal(&self) -> usize {
        self.dim
    }
    pub(crate) fn vectors_internal(&self) -> &Array2<f32> {
        &self.vectors
    }
    pub(crate) fn ids_internal(&self) -> &IdMap {
        &self.ids
    }

    pub(crate) fn from_parts(dim: usize, vectors: Array2<f32>, ids: IdMap) -> Self {
        Self { dim, vectors, ids }
    }
}

impl VectorIndex for IndexFlatIP {
    fn dim(&self) -> usize {
        self.dim
    }
    fn ntotal(&self) -> usize {
        self.vectors.nrows()
    }
    fn metric(&self) -> Metric {
        Metric::InnerProduct
    }

    fn add(&mut self, vectors: ArrayView2<f32>) -> Result<(), ClusterError> {
        validate_for_add(self.dim, &vectors)?;
        self.ids.extend_sequential(vectors.nrows())?;
        append_rows(&mut self.vectors, vectors);
        Ok(())
    }

    fn add_with_ids(&mut self, vectors: ArrayView2<f32>, ids: &[u64]) -> Result<(), ClusterError> {
        validate_for_add(self.dim, &vectors)?;
        if ids.len() != vectors.nrows() {
            return Err(ClusterError::IndexIdModeConflict(format!(
                "ids length {} != vectors {}",
                ids.len(),
                vectors.nrows()
            )));
        }
        self.ids.extend_explicit(ids)?;
        append_rows(&mut self.vectors, vectors);
        Ok(())
    }

    fn search(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        opts: SearchOpts,
    ) -> Result<SearchResult, ClusterError> {
        validate_for_query(self.dim, &queries, k, self.ntotal())?;
        let scores = ip_batch(queries, self.vectors.view(), Par::rayon(0));
        Ok(materialize_topk(
            scores,
            k,
            Direction::Largest,
            &self.ids,
            opts,
            queries,
            &self.vectors,
        ))
    }

    fn range_search(
        &self,
        queries: ArrayView2<f32>,
        threshold: f32,
        opts: SearchOpts,
    ) -> Result<RangeResult, ClusterError> {
        validate_for_query(self.dim, &queries, 1, self.ntotal())?;
        let scores = ip_batch(queries, self.vectors.view(), Par::rayon(0));
        Ok(materialize_range(
            scores,
            threshold,
            Direction::Largest,
            &self.ids,
            opts,
            queries,
            &self.vectors,
        ))
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn validate_for_add(dim: usize, vectors: &ArrayView2<f32>) -> Result<(), ClusterError> {
    let (n, d) = vectors.dim();
    if d != dim {
        return Err(ClusterError::DimensionMismatch {
            expected: dim,
            got: d,
        });
    }
    if n == 0 {
        return Ok(()); // adding zero rows is a no-op, not an error
    }
    if !vectors.iter().all(|v| v.is_finite()) {
        return Err(ClusterError::NonFinite);
    }
    Ok(())
}

fn validate_for_query(
    dim: usize,
    queries: &ArrayView2<f32>,
    k: usize,
    ntotal: usize,
) -> Result<(), ClusterError> {
    if ntotal == 0 {
        return Err(ClusterError::IndexEmpty);
    }
    if k == 0 {
        return Err(ClusterError::InvalidK(0));
    }
    let (_, d) = queries.dim();
    if d != dim {
        return Err(ClusterError::DimensionMismatch {
            expected: dim,
            got: d,
        });
    }
    if !queries.iter().all(|v| v.is_finite()) {
        return Err(ClusterError::NonFinite);
    }
    Ok(())
}

fn append_rows(dst: &mut Array2<f32>, src: ArrayView2<f32>) {
    if src.nrows() == 0 {
        return;
    }
    if dst.nrows() == 0 {
        *dst = src.to_owned();
    } else {
        // Vertical concat. ndarray's `append` along Axis(0) handles this.
        dst.append(Axis(0), src).expect("dim checked by caller");
    }
}

/// If `exclude_self` is set and the query vector exactly equals an indexed
/// vector at position `p`, return `Some(p)` so top-k/range can skip it.
///
/// "Exactly equals" means bitwise on the f32 values. This is the FAISS
/// convention — useful when the query batch is a slice of the database
/// vectors themselves.
fn self_position(
    enabled: bool,
    query: ndarray::ArrayView1<f32>,
    data: &Array2<f32>,
) -> Option<usize> {
    if !enabled {
        return None;
    }
    let q = query.as_slice().expect("contiguous query");
    let n = data.nrows();
    let d = data.ncols();
    let slice = data.as_slice().expect("contiguous data");
    for i in 0..n {
        let row = &slice[i * d..(i + 1) * d];
        if row == q {
            return Some(i);
        }
    }
    None
}

fn materialize_topk(
    scores: Array2<f32>,
    k: usize,
    direction: Direction,
    ids: &IdMap,
    opts: SearchOpts,
    queries: ArrayView2<f32>,
    data: &Array2<f32>,
) -> SearchResult {
    let nq = scores.nrows();

    // Each query is independent — partition + sort in parallel, then write
    // the results into the output arrays in order.
    let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let row = scores.row(i);
            let row_slice = row.as_slice().expect("contiguous score row");
            let skip = self_position(opts.exclude_self, queries.row(i), data);
            topk(row_slice, k, direction, skip)
        })
        .collect();

    let mut distances = Array2::<f32>::from_elem((nq, k), direction.sentinel());
    let mut labels = Array2::<i64>::from_elem((nq, k), -1);
    for (i, (vals, positions)) in results.into_iter().enumerate() {
        for (j, (v, p)) in vals.iter().zip(positions.iter()).enumerate() {
            distances[(i, j)] = *v;
            labels[(i, j)] = if *p < 0 {
                -1
            } else {
                ids.external(*p as usize) as i64
            };
        }
    }

    SearchResult { distances, labels }
}

fn materialize_range(
    scores: Array2<f32>,
    threshold: f32,
    direction: Direction,
    ids: &IdMap,
    opts: SearchOpts,
    queries: ArrayView2<f32>,
    data: &Array2<f32>,
) -> RangeResult {
    let nq = scores.nrows();

    // Each query produces its own (distances, labels) chunk in parallel.
    // `par_iter().collect()` preserves order, which is required for the
    // CSR-shape `lims` array to come out right.
    let chunks: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
        .into_par_iter()
        .map(|i| {
            let row = scores.row(i);
            let skip = self_position(opts.exclude_self, queries.row(i), data);
            let mut local_d: Vec<f32> = Vec::new();
            let mut local_l: Vec<i64> = Vec::new();
            for (pos, &score) in row.iter().enumerate() {
                if let Some(s) = skip {
                    if pos == s {
                        continue;
                    }
                }
                let keep = match direction {
                    Direction::Smallest => score <= threshold,
                    Direction::Largest => score >= threshold,
                };
                if keep {
                    local_d.push(score);
                    local_l.push(ids.external(pos) as i64);
                }
            }
            (local_d, local_l)
        })
        .collect();

    let total: usize = chunks.iter().map(|(d, _)| d.len()).sum();
    let mut lims: Vec<i64> = Vec::with_capacity(nq + 1);
    let mut distances: Vec<f32> = Vec::with_capacity(total);
    let mut labels: Vec<i64> = Vec::with_capacity(total);
    lims.push(0i64);
    for (d, l) in chunks {
        distances.extend(d);
        labels.extend(l);
        lims.push(distances.len() as i64);
    }

    RangeResult {
        lims,
        distances,
        labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    fn brute_force_l2(
        queries: ArrayView2<f32>,
        data: ArrayView2<f32>,
        k: usize,
    ) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
        let (nq, _) = queries.dim();
        let (n, _) = data.dim();
        let mut all_dists = Vec::with_capacity(nq);
        let mut all_labels = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = queries.row(qi);
            let mut entries: Vec<(f32, usize)> = (0..n)
                .map(|i| {
                    let x = data.row(i);
                    let mut acc = 0.0f32;
                    for j in 0..q.len() {
                        let diff = q[j] - x[j];
                        acc += diff * diff;
                    }
                    (acc, i)
                })
                .collect();
            entries.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let take = k.min(entries.len());
            all_dists.push(entries[..take].iter().map(|(d, _)| *d).collect());
            all_labels.push(entries[..take].iter().map(|(_, p)| *p).collect());
        }
        (all_dists, all_labels)
    }

    #[test]
    fn flat_l2_matches_brute_force() {
        let data = arr2(&[
            [0.0f32, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ]);
        let queries = arr2(&[[0.1f32, 0.1], [4.0, 4.0]]);

        let mut idx = IndexFlatL2::new(2);
        idx.add(data.view()).unwrap();

        let res = idx
            .search(queries.view(), 3, SearchOpts::default())
            .unwrap();
        let (bf_d, bf_l) = brute_force_l2(queries.view(), data.view(), 3);

        for q in 0..2 {
            for j in 0..3 {
                assert!((res.distances[(q, j)] - bf_d[q][j]).abs() < 1e-5);
                assert_eq!(res.labels[(q, j)], bf_l[q][j] as i64);
            }
        }
    }

    #[test]
    fn flat_ip_orders_by_largest() {
        let data = arr2(&[[1.0f32, 0.0], [0.0, 1.0], [0.7071, 0.7071]]);
        let queries = arr2(&[[1.0f32, 0.0]]);

        let mut idx = IndexFlatIP::new(2);
        idx.add(data.view()).unwrap();

        let res = idx
            .search(queries.view(), 3, SearchOpts::default())
            .unwrap();
        // Best match is index 0 (score 1.0), then 2 (~0.707), then 1 (0.0).
        assert_eq!(res.labels[(0, 0)], 0);
        assert_eq!(res.labels[(0, 1)], 2);
        assert_eq!(res.labels[(0, 2)], 1);
    }

    #[test]
    fn add_with_ids_returns_external_labels() {
        let data = arr2(&[[1.0f32, 0.0], [0.0, 1.0]]);
        let mut idx = IndexFlatIP::new(2);
        idx.add_with_ids(data.view(), &[100, 200]).unwrap();

        let q = arr2(&[[1.0f32, 0.0]]);
        let res = idx.search(q.view(), 1, SearchOpts::default()).unwrap();
        assert_eq!(res.labels[(0, 0)], 100);
    }

    #[test]
    fn exclude_self_skips_exact_match() {
        let data = arr2(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let mut idx = IndexFlatL2::new(2);
        idx.add(data.view()).unwrap();

        let q = arr2(&[[1.0f32, 0.0]]);
        let opts = SearchOpts { exclude_self: true };
        let res = idx.search(q.view(), 1, opts).unwrap();
        assert_ne!(res.labels[(0, 0)], 1);
    }

    #[test]
    fn range_search_l2_threshold() {
        let data = arr2(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]]);
        let mut idx = IndexFlatL2::new(2);
        idx.add(data.view()).unwrap();

        let q = arr2(&[[0.0f32, 0.0]]);
        let res = idx
            .range_search(q.view(), 1.5, SearchOpts::default())
            .unwrap();
        assert_eq!(res.lims, vec![0, 3]); // 0,1,2 within sq-dist 1.5
        assert_eq!(res.distances.len(), 3);
        assert_eq!(res.labels.len(), 3);
    }

    #[test]
    fn range_search_ip_threshold() {
        let data = arr2(&[[1.0f32, 0.0], [0.7071, 0.7071], [0.0, 1.0], [-1.0, 0.0]]);
        let mut idx = IndexFlatIP::new(2);
        idx.add(data.view()).unwrap();

        let q = arr2(&[[1.0f32, 0.0]]);
        let res = idx
            .range_search(q.view(), 0.5, SearchOpts::default())
            .unwrap();
        // Above-threshold matches: index 0 (1.0), index 1 (~0.7071). Index 2 = 0,
        // index 3 = -1 → both below.
        assert_eq!(res.lims, vec![0, 2]);
    }

    #[test]
    fn empty_index_search_errors() {
        let idx = IndexFlatL2::new(3);
        let q = arr2(&[[0.0f32, 0.0, 0.0]]);
        assert!(idx.search(q.view(), 1, SearchOpts::default()).is_err());
    }

    #[test]
    fn dim_mismatch_errors() {
        let mut idx = IndexFlatL2::new(3);
        idx.add(arr2(&[[1.0f32, 0.0, 0.0]]).view()).unwrap();
        let q = arr2(&[[1.0f32, 0.0]]); // wrong d
        assert!(idx.search(q.view(), 1, SearchOpts::default()).is_err());
    }
}
