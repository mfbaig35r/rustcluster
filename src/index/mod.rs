//! Vector indexing engine — flat exact search and (later) IVF/HNSW/PQ.
//!
//! v1 ships flat exact search (`IndexFlatL2`, `IndexFlatIP`) with top-k,
//! range search, and a fused all-pairs similarity-graph kernel.
//!
//! Three-layer architecture, matching the rest of the crate:
//! 1. PyO3 boundary in `src/lib.rs` — input validation, GIL release.
//! 2. Index logic in this module — `VectorIndex` trait, ID handling.
//! 3. Hot kernels in `kernel`, `topk` — raw `&[f32]` slices.
//!
//! All indexes are f32-only. f64 doubles cache footprint with no real
//! benefit for embedding workloads.

pub mod flat;
pub mod ids;
pub mod kernel;
pub mod similarity_graph;
pub mod topk;

pub use flat::{IndexFlatIP, IndexFlatL2};
pub use ids::IdMap;
pub use similarity_graph::EdgeList;

use ndarray::{Array2, ArrayView2};

use crate::error::ClusterError;

/// Distance metric a flat index operates under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance. Smaller = closer.
    L2,
    /// Inner product. Larger = closer. For cosine, normalize on add.
    InnerProduct,
}

impl Metric {
    pub fn as_str(&self) -> &'static str {
        match self {
            Metric::L2 => "l2",
            Metric::InnerProduct => "ip",
        }
    }
}

/// Options controlling a search.
#[derive(Debug, Clone, Copy, Default)]
pub struct SearchOpts {
    /// Skip entries whose external id equals the query's external id.
    /// Useful when the query set IS the indexed set.
    pub exclude_self: bool,
}

/// Result of a top-k search over a batch of queries.
#[derive(Debug)]
pub struct SearchResult {
    /// Distances/scores per query, shape (nq, k). Filled with sentinel
    /// (`f32::INFINITY` for L2, `f32::NEG_INFINITY` for IP) when fewer
    /// than k results were available.
    pub distances: Array2<f32>,
    /// External labels per query, shape (nq, k). `-1` for missing slots.
    pub labels: Array2<i64>,
}

/// Result of a range search over a batch of queries — CSR-shaped.
///
/// `lims[i]..lims[i+1]` index into `distances` and `labels` for query `i`.
/// Matches FAISS `range_search` semantics.
#[derive(Debug)]
pub struct RangeResult {
    pub lims: Vec<i64>,
    pub distances: Vec<f32>,
    pub labels: Vec<i64>,
}

/// Common interface for all vector indexes.
pub trait VectorIndex {
    fn dim(&self) -> usize;
    fn ntotal(&self) -> usize;
    fn metric(&self) -> Metric;

    /// Append vectors using sequential ids `ntotal..ntotal+n`.
    /// Errors if the index has previously seen explicit ids.
    fn add(&mut self, vectors: ArrayView2<f32>) -> Result<(), ClusterError>;

    /// Append vectors with explicit external u64 ids.
    /// Errors if the index has previously been populated with sequential ids
    /// or if `ids.len() != vectors.nrows()`.
    fn add_with_ids(&mut self, vectors: ArrayView2<f32>, ids: &[u64]) -> Result<(), ClusterError>;

    /// Top-k search.
    fn search(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
        opts: SearchOpts,
    ) -> Result<SearchResult, ClusterError>;

    /// Range search — return all neighbors within `threshold` of each query.
    /// For L2, "within" means `dist <= threshold`. For IP, `score >= threshold`.
    fn range_search(
        &self,
        queries: ArrayView2<f32>,
        threshold: f32,
        opts: SearchOpts,
    ) -> Result<RangeResult, ClusterError>;
}
