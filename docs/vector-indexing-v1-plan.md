# rustcluster.index v1 Implementation Plan

## Goal

Replace FAISS in the production "item similarity graph" workload (Databricks notebook) with a pure-Rust implementation that integrates cleanly with the rest of `rustcluster`. v1 ships only what that workload needs; HNSW, PQ, and IVF land in subsequent versions.

The motivating workload, in one line:

```python
src_ids, dst_ids, scores = index.similarity_graph(embeddings, threshold=0.7, exclude_self=True)
```

…replacing this current code path:

```python
index = faiss.IndexFlatIP(d)
index.add(embeddings)
for i in range(0, n, batch_size):
    lims, distances, labels = index.range_search(embeddings[i:i+batch_size], 0.7)
    # manually map labels back to gc_item_number
    # manually filter self-matches
    # accumulate triples for Delta write
```

## Scope decisions (locked)

- **Submodule, not sibling crate.** `rustcluster.index` lives inside the existing crate, gated by Cargo feature `index` (default on). Cosine/IVF/HNSW/PQ become further features later. No second PyPI package.
- **f32 only.** No f64 dispatch in the index module. Embedding vectors are f32 in practice; f64 destroys cache locality. f32-only halves the surface area and unblocks SIMD.
- **Pure Rust.** No `usearch` C++ FFI, no `cxx` toolchain. Preserves the "pip install just works" promise.
- **External IDs from day one.** `add_with_ids(vectors, ids: u64)` is in v1, not v2. The notebook proves users need it.
- **FAISS-compatible signatures where reasonable.** A user should be able to `import rustcluster.index as faiss` and have most of their code work. We match `(lims, distances, labels)` for `range_search`, `(distances, labels)` for `search`.

## Out of scope for v1

- HNSW (next milestone)
- IVFFlat (milestone after that)
- Product Quantization (later still)
- Deletes
- Pre-filtering during traversal
- mmap loading
- Concurrent reads-during-writes

## Architecture

Three layers, matching the existing rustcluster pattern:

```
src/index/
  mod.rs               # public traits, type re-exports
  flat.rs              # IndexFlatL2, IndexFlatIP — owns vectors + IDs
  ids.rs               # IdMap — external u64 ↔ internal usize bijection
  topk.rs              # heap-based top-k selection
  range.rs             # range search emit logic (per-query CSR)
  similarity_graph.rs  # tiled all-pairs above-threshold kernel
  kernel.rs            # tile-shaped GEMM glue over faer
  persistence.rs       # safetensors + JSON save/load
src/lib.rs             # PyO3 bindings (pyclass wrappers, GIL release)
python/rustcluster/
  index.py             # re-exports + thin Python conveniences
```

### Layer 1 — PyO3 boundary (`src/lib.rs`)

For each Python-facing class:
- Validate dtype (must be f32 contiguous), shape (must be 2D), dim consistency vs index.
- Convert `numpy.PyReadonlyArray2<f32>` to `ndarray::ArrayView2<f32>`.
- `py.allow_threads(|| ...)` around all compute.
- Map Rust errors to `PyValueError` / `PyRuntimeError`.

### Layer 2 — Index logic (`src/index/flat.rs`)

```rust
pub trait VectorIndex {
    fn dim(&self) -> usize;
    fn ntotal(&self) -> usize;
    fn metric(&self) -> Metric;
    fn add(&mut self, vectors: ArrayView2<f32>) -> Result<()>;
    fn add_with_ids(&mut self, vectors: ArrayView2<f32>, ids: &[u64]) -> Result<()>;
    fn search(&self, queries: ArrayView2<f32>, k: usize, opts: SearchOpts)
        -> Result<SearchResult>;
    fn range_search(&self, queries: ArrayView2<f32>, threshold: f32, opts: SearchOpts)
        -> Result<RangeResult>;
}

pub struct SearchOpts {
    pub exclude_self: bool,   // skip i==j matches when query is from index
}

pub struct SearchResult {
    pub distances: Array2<f32>,  // (nq, k)
    pub labels: Array2<i64>,     // (nq, k); -1 for missing
}

pub struct RangeResult {
    pub lims: Vec<i64>,          // length nq+1
    pub distances: Vec<f32>,     // length lims[nq]
    pub labels: Vec<i64>,        // length lims[nq]
}

pub enum Metric { L2, InnerProduct }
```

`IndexFlatL2` and `IndexFlatIP` are concrete structs holding:
- `vectors: Array2<f32>` (n × d, owned, contiguous)
- `norms_sq: Vec<f32>` (cached squared norms for L2 distance trick)
- `ids: IdMap` (sequential by default; external u64 if `add_with_ids` was used)

### Layer 3 — Hot kernels

**Top-k (`topk.rs`):** binary max-heap of capacity k. For k ≤ 16 we add a specialized branchless insertion-sort variant. The heap is reused per query (reset, not realloc) when batching.

**Tile kernel (`kernel.rs`):**

```rust
// Compute X[i0..i1] @ Y[j0..j1]^T via faer, write into a tile-local buffer.
// Tile sizes chosen so the result tile fits in L1 (16-32 KB).
pub fn dot_tile(
    x: ArrayView2<f32>,    // (i1-i0, d)
    y: ArrayView2<f32>,    // (j1-j0, d)
    out: &mut Array2<f32>, // (i1-i0, j1-j0), pre-allocated
);

// L2 distance from cached IP using ||a-b||² = ||a||² + ||b||² - 2 a·b.
// Requires precomputed norms.
pub fn l2_from_ip(
    ip_tile: &Array2<f32>,
    x_norms: &[f32],
    y_norms: &[f32],
    out: &mut Array2<f32>,
);
```

**Similarity graph kernel (`similarity_graph.rs`):**

This is the heart of v1. It does NOT call `range_search` in a loop; it's a fused kernel that streams edges directly without materializing the full n×n matrix.

```rust
pub fn similarity_graph(
    x: ArrayView2<f32>,        // assumed L2-normalized for cosine
    threshold: f32,
    exclude_self: bool,
    ids: Option<&[u64]>,
) -> EdgeList {
    let n = x.nrows();
    let tile = pick_tile_size(x.ncols()); // ~256 for d=128, ~128 for d=1536

    // Iterate upper-triangle tiles (i0 ≤ j0). Symmetric self-similarity,
    // so we compute each pair once and emit both directions if needed.
    let tile_pairs: Vec<(usize, usize)> = upper_triangle_tiles(n, tile);

    let edges: Vec<EdgeChunk> = tile_pairs
        .par_iter()
        .map(|&(i0, j0)| {
            let mut local = EdgeChunk::with_capacity(1024);
            let mut score_tile = Array2::<f32>::zeros((tile, tile));
            dot_tile(slice_i, slice_j, &mut score_tile);
            for li in 0..tile_h {
                let j_start = if i0 == j0 { li + 1 } else { 0 };
                for lj in j_start..tile_w {
                    let s = score_tile[(li, lj)];
                    if s >= threshold {
                        let gi = i0 + li;
                        let gj = j0 + lj;
                        if exclude_self && gi == gj { continue; }
                        let src = ids.map(|m| m[gi]).unwrap_or(gi as u64);
                        let dst = ids.map(|m| m[gj]).unwrap_or(gj as u64);
                        local.push(src, dst, s);
                        local.push(dst, src, s); // emit both directions
                    }
                }
            }
            local
        })
        .collect();

    EdgeList::concat(edges) // three parallel Vecs (src, dst, score)
}
```

Why this beats FAISS's `range_search` loop for the notebook's workload:
- Single pass over the matrix; FAISS's per-query loop re-reads the database.
- Upper-triangle only — half the compute when query == database.
- No CSR materialization; emits flat edge triples ready for Delta.
- Per-tile output buffers, merged once at the end.

The Python API is a thin wrapper:

```python
class IndexFlatIP:
    def similarity_graph(self, threshold: float, exclude_self: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (src_ids, dst_ids, scores) as three NumPy arrays."""
```

Note: this method lives on the index, operating over its own stored vectors. There's also a free function `rustcluster.index.similarity_graph(X, threshold, ids=...)` for one-shot use.

## Public Python API (v1)

```python
from rustcluster.index import IndexFlatL2, IndexFlatIP

# Construction
index = IndexFlatIP(dim=1536)

# Add (sequential IDs)
index.add(embeddings)                             # f32, (n, dim)

# Add with external IDs
index.add_with_ids(embeddings, ids=gc_item_numbers)  # ids: u64 array

# Top-k
distances, labels = index.search(queries, k=10)
distances, labels = index.search(queries, k=10, exclude_self=True)

# Range search (FAISS-compatible)
lims, distances, labels = index.range_search(queries, threshold=0.7)
lims, distances, labels = index.range_search(queries, threshold=0.7, exclude_self=True)

# Similarity graph (the production workload)
src, dst, scores = index.similarity_graph(threshold=0.7, exclude_self=True)

# Persistence
index.save("items.rci/")
index = IndexFlatIP.load("items.rci/")

# Introspection
index.dim          # 1536
index.ntotal       # number of vectors
index.metric       # "ip" | "l2"
index.is_trained   # always True for Flat
```

## Persistence layout

A saved index is a directory:

```
items.rci/
  metadata.json     # { "type": "IndexFlatIP", "dim": 1536, "ntotal": 312000,
                    #   "metric": "ip", "format_version": 1, "has_ids": true }
  vectors.safetensors   # { "vectors": (n, d) f32 }
  ids.safetensors       # { "ids": (n,) u64 }   (only if has_ids)
```

Reuses the snapshot directory pattern. Future index types add tensors but keep `metadata.json` shape stable. `format_version` versioning from day one.

## The Databricks notebook, after v1

```python
from rustcluster.index import IndexFlatIP

embeddings = np.vstack(data["embedding"].tolist()).astype("float32")
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

index = IndexFlatIP(dim=embeddings.shape[1])
index.add_with_ids(embeddings, ids=np.array(gc_item_numbers, dtype=np.uint64))

src, dst, scores = index.similarity_graph(threshold=0.7, exclude_self=True)

batch_df = spark.createDataFrame(
    pd.DataFrame({"source_gc_item_number": src,
                  "similar_gc_item_number": dst,
                  "similarity_score": scores})
)
batch_df.write.format("delta").mode("overwrite").saveAsTable(similarity_index_table)
```

The batched-write loop disappears. The manual ID remapping disappears. The self-match filter disappears.

## Milestones

| # | Title | Deliverable | Estimated weeks |
|---|-------|-------------|-----------------|
| 1 | Skeleton + Flat top-k | `IndexFlatL2`, `IndexFlatIP`, `search(k)`, brute-force tests | 2 |
| 2 | External IDs + range_search | `add_with_ids`, `range_search`, `exclude_self` flag, FAISS-compatible CSR return | 1.5 |
| 3 | Tiled similarity_graph kernel | `similarity_graph` API + faer tile kernel + benchmark vs FAISS | 2 |
| 4 | Persistence + Python polish | safetensors save/load, NumPy interop, end-to-end notebook port | 1 |
| 5 | Hardening | Edge cases, error messages, docs, ann-benchmarks Recall@1=1.0 sanity | 1 |

**Total: ~7.5 engineer-weeks for the smallest useful v1.** Plan for ~10–12 to absorb the inevitable surprises (faer tile-shape tuning, PyO3 lifetime gotchas, safetensors edge cases).

Critical path: milestones 3 and 5. The tile kernel is where performance is won or lost; the hardening pass is where the API stops embarrassing us.

## Performance targets

For the production workload (n ≈ 300K, d = 1536, threshold = 0.7, normalized embeddings, single Apple M-series machine):

- **Correctness:** edge list bit-identical to FAISS `IndexFlatIP.range_search` (modulo floating-point rounding < 1e-5 absolute).
- **Throughput:** at least match FAISS wall time for the full pipeline. Target 1.5–3× faster on the similarity-graph path because of the symmetry and fused emission.
- **Memory:** peak below 2× the input matrix size. The output edge list dominates if threshold is loose; document that and let the user tune.

Benchmarks live in `benches/index_flat.rs` and `experiments/exp_index_similarity_graph.py`. Compare against `faiss-cpu` on the same machine, same inputs, same threshold sweep (0.5, 0.7, 0.85, 0.95).

## Tests

- **Brute-force parity.** For random inputs, `index.search(Q, k)` matches a numpy ground truth within float tolerance.
- **Range search parity.** `index.range_search` matches `faiss.IndexFlatIP.range_search` lims/distances/labels (sorted by distance within each query).
- **Similarity graph parity.** `similarity_graph(X, t)` matches the result of running `range_search(X, t)` and stitching, with `exclude_self`.
- **External ID round-trip.** `add_with_ids` then `search` returns the user's IDs.
- **Persistence round-trip.** Save then load yields an index that produces identical search results.
- **Edge cases.** Empty index, single vector, k larger than ntotal, threshold outside metric range, NaN inputs.
- **Property test.** Random vectors + random threshold: every emitted edge satisfies `score >= threshold`; no edge with `score >= threshold` is missing.

## Decisions

1. **Tile size: dim-aware constant chosen at construction.** Lookup:
   ```
   d ≤ 64    → tile = 256
   d ≤ 256   → tile = 128
   d ≤ 1024  → tile = 96
   d > 1024  → tile = 64
   ```
   Override via `RUSTCLUSTER_TILE_SIZE` env var for benchmarking. No runtime autotuning in v1.

2. **`similarity_graph` emits both directions by default.** Matches FAISS `range_search` behavior — each pair appears as `(i, j)` and `(j, i)`. Add a `unique_pairs: bool = False` flag to suppress duplicates (only emit where `i < j`). Default preserves drop-in compatibility with the existing FAISS notebook.

3. **No output cap in v1.** With a loose threshold the edge list can blow memory; the user picks the threshold and manages memory by batching inputs (as the current notebook already does). Document the n² worst case in the docstring. A streaming `similarity_graph_iter()` lands in v2.

## What's NOT happening in v1 (and why)

- **IVFFlat.** The notebook works fine with exact search at current scale. Add IVF when (a) someone has >5M vectors and Flat is the bottleneck, or (b) we need to index a separate query corpus from a frozen database.
- **HNSW.** Out of scope until we hit a top-k workload that exact search can't serve. When we do, wrap `hnsw_rs` (per the deep research).
- **PQ.** Only useful for memory-constrained workloads; not the case here.
- **Filters / pre-filter masks.** Post-filter in Python is fine for v1.

## v2 candidates (post-ship)

In rough priority order based on what unlocks the most value next:

1. **`IndexIVFFlat` built on `ClusterSnapshot`.** Train via existing `KMeans`, store inverted lists, support `search`/`range_search`/`similarity_graph` with `nprobe`. Lets users with millions of items skip the exact-search wall.
2. **Streaming `similarity_graph`.** Iterator/callback variant for cases where the edge list won't fit in memory.
3. **`add_with_ids` for IVF, with rebalancing constraints documented.**
4. **`IndexHNSW` wrapping `hnsw_rs`.** Once we have a top-k-heavy workload that justifies it.
5. **Pre-filter mask in `range_search` and `search`** (skip IDs in a bitset).
6. **IVF-PQ.** Memory compression, only when needed.
