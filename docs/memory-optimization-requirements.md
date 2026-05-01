# Memory Optimization Requirements

## Problem Statement

rustcluster's PCA and clustering pipeline requires the full dataset in memory on a single node. On Databricks (28 GB Standard_DS4_v2), the JVM takes ~14-16 GB, leaving ~12 GB for Python. A 312K × 1536d embedding dataset consumes ~1.9 GB in f32, but the PCA transform path creates a 3.8 GB f64 copy internally (faer Mat::from_fn), causing OOM when combined with the Pandas DataFrame overhead (~3 GB of Python float objects during toPandas).

The current workaround is manual: fit PCA on a sample, transform in a Python loop, drop DataFrame columns aggressively. These should be built into the library.

## Memory Profile: 312K × 1536d → 128d Pipeline

### Current (v0.3.3)

| Stage | Peak Python Memory | Notes |
|-------|-------------------|-------|
| toPandas() with embedding lists | ~5 GB | Python list-of-lists + numpy copy coexist |
| After dropping embedding column | ~2 GB | numpy f32 array only |
| PCA fit (50K sample) | +2.5 GB | faer centered matrix f64 (50K × 1536 × 8) |
| PCA transform (312K full) | +3.8 GB | faer centered matrix f64 (312K × 1536 × 8) ← OOM |
| PCA transform (50K chunks) | +600 MB | faer centered matrix f64 (50K × 1536 × 8) ← workaround |
| Reduced data f64 (312K × 128) | 320 MB | Final output |
| Clustering (312K × 128) | ~50 MB | Minimal overhead |

### Target

| Stage | Peak Python Memory | How |
|-------|-------------------|-----|
| Arrow extraction | ~2 GB | Skip Python list overhead entirely |
| PCA fit (50K sample, f32) | +300 MB | f32 faer matrices (half of f64) |
| PCA transform (chunked, f32) | +300 MB | Built-in chunking + f32 path |
| Reduced data f32 (312K × 128) | 160 MB | f32 output |
| **Total peak** | **~2.6 GB** | Down from 8-9 GB |

## Requirements

### P0: Native Chunked Transform

**What:** Add `chunk_size` parameter to `EmbeddingReducer.transform()` and `fit_transform()`.

```python
# Current (manual chunking required)
chunks = []
for start in range(0, n, 50_000):
    chunks.append(reducer.transform(embeddings[start:start+50_000]))
X = np.vstack(chunks)

# Target
X = reducer.transform(embeddings, chunk_size=50_000)
```

**Implementation:**
- Python-side only — loop in `experimental.py`, call Rust transform per chunk, vstack
- Default `chunk_size=None` (current behavior for backward compatibility)
- When set, processes data in chunks and returns the concatenated result
- Memory: only one chunk's f64 matrix alive at a time

**Effort:** ~30 lines in `experimental.py`. No Rust changes.

### P0: Arrow-Based Embedding Extraction

**What:** A utility function to extract embeddings from a Spark DataFrame without going through Python lists (which add ~3 GB of overhead for 312K × 1536d).

```python
from rustcluster.utils import extract_embeddings_from_spark

# Current (3+ GB Python list overhead)
pdf = df.toPandas()
embeddings = np.array(pdf["embedding"].tolist(), dtype=np.float32)

# Target (zero Python list overhead)
embeddings, pdf = extract_embeddings_from_spark(
    df,
    embedding_col="embedding",
    metadata_cols=["parent_supplier_id", "supplier_name", "commodity", "sub_commodity"],
    dtype=np.float32,
)
```

**Implementation:**
- Uses `toLocalIterator()` to stream rows (JVM releases each batch after Python consumes it)
- Builds numpy array incrementally — never holds Python list-of-lists
- Returns metadata as a DataFrame (without embedding column)
- Optional `sample_n` parameter to extract only a random sample

**Effort:** ~60 lines. Pure Python utility, no Rust changes. Optional dependency on pyspark (only imported if used).

### P1: f32 PCA Path

**What:** Keep f32 through the PCA pipeline instead of converting to f64. Halves memory for all matrix operations.

Currently, `compute_pca()` and `project_data()` in `reduction.rs` work exclusively in f64:
- `Mat::from_fn(n, d, |i, j| data[i * d + j] - mean[j])` creates an f64 matrix regardless of input
- `project_data<F: Scalar>` converts input to f64 via `to_f64_lossy()` then back to F

**Target:**
- faer supports `Mat<f32>` — use it when input is f32
- Mean and components stored as f32 when fitted on f32 data
- Output is f32 when input is f32
- PCA quality at f32 is sufficient for embedding clustering (validated by existing test_pca_f32_projection)

**Impact:**
- fit: centered matrix goes from n × d × 8 to n × d × 4 (50K × 1536: 600 MB → 300 MB)
- transform: same halving (312K × 1536: 3.8 GB → 1.9 GB per chunk, or 300 MB → 150 MB per 50K chunk)
- Reduced output: 312K × 128 × 4 = 160 MB instead of 320 MB

**Considerations:**
- faer's f32 GEMM uses the same SIMD kernels — performance should be equal or better (more data fits in cache)
- SVD/eigendecomp at f32 for small matrices (k × k, k ≤ 140) — should be fine
- Save format would need to support f32 components (version 2 of RCPC format, or store dtype in header)
- Backward compatibility: f64 remains default, f32 opt-in via parameter

**Effort:** ~200 lines Rust. Moderate — need to make compute_pca and project_data generic over float type, or add f32 variants.

### P1: Sample-Fit Built Into EmbeddingReducer

**What:** Add `fit_sample_size` parameter so users don't have to manually sample.

```python
# Current
sample_idx = np.random.RandomState(42).choice(n, 50_000, replace=False)
reducer.fit(embeddings[sample_idx])

# Target
reducer = EmbeddingReducer(target_dim=128, fit_sample_size=50_000)
reducer.fit(embeddings)  # internally samples if n > fit_sample_size
```

**Implementation:**
- Python-side: if `fit_sample_size` is set and `n > fit_sample_size`, sample before calling Rust fit
- Rust side unchanged
- Default `None` (fit on full data for backward compatibility)

**Effort:** ~15 lines in `experimental.py`.

### P2: In-Place Transform

**What:** Avoid allocating the full faer `Mat` for transform by writing results directly into a pre-allocated output array.

Currently `project_data()` creates:
1. `Mat::from_fn(n, d, ...)` — full centered input matrix (n × d × 8 bytes)
2. `Mat::from_fn(d, target, ...)` — components matrix (small)
3. `result_mat = &x_centered * &comp_mat` — output matrix (n × target × 8)

The centered matrix is the memory hog. An in-place approach would:
1. Process rows in blocks (e.g., 10K rows)
2. Center block into a small buffer
3. Multiply block × components into output slice
4. Reuse the centering buffer

**Impact:** Transform peak drops from O(n × d) to O(block_size × d). For block=10K, d=1536: 120 MB instead of 3.8 GB.

**Effort:** ~150 lines Rust. Moderate — need to manage faer matrix views and output slicing.

### P2: Streaming PCA Fit (Incremental Covariance)

**What:** Fit PCA without materializing the full n × d centered matrix by computing the covariance matrix incrementally.

The randomized PCA algorithm requires Y = X_centered × Omega (n × d × k matmul). This can be computed in blocks:

```
Y = Σ_blocks (X_block - mean) × Omega
```

Each block contributes to Y additively. Only one block's centered data is alive at a time.

**Requirements:**
- Two-pass: first pass computes mean, second pass computes Y
- Or single-pass with Welford's online mean + streaming matmul
- QR and eigendecomp operate on Y (n × k) and small matrices — no change needed

**Impact:** fit() peak drops from O(n × d) to O(block_size × d + n × k). For 312K × 1536, k=138: 3.8 GB → 120 MB + 345 MB.

**Effort:** ~300 lines Rust. Significant — restructures compute_pca internals.

### P3: Spark UDF Wrapper

**What:** Run clustering as a Spark pandas UDF so computation happens on executors, not the driver.

```python
from rustcluster.spark import cluster_udf

result_df = df.groupBy("commodity_cluster").applyInPandas(
    cluster_udf(n_clusters=25, reduction_dim=None),
    schema=output_schema,
)
```

**When useful:** When the driver node is small but executors have plenty of memory. Not needed for most cases — single-node clustering on reduced data is fast enough.

**Effort:** ~100 lines Python wrapper. Requires careful schema handling.

## Implementation Order

```
Phase 1 (quick wins, no Rust changes):
  P0: Native chunked transform        ~30 lines Python
  P0: Arrow extraction helper          ~60 lines Python
  P1: Sample-fit parameter             ~15 lines Python

Phase 2 (Rust changes):
  P1: f32 PCA path                     ~200 lines Rust
  P2: In-place transform               ~150 lines Rust

Phase 3 (algorithmic changes):
  P2: Streaming PCA fit                ~300 lines Rust
  P3: Spark UDF wrapper                ~100 lines Python
```

## Verification

For each optimization:
1. Existing tests pass (182 Rust + 234 Python)
2. Memory profiled with `tracemalloc` on 312K × 1536d dataset
3. Output is numerically equivalent (or within f32 tolerance for P1)
4. Benchmarked on Databricks Standard_DS4_v2 (28 GB) — no OOM
5. Performance does not regress (chunked transform may add <1s overhead)

## Success Criteria

The full pipeline (load → reduce → cluster → save) runs on a **28 GB single-node Databricks cluster** with 312K × 1536d embeddings without any manual memory management by the user. No `gc.collect()`, no manual chunking, no dropping columns.

```python
# This should Just Work on 28 GB
from rustcluster.utils import extract_embeddings_from_spark
from rustcluster.experimental import EmbeddingReducer, EmbeddingCluster

embeddings, pdf = extract_embeddings_from_spark(df, "embedding", metadata_cols)
reducer = EmbeddingReducer(target_dim=128, fit_sample_size=50_000)
X = reducer.fit_transform(embeddings, chunk_size=50_000)
model = EmbeddingCluster(n_clusters=20, reduction_dim=None).fit(X)
```
