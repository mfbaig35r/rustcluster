# EmbeddingReducer: Requirements & Design

## Problem Statement

Experiment 6 proved that PCA is 90% of the EmbeddingCluster pipeline runtime (570s of 637s). Users who want to try multiple clustering configurations (different K, different algorithms, different metrics) currently re-run PCA every time. This is wasteful — PCA is a data transformation, not an algorithm. Its output should be computed once and reused.

Additionally, the 60-second target requires either skipping PCA (Matryoshka) or making PCA dramatically faster (faer/BLAS). Both paths benefit from PCA being a first-class artifact rather than a hidden internal step.

## Requirements

### Core: EmbeddingReducer as Standalone Transformer

```python
from rustcluster.experimental import EmbeddingReducer

# Fit and transform (pay the cost once)
reducer = EmbeddingReducer(target_dim=128, method="pca")
X_reduced = reducer.fit_transform(embeddings)  # (323K, 1536) → (323K, 128)

# Transform new data with the same projection
X_new_reduced = reducer.transform(new_embeddings)

# Save/load fitted state
reducer.save("cross_pca_128.bin")
reducer = EmbeddingReducer.load("cross_pca_128.bin")
```

**Must support:**
- `fit(X)` — compute PCA components from X, store fitted state
- `transform(X)` — project X using fitted components, L2-renormalize
- `fit_transform(X)` — fit + transform in one call
- `save(path)` — serialize fitted state (mean, components, target_dim) to disk
- `load(path)` — deserialize and reconstruct
- `method="pca"` — randomized PCA (default)
- `method="matryoshka"` — prefix truncation (no fitting needed, just slices columns)
- f32 and f64 input support
- L2 re-normalization after reduction (cosine geometry preservation)

**Fitted state to persist:**
- `mean`: Vec<f64>, shape (d,) — column means for centering
- `components`: Vec<f64>, shape (d, target_dim) — projection matrix
- `target_dim`: usize
- `input_dim`: usize
- `method`: String

**Serialization format:** Simple binary with header (magic bytes, version, dims) followed by flat f64 arrays. No external dependency (not pickle — this is a Rust-native format that can be loaded without Python).

### Core: Expose Reduced Data from EmbeddingCluster

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=128)
model.fit(embeddings)

# Access the intermediate reduced data
X_reduced = model.reduced_data_  # (323K, 128) numpy array, L2-normalized
np.save("cross_reduced_128.npy", X_reduced)
```

**Must:**
- Cache the reduced data after PCA in the EmbeddingCluster fitted state
- Expose as a property returning a numpy array
- Return None if `reduction_dim=None` (no reduction was performed)
- Return a copy (not a reference to internal state)

### Core: Accept Pre-Reduced Data

```python
# Skip PCA entirely if data is already reduced
X_reduced = np.load("cross_reduced_128.npy")
model = EmbeddingCluster(n_clusters=50, reduction_dim=None)
model.fit(X_reduced)  # 128d input, no PCA, ~5s
```

This already works today by setting `reduction_dim=None`. No code changes needed — just documentation.

### Integration: Use EmbeddingReducer Output with Any Algorithm

```python
reducer = EmbeddingReducer(target_dim=128)
X_reduced = reducer.fit_transform(embeddings)
np.save("reduced.npy", X_reduced)

# All of these work on the cheap reduced data:
from rustcluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
from rustcluster.experimental import EmbeddingCluster

KMeans(n_clusters=50, metric="cosine").fit(X_reduced)
DBSCAN(eps=0.3, metric="cosine").fit(X_reduced)
HDBSCAN(min_cluster_size=100, metric="cosine").fit(X_reduced)
AgglomerativeClustering(n_clusters=50, linkage="average", metric="cosine").fit(X_reduced)
EmbeddingCluster(n_clusters=50, reduction_dim=None).fit(X_reduced)
```

No code changes needed for the existing algorithms — they already accept arbitrary 2D arrays. The key documentation point: use `metric="cosine"` when clustering PCA-reduced embeddings (the data is L2-normalized after reduction).

## Speed Requirements

### PCA Backend: faer Integration

The current PCA bottleneck is the hand-rolled rayon matmul for Y = X·Omega (323K × 1536 × 138). Replace with faer for 3-5x speedup:

| Backend | Y = X·Omega Time (est.) | Total PCA Time (est.) |
|---------|------------------------|----------------------|
| Current (rayon row-blocks) | ~500s | ~570s |
| faer (blocked matmul) | ~100-170s | ~130-200s |
| OpenBLAS/MKL | ~50-80s | ~80-120s |

**faer is the recommended first step** — pure Rust, no system dependency, preserves the "pip install and it works" story.

Target: reduce PCA from 570s to under 200s for 323K × 1536 → 128.

### Matryoshka Documentation

For users creating new embeddings from models that support prefix truncation:

```python
# At embedding time — request 128d directly
response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small",
    dimensions=128  # ← this eliminates PCA entirely
)

# No PCA needed — cluster directly
model = EmbeddingCluster(n_clusters=50, reduction_dim=None)
model.fit(embeddings_128d)  # ~5s instead of ~637s
```

This should be prominently documented in the embedding clustering guide.

## Architecture

### Rust Side

```
src/embedding/
  reducer.rs  (new)
    - EmbeddingReducerState { mean, components, target_dim, input_dim, method }
    - fit_pca(data, n, d, target_dim) → EmbeddingReducerState
    - transform(data, n, d, state) → Vec<f64>
    - save_state(state, path) → Result
    - load_state(path) → Result<EmbeddingReducerState>
```

### PyO3 Side

```rust
#[pyclass]
struct EmbeddingReducer {
    state: Option<EmbeddingReducerState>,
    target_dim: usize,
    method: String,
}

#[pymethods]
impl EmbeddingReducer {
    fn fit(&mut self, x) -> PyResult<()>
    fn transform(&self, x) -> PyResult<PyArray2<f64>>
    fn fit_transform(&mut self, x) -> PyResult<PyArray2<f64>>
    fn save(&self, path: &str) -> PyResult<()>
    fn load(path: &str) -> PyResult<Self>  // static method
}
```

### Python Side

```python
# In rustcluster/experimental.py
class EmbeddingReducer:
    def __init__(self, target_dim=128, method="pca"):
        ...
    def fit(self, X): ...
    def transform(self, X): ...
    def fit_transform(self, X): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

## Non-Goals

- **Not building a general PCA library.** This is specifically for embedding preprocessing.
- **Not supporting sparse data.** Embeddings are always dense.
- **Not implementing incremental/online PCA.** The batch is available upfront.
- **Not supporting explained variance selection** (e.g., "keep 95% variance"). Fixed target_dim only.

## Implementation Notes

### Re-normalization

After PCA projection, data is NOT unit-norm. Must L2-normalize rows before returning from `transform()` / `fit_transform()`. This preserves cosine geometry for downstream spherical K-means or cosine-metric algorithms.

### Matryoshka Method

When `method="matryoshka"`:
- `fit()` is a no-op (no components to learn)
- `transform()` just slices first `target_dim` columns and L2-normalizes
- `save()` only saves method and target_dim (no components)

### Compatibility

The saved format must be versioned so future changes don't break old files. Header:
```
[magic: 4 bytes "RCPC"]
[version: u32]
[input_dim: u64]
[target_dim: u64]
[method_len: u32]
[method: UTF-8 bytes]
[mean: input_dim × f64]
[components: input_dim × target_dim × f64]
```

## Testing

- `test_fit_transform_shape` — output is (n, target_dim)
- `test_transform_matches_fit_transform` — fit then transform matches fit_transform
- `test_unit_norm_after_transform` — all rows have norm 1.0
- `test_save_load_roundtrip` — save, load, transform produces identical output
- `test_matryoshka_truncates_correctly` — first target_dim columns preserved
- `test_transform_new_data` — transform on held-out data produces (m, target_dim)
- `test_f32_input` — f32 data accepted and produces f64 output
- `test_existing_algorithms_on_reduced` — KMeans, DBSCAN work on reduced data
