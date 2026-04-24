# rustcluster

Fast, Rust-backed clustering for Python. Six algorithms, sklearn-compatible API, purpose-built embedding clustering with 11x PCA speedup.

## Highlights

- **6 algorithms**: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, EmbeddingCluster
- **EmbeddingCluster** — purpose-built pipeline for OpenAI/Cohere/Voyage embeddings (L2-normalize → PCA → spherical K-means)
- **EmbeddingReducer** — standalone PCA transformer with save/load (fit once, cluster for free)
- **faer-accelerated PCA** — 11x faster than hand-rolled matmul via SIMD-optimized GEMM
- **3 distance metrics**: euclidean, cosine, manhattan
- **3 evaluation metrics**: silhouette score, Calinski-Harabasz, Davies-Bouldin
- **KD-tree acceleration** for DBSCAN/HDBSCAN neighbor queries (10-200x on low-d data)
- **Native f32/f64** — no silent upcast, doubles cache efficiency with f32
- **Cluster slotting** — snapshot fitted clusters, assign new points 100x faster than refitting
- **Pickle serialization** for all fitted models
- **GIL released** during all compute — plays well with threads and async
- **461 tests** across Rust and Python

## Installation

```bash
pip install rustcluster
```

Or from source (requires [Rust toolchain](https://rustup.rs/) + Python 3.10+):

```bash
pip install maturin
git clone https://github.com/mfbaig35r/rustcluster.git
cd rustcluster
maturin develop --release
```

## Quickstart

### K-Means

```python
from rustcluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
model.fit(X)
model.labels_           # cluster assignments
model.cluster_centers_  # centroids (k x d)
model.inertia_          # sum of squared distances
model.predict(X_new)    # assign new data
```

### Embedding Clustering

Purpose-built pipeline for dense embedding vectors (OpenAI, Cohere, Voyage, etc.):

```python
from rustcluster.experimental import EmbeddingCluster

model = EmbeddingCluster(n_clusters=50, reduction_dim=128)
model.fit(embeddings)          # L2-normalize → PCA → spherical K-means
model.labels_                  # cluster assignments
model.cluster_centers_         # unit-norm centroids in reduced space
model.intra_similarity_        # per-cluster cosine similarity
model.reduced_data_            # access PCA-reduced data
```

### EmbeddingReducer (Fit Once, Cluster Many)

PCA is 99% of the embedding pipeline runtime. Separate reduction from clustering to iterate for free:

```python
from rustcluster.experimental import EmbeddingReducer

# Pay the PCA cost once
reducer = EmbeddingReducer(target_dim=128)
X_reduced = reducer.fit_transform(embeddings)  # 323K × 1536 → 128 in ~56s
reducer.save("pca_128.bin")

# Iterate on clustering for free
reducer = EmbeddingReducer.load("pca_128.bin")
X_reduced = reducer.transform(new_embeddings)

EmbeddingCluster(n_clusters=50, reduction_dim=None).fit(X_reduced)   # ~4s
EmbeddingCluster(n_clusters=100, reduction_dim=None).fit(X_reduced)  # ~8s
EmbeddingCluster(n_clusters=200, reduction_dim=None).fit(X_reduced)  # ~15s
```

Matryoshka models (e.g., text-embedding-3-small) can skip PCA entirely:

```python
reducer = EmbeddingReducer(target_dim=128, method="matryoshka")
X_reduced = reducer.fit_transform(embeddings)  # instant — just truncates + L2-normalizes
```

See the [embedding clustering guide](docs/embedding-clustering-guide.md) for full documentation.

### Cluster Slotting (Incremental Assignment)

Fit once, assign new data forever. Snapshot freezes cluster centroids — new points are assigned without re-clustering:

```python
from rustcluster import KMeans, ClusterSnapshot

# Fit and snapshot
model = KMeans(n_clusters=50).fit(X_train)
snapshot = model.snapshot()
snapshot.save("clusters/")

# Later: load and assign new data (no refit needed)
snapshot = ClusterSnapshot.load("clusters/")
labels = snapshot.assign(X_new)                   # 100x faster than refitting
```

Works with KMeans, MiniBatchKMeans, and EmbeddingCluster. EmbeddingCluster snapshots bake in the full preprocessing pipeline (L2-normalize, PCA, spherical assignment).

**Confidence scoring and rejection:**

```python
result = snapshot.assign_with_scores(X_new, confidence_threshold=0.3)
result.labels_       # -1 for rejected points
result.confidences_  # [0, 1) — higher means more decisive assignment
result.distances_    # distance to nearest centroid
result.rejected_     # boolean mask
```

**Drift detection:**

```python
report = snapshot.drift_report(X_recent)
report.global_mean_distance_  # compare to training baseline
report.relative_drift_        # per-cluster drift
```

**Persistence:** safetensors (centroids) + JSON (metadata). A 50-cluster, 128d snapshot is ~50 KB vs GBs of training data.

Validated on 323K CROSS ruling embeddings: 113x speedup, 99.86% training fidelity, equivalent purity to full refit.

### Mini-Batch K-Means

```python
from rustcluster import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=3, batch_size=256, random_state=42)
model.fit(X_large)      # scales to large datasets
```

### DBSCAN

```python
from rustcluster import DBSCAN

model = DBSCAN(eps=0.5, min_samples=5)
model.fit(X)
model.labels_                # -1 for noise
model.core_sample_indices_   # core point indices
```

### HDBSCAN

```python
from rustcluster import HDBSCAN

model = HDBSCAN(min_cluster_size=5)
model.fit(X)
model.labels_              # -1 for noise
model.probabilities_       # soft membership [0, 1]
model.cluster_persistence_ # per-cluster stability
```

### Agglomerative Clustering

```python
from rustcluster import AgglomerativeClustering

model = AgglomerativeClustering(n_clusters=3, linkage="ward")
model.fit(X)
model.labels_     # cluster assignments
model.children_   # merge history
model.distances_  # distance at each merge
```

## Evaluation Metrics

```python
from rustcluster import silhouette_score, calinski_harabasz_score, davies_bouldin_score

silhouette_score(X, labels)         # [-1, 1], higher is better
calinski_harabasz_score(X, labels)  # higher is better
davies_bouldin_score(X, labels)     # lower is better
```

## Distance Metrics

All algorithms accept a `metric` parameter:

```python
KMeans(n_clusters=5, metric="cosine")
DBSCAN(eps=0.3, metric="manhattan")
HDBSCAN(min_cluster_size=5, metric="euclidean")
```

| Metric | Aliases | KD-tree acceleration | Notes |
|--------|---------|---------------------|-------|
| `"euclidean"` | `"l2"` | Yes | Default for all algorithms |
| `"cosine"` | | No (brute force) | K-means forces Lloyd (Hamerly assumes Euclidean) |
| `"manhattan"` | `"cityblock"`, `"l1"` | Yes | |

Ward linkage requires euclidean metric.

## Performance

### K-Means vs scikit-learn

Single-threaded, `n_init=1`, median of 5 runs:

| n | d | k | Speedup vs sklearn |
|---|---|---|---|
| 1,000 | 8 | 8 | 2.9x |
| 10,000 | 8 | 8 | 2.4x |
| 100,000 | 8 | 32 | 3.2x |
| 100,000 | 32 | 32 | 1.4x |

DBSCAN and HDBSCAN use KD-tree acceleration for `d <= 16` with euclidean or manhattan metrics, reducing neighbor queries from O(n^2) to O(n log n).

### Embedding Clustering

Measured on 323K embeddings (text-embedding-3-small, 1536d → 128d, K=98, Apple Silicon):

| Workflow | Time |
|----------|------|
| Full pipeline (PCA + cluster) | **58s** |
| Subsequent run (cached reduced data) | **7.5s** |
| 5 clustering configs on cached data | **74s** |
| Matryoshka (no PCA needed) | **~5s** |

Full benchmarks: `python benches/benchmark.py`

## Serialization

All models support pickle:

```python
import pickle

model = KMeans(n_clusters=3).fit(X)
data = pickle.dumps(model)
model_restored = pickle.loads(data)  # fitted state preserved
```

EmbeddingReducer uses a compact binary format:

```python
reducer.save("pca_128.bin")                   # 1.5 KB
reducer = EmbeddingReducer.load("pca_128.bin") # instant
```

ClusterSnapshot uses safetensors + JSON for portable, safe persistence:

```python
snapshot = model.snapshot()
snapshot.save("clusters/")                     # safetensors + metadata.json
snapshot = ClusterSnapshot.load("clusters/")   # zero-copy load
```

## Development

```bash
maturin develop --release              # build
cargo test --no-default-features --lib # Rust tests (197)
pytest tests/ -v                       # Python tests (264)
python benches/benchmark.py            # benchmark vs sklearn
cargo fmt -- --check                   # formatting
cargo clippy --no-default-features --lib -- -D warnings  # linting
```

## Architecture

Three-layer kernel design separating concerns:

1. **PyO3 boundary** (`src/lib.rs`) — input validation, GIL release, dtype dispatch
2. **Algorithm logic** (`src/kmeans.rs`, etc.) — iteration, convergence, ndarray types
3. **Hot kernel** (`src/utils.rs`, `src/distance.rs`) — raw `&[F]` slices for auto-vectorization

The embedding pipeline adds:

4. **Embedding module** (`src/embedding/`) — spherical K-means, PCA (faer-backed), vMF refinement, EmbeddingReducer

See [docs/architecture-decisions.md](docs/architecture-decisions.md) for details and [docs/lessons-building-rustcluster.md](docs/lessons-building-rustcluster.md) for the full build story.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add algorithms, distance metrics, and tests.

## License

MIT
