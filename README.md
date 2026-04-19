# rustcluster

Fast, Rust-backed clustering for Python. Five algorithms, sklearn-compatible API, 1.3-6.4x faster K-means.

## Highlights

- **5 algorithms**: KMeans (Lloyd + Hamerly), MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
- **3 distance metrics**: euclidean, cosine, manhattan
- **3 evaluation metrics**: silhouette score, Calinski-Harabasz, Davies-Bouldin
- **KD-tree acceleration** for DBSCAN/HDBSCAN neighbor queries (10-200x on low-d data)
- **Native f32/f64** — no silent upcast, doubles cache efficiency with f32
- **Pickle serialization** for all fitted models
- **GIL released** during all compute — plays well with threads and async
- **320 tests** across Rust and Python

## Installation

```bash
pip install rustcluster
```

Or from source (requires [Rust toolchain](https://rustup.rs/) + Python 3.9+):

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

K-means benchmark vs scikit-learn (single-threaded, `n_init=1`, median of 5 runs):

| n | d | k | Speedup vs sklearn |
|---|---|---|---|
| 1,000 | 8 | 8 | 2.9x |
| 10,000 | 8 | 8 | 2.4x |
| 100,000 | 8 | 32 | 3.2x |
| 100,000 | 32 | 32 | 1.4x |

DBSCAN and HDBSCAN use KD-tree acceleration for `d <= 16` with euclidean or manhattan metrics, reducing neighbor queries from O(n^2) to O(n log n).

Full benchmark: `python benches/benchmark.py`

## Serialization

All models support pickle:

```python
import pickle

model = KMeans(n_clusters=3).fit(X)
data = pickle.dumps(model)
model_restored = pickle.loads(data)  # fitted state preserved
```

## Development

```bash
maturin develop --release              # build
cargo test --no-default-features --lib # Rust tests (128)
pytest tests/ -v                       # Python tests (192)
python benches/benchmark.py            # benchmark vs sklearn
cargo fmt -- --check                   # formatting
cargo clippy --no-default-features --lib -- -D warnings  # linting
```

## Architecture

Three-layer kernel design separating concerns:

1. **PyO3 boundary** (`src/lib.rs`) — input validation, GIL release, dtype dispatch
2. **Algorithm logic** (`src/kmeans.rs`, etc.) — iteration, convergence, ndarray types
3. **Hot kernel** (`src/utils.rs`, `src/distance.rs`) — raw `&[F]` slices for auto-vectorization

See [docs/architecture-decisions.md](docs/architecture-decisions.md) for details and [docs/lessons-building-rustcluster.md](docs/lessons-building-rustcluster.md) for the full build story.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add algorithms, distance metrics, and tests.

## License

MIT
