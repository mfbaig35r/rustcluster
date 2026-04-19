# rustcluster

A Rust-backed Python clustering library. Fast, memory-efficient K-means clustering with an sklearn-inspired API.

## Installation

Requires Rust toolchain and Python 3.9+.

```bash
# Install maturin if you don't have it
pip install maturin

# Build and install in development mode
maturin develop --release

# Or install with pip
pip install -e .
```

## Quickstart

```python
import numpy as np
from rustcluster import KMeans

X = np.array([
    [1.0, 2.0],
    [1.1, 2.1],
    [8.0, 8.0],
    [8.2, 8.1],
], dtype=np.float64)

model = KMeans(n_clusters=2, random_state=42)
model.fit(X)

print(model.labels_)           # cluster assignments
print(model.cluster_centers_)  # centroid coordinates
print(model.inertia_)          # sum of squared distances
print(model.n_iter_)           # iterations to converge

# Predict on new data
preds = model.predict(X)
```

## API

### `KMeans(n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10)`

**Parameters:**
- `n_clusters` — number of clusters
- `max_iter` — maximum iterations per run (default: 300)
- `tol` — convergence tolerance on centroid shift (default: 1e-4)
- `random_state` — seed for reproducibility (default: 0)
- `n_init` — number of initializations, best inertia wins (default: 10)

**Methods:**
- `fit(X)` — fit the model to data
- `predict(X)` — assign new data to nearest fitted centroids
- `fit_predict(X)` — fit and return labels

**Fitted attributes:**
- `labels_` — cluster labels for training data
- `cluster_centers_` — centroid coordinates (k x d)
- `inertia_` — sum of squared distances to nearest centroid
- `n_iter_` — number of iterations in the best run

## Development

```bash
# Build in development mode
maturin develop

# Build release mode (optimized)
maturin develop --release

# Run Rust tests
cargo test

# Run Python tests
pytest tests/ -v

# Run benchmarks (requires scikit-learn)
python benches/benchmark.py
```

## Architecture

- `src/` — Rust core: K-means algorithm, distance kernels, PyO3 bindings
- `python/rustcluster/` — thin Python wrapper with input normalization
- `tests/` — pytest integration tests
- `benches/` — performance benchmarks vs scikit-learn
- `docs/` — research and architecture decisions

## Status

Early MVP focused on K-means. See `docs/vision.md` for the roadmap.
