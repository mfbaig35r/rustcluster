# Contributing to rustcluster

Thanks for your interest in contributing. This guide covers how to set up the project, understand the architecture, and add new algorithms or distance metrics.

## Development Setup

**Prerequisites:** [Rust stable toolchain](https://rustup.rs/), Python 3.9+

```bash
git clone https://github.com/mfbaig35r/rustcluster.git
cd rustcluster
python -m venv .venv && source .venv/bin/activate
pip install maturin numpy pytest scikit-learn
maturin develop --release
```

**Verify:**

```bash
cargo test --no-default-features --lib   # 128 Rust tests
pytest tests/ -v                          # 192 Python tests
```

## Architecture

rustcluster uses a **three-layer kernel** pattern. Every algorithm follows the same data flow:

```
Layer 1: PyO3 boundary    src/lib.rs          — validates input, releases GIL, dispatches dtype (f32/f64)
Layer 2: Algorithm logic   src/<algo>.rs       — iteration, convergence, ndarray types
Layer 3: Hot kernel        src/utils.rs        — raw &[F] slices, auto-vectorized distance computation
                           src/distance.rs
```

### File Map

| File | Role |
|------|------|
| `src/distance.rs` | `Scalar` trait (f32/f64), `Distance<F>` trait, metric impls (`SquaredEuclidean`, `CosineDistance`, `ManhattanDistance`), `Metric` enum |
| `src/utils.rs` | Distance kernels (`squared_euclidean_generic`, `assign_nearest_with`), input validation |
| `src/kdtree.rs` | KD-tree spatial index, `BBoxDistance` trait for pruning |
| `src/error.rs` | `ClusterError` enum (shared across all algorithms), PyO3 exception conversion |
| `src/kmeans.rs` | K-means: Lloyd + Hamerly dispatch, kmeans++ init, shared helpers |
| `src/hamerly.rs` | Hamerly's accelerated K-means |
| `src/minibatch_kmeans.rs` | Mini-batch K-means with streaming centroid updates |
| `src/dbscan.rs` | DBSCAN with KD-tree accelerated neighbor scan |
| `src/hdbscan.rs` | HDBSCAN: core distances, MST, hierarchy, stability extraction |
| `src/agglomerative.rs` | Agglomerative clustering with Lance-Williams updates |
| `src/metrics.rs` | Silhouette, Calinski-Harabasz, Davies-Bouldin scores |
| `src/lib.rs` | PyO3 module: `#[pyclass]` definitions, dtype dispatch, pickle support |
| `python/rustcluster/__init__.py` | Thin Python wrappers, input normalization, pickle delegation |
| `tests/test_*.py` | Python integration tests (one file per algorithm + metrics + serialization) |

### Key Traits

```rust
// In src/distance.rs

pub trait Scalar: Float + Send + Sync + Sum + Debug + 'static {
    fn to_f64_lossy(self) -> f64;
    fn from_f64_lossy(v: f64) -> Self;
}

pub trait Distance<F: Scalar>: Send + Sync + Clone + Copy + 'static {
    fn distance(a: &[F], b: &[F]) -> F;
    fn to_metric(raw: f64) -> f64 { raw }  // override for SquaredEuclidean (.sqrt())
}

pub enum Metric { Euclidean, Cosine, Manhattan }
```

Every algorithm is generic over `<F: Scalar, D: Distance<F>>`. At the Python boundary, the `Metric` enum provides runtime dispatch to the monomorphized generic.

## How to Add a New Algorithm

Use any existing algorithm as a template. The pattern is identical across all five.

### Step 1: Create `src/new_algo.rs`

```rust
use crate::distance::{CosineDistance, Distance, ManhattanDistance, Metric, Scalar, SquaredEuclidean};
use crate::error::ClusterError;
use crate::utils::validate_data_generic;

pub struct NewAlgoState<F: Scalar> {
    pub labels: Vec<i64>,
    // ... other fitted attributes
}

// Public entry points — one per dtype, dispatching on Metric
pub fn run_new_algo_with_metric(
    data: &ArrayView2<f64>, /* params */
    metric: Metric,
) -> Result<NewAlgoState<f64>, ClusterError> {
    match metric {
        Metric::Euclidean => run_generic::<f64, SquaredEuclidean>(data, /* params */),
        Metric::Cosine => run_generic::<f64, CosineDistance>(data, /* params */),
        Metric::Manhattan => run_generic::<f64, ManhattanDistance>(data, /* params */),
    }
}

// Same for f32
pub fn run_new_algo_with_metric_f32(...) -> Result<NewAlgoState<f32>, ClusterError> { ... }

// Generic implementation
fn run_generic<F: Scalar, D: Distance<F>>(...) -> Result<NewAlgoState<F>, ClusterError> {
    validate_data_generic(data)?;
    // ... your algorithm using D::distance(a, b) for distances
}
```

### Step 2: Register the module

In `src/lib.rs`, add at the top:

```rust
pub mod new_algo;
```

### Step 3: Add PyO3 bindings

In `src/lib.rs` inside `mod python_bindings`, add a `#[pyclass]` following the existing pattern:

```rust
#[pyclass]
struct NewAlgo {
    // parameters + fitted: Option<NewAlgoFitted>
}

#[pymethods]
impl NewAlgo {
    #[new]
    fn new(/* params */) -> PyResult<Self> { ... }
    
    fn fit(&mut self, py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try f64 first, then f32
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f64>>() { ... }
        if let Ok(arr) = x.extract::<PyReadonlyArray2<'_, f32>>() { ... }
    }
    
    fn fit_predict(...) { ... }
    #[getter] fn labels_(...) { ... }
    fn __repr__(...) { ... }
    fn __getstate__(...) { ... }  // pickle
    fn __setstate__(...) { ... }
    fn __getnewargs__(...) { ... }
}
```

Register in the module function:

```rust
m.add_class::<NewAlgo>()?;
```

### Step 4: Add Python wrapper

In `python/rustcluster/__init__.py`:

```python
from rustcluster._rustcluster import NewAlgo as _RustNewAlgo

class NewAlgo:
    def __init__(self, **params):
        self._model = _RustNewAlgo(**params)
    
    def fit(self, X):
        X = _prepare_array(X)
        self._model.fit(X)
        return self
    
    # ... properties, __repr__, __getstate__, __setstate__
```

Add to `__all__`.

### Step 5: Add tests

Create `tests/test_new_algo.py` covering:
- Basic fit (returns self, labels length, attribute shapes)
- Correctness on well-separated data
- Edge cases (single cluster, k=n, identical points)
- Parameter validation (invalid values raise ValueError)
- Properties before fit raise RuntimeError
- f32 support (dtype preserved)
- Pickle round-trip
- Different metrics work
- Reproducibility (deterministic results)

Add Rust unit tests in `src/new_algo.rs` under `#[cfg(test)] mod tests`.

## How to Add a New Distance Metric

### Step 1: Implement the kernel

In `src/distance.rs`:

```rust
#[derive(Debug, Clone, Copy)]
pub struct NewMetric;

impl<F: Scalar> Distance<F> for NewMetric {
    #[inline(always)]
    fn distance(a: &[F], b: &[F]) -> F {
        // Simple counted loop for auto-vectorization
        // ...
    }
    
    // Override if the raw distance needs conversion (like SquaredEuclidean -> sqrt)
    // fn to_metric(raw: f64) -> f64 { raw }
}
```

### Step 2: Add to the Metric enum

```rust
pub enum Metric { Euclidean, Cosine, Manhattan, NewMetric }

impl Metric {
    pub fn from_str(s: &str) -> Result<Self, ClusterError> {
        match s.to_lowercase().as_str() {
            // ... existing ...
            "new_metric" => Ok(Metric::NewMetric),
            _ => Err(...)
        }
    }
}
```

### Step 3: Add match arms everywhere

Every `match metric { ... }` block in every algorithm's `run_*_with_metric` function and in `lib.rs` predict methods needs a new arm. Search for `Metric::Manhattan` to find all locations.

### Step 4: Optional — KD-tree support

If your metric supports axis-aligned bounding-box pruning, implement `BBoxDistance` in `src/kdtree.rs`:

```rust
impl BBoxDistance for NewMetric {
    fn min_dist_to_bbox(point: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64 { ... }
}
```

And update `KdTree::should_use` to include the new metric.

### Step 5: Add tests

- Unit tests in `src/distance.rs` for the kernel
- Update each Python test file's `test_invalid_metric` to use a different invalid metric name
- Add metric-specific tests to each algorithm's test file

## Code Style

- **Rust**: `cargo fmt` for formatting, `cargo clippy --no-default-features --lib -- -D warnings` for linting
- **No `unsafe`** unless absolutely necessary with justification
- **No `panic = "abort"`** — let PyO3 translate panics to Python exceptions
- **Three-layer pattern**: hot kernels on raw `&[F]` slices, algorithm logic with ndarray, PyO3 at the boundary
- **Python wrappers are thin** — all logic in Rust, Python handles input normalization and pickle delegation

## PR Process

1. Fork and create a feature branch from `main`
2. One algorithm or one metric per PR
3. Ensure all tests pass:
   ```bash
   cargo test --no-default-features --lib
   maturin develop --release && pytest tests/ -v
   cargo fmt -- --check
   cargo clippy --no-default-features --lib -- -D warnings
   ```
4. CI runs automatically on PRs (Linux/macOS/Windows, Python 3.9 + 3.12)
5. Include both Rust unit tests and Python integration tests

## Performance Guidelines

- Benchmark before and after changes to hot paths
- Use `RUSTFLAGS="-C target-cpu=native" cargo bench` for local kernel benchmarks
- Never bake `target-cpu=native` into distributed code
- Keep distance kernels as simple counted loops — let LLVM auto-vectorize
- Batch all work into a single Rust call — never cross the PyO3 boundary per-point
