# Architecture Decisions for rustcluster MVP

Decisions made 2026-04-18 based on research into the Rust ML ecosystem, PyO3 performance patterns, scikit-learn internals, and deep research into algorithm tradeoffs and kernel design.

## Summary Table

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Core data type | `ndarray::Array2<f64>` (API layer), `&[f64]` slices (hot kernel) | Zero-copy from NumPy at boundary; raw slices in inner loop for auto-vectorization |
| Algorithm | Tuned Lloyd (v1), Hamerly (v1.1), Yinyang (future) | Lloyd is correct baseline; Hamerly is the right first accelerator for d<20 — NOT Elkan |
| Initialization | kmeans++ (greedy, matching sklearn's policy) | Big win over random, fully in Rust, fair benchmark comparison |
| Parallelism | `rayon` for assignment step | Easy, embarrassingly parallel, big speedup |
| Distance kernel | Auto-vectorized `&[f64]` loop with `#[inline(always)]` | LLVM auto-vectorization is close to explicit AVX2; explicit SIMD is a v2 optimization |
| GIL | Release during all compute via `py.allow_threads()` | Standard practice (polars, tokenizers all do this) |
| Build optimization | `lto="fat"`, `codegen-units=1`, `opt-level=3` | Free 10-20% performance from cross-crate inlining |
| Panic handling | No `panic = "abort"` — let PyO3 translate panics | `abort` kills the Python process; PyO3 converts panics to Python exceptions |
| Python interop | `PyReadonlyArray2` in, strict C-contiguous check, `PyArray::from_owned_array` out | Zero-copy input on fast path; reject strided/Fortran arrays |
| Dtype policy | Reject non-f64 at Rust boundary; Python convenience wrapper normalizes | No silent conversion overhead in the hot path |

## Target Performance Regime

Our realistic competitive advantage is:
- **Low dimensions (d < 20)** — sklearn's BLAS gemm trick has overhead that doesn't pay off here. At d=8, one point fits a single 64-byte cache line. Direct row-wise kernels on contiguous `&[f64]` are faster than forming intermediate distance matrices.
- **Small-to-medium n (1K-100K)** — less startup/allocation overhead than Cython. No Python-level parameter validation chains, no intermediate array allocations.
- **k-means++ init** — fully in Rust vs sklearn's partially-Python implementation. 10-30% of total runtime on large datasets.
- **Memory efficiency** — no `O(n*k)` distance matrices (Lloyd), or `O(n)` bounds (Hamerly) vs sklearn's Elkan at `O(n*k)`.

This covers a lot of real-world clustering: customer segmentation, geospatial, low-d embeddings.

**Where we will NOT win (without BLAS):** high-dimensional data (d > 100) where sklearn's gemm trick delegates to MKL/OpenBLAS. That's a v3 concern.

## Algorithm Strategy

The deep research refined our algorithm roadmap significantly. Key finding: **Elkan is NOT the right accelerator for our target regime.**

### Why Not Elkan
- Stores `k` lower bounds per point → `O(n*k)` memory
- At n=100K, k=100: ~80MB extra memory just for bounds
- sklearn's own maintainers note that GEMM-boosted Lloyd now beats Elkan in many regimes
- Memory pressure causes cache thrashing in exactly the regime we're targeting

### Recommended Engine Dispatch (Future)

```rust
enum ExactEngine {
    Lloyd,     // very small k, tiny d — minimal overhead
    Hamerly,   // d <= 32, k <= 256 — one bound per point, O(n) memory
    Yinyang,   // larger k, moderate d — grouped bounds, O(n*t) memory
}

fn choose_engine(n: usize, d: usize, k: usize) -> ExactEngine {
    if d <= 16 && k <= 32 {
        ExactEngine::Lloyd
    } else if d <= 32 && k <= 256 {
        ExactEngine::Hamerly
    } else {
        ExactEngine::Yinyang
    }
}
```

These thresholds are a starting heuristic — we benchmark the dispatch rule itself and treat it as a model-selection problem.

### MVP Scope
- v1 ships **tuned Lloyd** as the correctness baseline
- v1.1 adds **Hamerly** as the first exact accelerator (single lower bound per point, `O(n)` memory, fastest for d < 50 per the Hamerly paper)
- Yinyang is v2+

## Kernel Architecture — Three Layers

The deep research strongly recommends separating the kernel from the API abstraction. Don't write the hottest loop against `ArrayView2` — lower the data into raw slices.

### Layer 1: Python Boundary (PyO3)
```rust
fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
    if !x.is_c_contiguous() {
        return Err(PyValueError::new_err(
            "rustcluster requires a C-contiguous float64 array"
        ));
    }
    let view = x.as_array(); // zero-copy borrow
    py.allow_threads(|| {
        self.inner_fit(&view);
    });
    Ok(())
}
```

### Layer 2: Algorithm Logic (ndarray)
- Validates shapes, manages iteration, tracks convergence
- Owns centroid and assignment buffers (pre-allocated, reused across iterations)
- Calls into Layer 3 for the hot path

### Layer 3: Hot Kernel (raw slices)
```rust
#[inline(always)]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}
```

This structure lets LLVM auto-vectorize the inner loop (contiguous memory, counted loop, no abstraction barriers) while keeping the algorithm logic readable at Layer 2.

### Why This Matters
- `linfa-clustering` uses `Zip`-based ndarray iteration — elegant but opaque to the vectorizer
- The `kmeans` crate uses raw `&[f64]` + hand-written SIMD — fast but `unsafe`-heavy
- We take the middle path: raw slices for auto-vectorization, no `unsafe` needed

## PyO3 Boundary — Strict Fast Path

### Input Policy
- **Require C-contiguous `float64`** at the Rust boundary
- Reject Fortran-order, strided, and non-f64 arrays with a clear error
- Python convenience wrapper handles normalization: `np.ascontiguousarray(x, dtype=np.float64)`
- `as_array()` is zero-copy but works on any stride (slow); we want `as_slice()` semantics (fast)

### Output Policy
- Labels: `PyArray1::from_vec(py, labels)` — one copy, ownership transfer
- Centroids: `PyArray2::from_owned_array(py, centroids)` — one memcpy, unavoidable (separate allocators)
- Both copies happen once after compute completes — negligible vs runtime

### Call Overhead
- PyO3 function call: ~68ns (vs ~43ns pure Python no-op, ~40ns Cython cpdef)
- Confirms: batch all work into a single Rust call. Never cross the boundary per-point or per-iteration.

## Crate Dependencies

```
pyo3          — Python bindings
numpy         — NumPy <-> ndarray zero-copy (rust-numpy crate)
ndarray       — Core array type (API/algorithm layer)
rayon         — Parallel assignment step
rand          — RNG for initialization
rand_distr    — WeightedIndex for kmeans++
thiserror     — Error types
```

Not including for MVP:
- `pulp` — explicit SIMD (v2, behind runtime dispatch)
- `faer` / `blas-sys` — BLAS for high-d gemm path (v3)
- `nalgebra` — wrong default layout (Fortran-order), avoid in hot path

## Build Configuration

### Release Profile (Cargo.toml)
```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = "debuginfo"
debug = false
```

**Do NOT use `panic = "abort"`** — it kills the Python process instead of letting PyO3 translate panics to Python exceptions.

### Local Benchmarking Only
```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

**Do NOT bake `target-cpu=native` into distributed wheels** — they need portable codegen. Explicit SIMD (v2) should use runtime ISA dispatch instead.

### Apple Silicon
- No AVX2/AVX-512 — ARM uses NEON (128-bit)
- LLVM auto-vectorizes to NEON with the same iterator patterns
- Separate `aarch64-apple-darwin` wheels required
- Rosetta 2 runs x86 wheels but defeats the performance purpose

## Benchmarking Strategy

### Three Layers (from deep research)

1. **Kernel microbenchmark** (Criterion) — assignment-loop throughput with fixed centroids, no Python boundary
2. **FFI microbenchmark** — cost of PyO3 call + array borrow/validation
3. **End-to-end Python benchmark** (pytest-benchmark) — `fit()` with identical init, thread settings, dtype/layout

### Methodology
- **Single-threaded first** — sklearn has known OpenMP/BLAS threading regressions. Isolate kernel performance before measuring threading.
- **Fixed init centers** when comparing kernels (not initialization quality)
- **`n_init=1`** unless benchmarking initialization specifically
- **Data generated outside measured region**
- **Sweep separately**: d, n, k (don't conflate dimensions)

### Sweep Matrix
| Parameter | Values |
|-----------|--------|
| d (features) | 2, 8, 16, 32, 64, 128, 256, 512 |
| n (samples) | 1K, 10K, 100K, 1M |
| k (clusters) | 8, 32, 128, 512 |
| Layout | C-order f64, Fortran f64 (rejected), f32 (rejected) |
| Threads | 1, 2, 4, 8 (separate study) |

### Reference Point
linfa achieved ~1.3x over sklearn on 1M points. "Rust" alone doesn't give 10x — **specialization, copy avoidance, kernel design, and threading discipline** are where the real wins come from.

## What We're NOT Doing in v1

- No BLAS integration (high-d optimization, v3)
- No Elkan's algorithm (wrong tradeoff for our regime — ever)
- No explicit SIMD intrinsics (auto-vectorization first, explicit SIMD v2 behind runtime dispatch)
- No SoA memory layout experiments
- No sparse matrix support
- No GPU support
- No `panic = "abort"`
- No `target-cpu=native` in distributed wheels

## Future Expansion Path

1. **v1.1**: Hamerly's algorithm (first exact accelerator, `O(n)` memory, d < 50 sweet spot)
2. **v2**: Explicit SIMD with runtime ISA dispatch, Yinyang for larger k, mini-batch K-means, engine auto-selection
3. **v3**: BLAS/faer integration for high-d gemm path, DBSCAN, additional distance metrics, f32 native support
4. **v4**: Sparse support, cosine K-means, model serialization, community contributor framework
