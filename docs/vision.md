# rustcluster — Vision

## What This Is

A Rust-backed Python machine learning library, starting with clustering algorithms. The goal is to combine Rust's performance and memory efficiency with Python's ecosystem and usability — delivering a library that feels native to Python users but runs at compiled-language speed.

This is not a Rust ML library with Python bindings bolted on. This is a Python library with Rust under the hood.

## Why This Exists

The Python ML ecosystem has a gap:

- **scikit-learn** is mature but constrained by Cython/C, GIL limitations, and legacy architecture decisions. It's hard to beat on high-dimensional BLAS workloads, but leaves performance on the table for low-to-medium dimensional data, memory efficiency, and startup overhead.
- **linfa** is the Rust ML ecosystem's sklearn equivalent, but it's pure Rust with no Python interop — inaccessible to the vast majority of ML practitioners.
- **polars** proved that Rust-backed Python libraries can be fast, ergonomic, and community-driven — but it's dataframes, not ML.

There is no community-driven, Rust-backed Python ML toolkit with a clean architecture and contributor story. That's the gap.

## Who This Is For

### Users
Python developers and data scientists who want:
- A familiar sklearn-like API
- Better performance on common workloads (low-to-medium dimensional clustering, medium-sized datasets)
- Lower memory footprint
- NumPy interoperability
- No need to know or care about Rust

### Contributors
Rust developers and ML practitioners who want:
- A clean, modular codebase to contribute algorithms to
- Clear trait contracts and isolated modules
- A real-world Rust+Python project to learn from
- Shared infrastructure (distance functions, validation, benchmarking) so they can focus on algorithm logic

## Architecture Philosophy

### Core Principles

1. **Algorithm as module** — each algorithm is a self-contained module that depends on shared core infrastructure but not on other algorithms. Adding DBSCAN doesn't require understanding K-means internals.

2. **Shared core** — distance functions, input validation, array utilities, error types, and Python binding helpers live in a common core. Every algorithm builds on this foundation.

3. **Thin Python layer** — the Python API is a thin wrapper around Rust. Core algorithm logic never lives in Python. The Python side handles ergonomics (docstrings, repr, sklearn-familiar naming) while Rust handles computation.

4. **Generalize from concrete implementations** — don't design abstract trait hierarchies before having multiple algorithms to generalize from. Extract shared patterns only when you have two or more concrete implementations that need them.

5. **Performance by default** — zero-copy NumPy input, GIL release during compute, LLVM auto-vectorization, rayon parallelism, pre-allocated buffers. Performance isn't a feature; it's the architecture.

### Module Structure (Target)

```
rustcluster/
  core/              — shared traits, distance functions, validation, array utils
    distance.rs      — Euclidean, Manhattan, Cosine (trait-based)
    validation.rs    — input shape/type checks
    error.rs         — consistent error hierarchy
    utils.rs         — common numerical operations
  
  algorithms/
    kmeans/          — K-means (Lloyd's, Hamerly's, mini-batch)
    dbscan/          — DBSCAN
    hdbscan/         — HDBSCAN
    agglomerative/   — agglomerative clustering
    ...
  
  metrics/           — silhouette, calinski-harabasz, davies-bouldin
  
  python/            — PyO3 binding layer
    rustcluster/
      __init__.py
      _kmeans.py     — thin Python wrapper
      _dbscan.py     — thin Python wrapper
      ...
```

Each algorithm module:
- Implements shared traits (Fit, Predict, or algorithm-specific variants)
- Has its own Rust unit tests
- Has corresponding Python integration tests
- Has benchmark scripts against sklearn equivalents

### Distance Traits (Future)

```rust
pub trait Distance: Send + Sync {
    fn compute(&self, a: &[f64], b: &[f64]) -> f64;
}
```

Algorithms accept a generic distance, defaulting to Euclidean. This enables cosine K-means, Manhattan DBSCAN, etc. without duplicating algorithm code.

### What Makes This Contributable

- **Clear contracts** — implement the right traits with the right signatures and your algorithm integrates
- **Isolated modules** — contributors can add an algorithm without touching existing ones
- **Shared test harness** — common toy datasets and validation patterns
- **Shared benchmarks** — standardized suite so every algorithm gets compared fairly to its sklearn equivalent
- **Good docs** — contributor guide, architecture overview, "how to add an algorithm" walkthrough

## Roadmap

### v1 — MVP (Current)
- K-means (Lloyd's algorithm)
- kmeans++ initialization
- rayon parallelism for assignment step
- sklearn-compatible API (fit, predict, fit_predict)
- NumPy interop via PyO3 + rust-numpy
- pytest suite + benchmark script
- pip-installable via maturin

### v2 — Second Algorithm + Trait Extraction
- DBSCAN or Mini-batch K-means (second concrete algorithm)
- Extract shared traits from two implementations (Distance, Fit, Predict)
- Hamerly's algorithm variant for K-means
- Explicit SIMD for low-d distance kernels
- Contributor guide

### v3 — Ecosystem Growth
- HDBSCAN
- Agglomerative clustering
- Clustering metrics (silhouette, calinski-harabasz)
- Custom distance metric support
- float32 support
- BLAS integration for high-d performance
- CI/CD with multi-platform wheel builds (x86_64, aarch64, Linux, macOS, Windows)
- PyPI distribution

### v4+ — Community-Driven Expansion
- Sparse matrix support
- Cosine K-means
- Model serialization
- Additional algorithm families (beyond clustering)
- Community contributions
- Stabilized trait API

## Competitive Position

| Regime | vs scikit-learn |
|--------|----------------|
| Low-d (d < 20), medium n | 2-5x faster, lower memory |
| Low-d, large n | 1.5-3x faster, lower memory |
| Medium-d (20-50) | Competitive performance, lower memory |
| High-d (d > 100) | Needs BLAS integration (v3) to compete on speed; still wins on memory |

We don't need to beat sklearn everywhere. We need to be **genuinely faster for the workloads most people actually run** — and most real-world clustering is low-to-medium dimensional.

## Inspiration

- **polars** — proved Rust-backed Python libraries can be fast, ergonomic, and community-loved
- **linfa** — the right modular architecture for Rust ML, but missing the Python bridge
- **pydantic-core** — clean PyO3 integration patterns, thin Python layer over Rust
- **tokenizers** — HuggingFace showed Rust+Python can ship to millions of users

## Non-Goals

- Not trying to replace sklearn entirely
- Not building a deep learning framework
- Not targeting GPU compute (at least not in the foreseeable future)
- Not building dataframe-native APIs (that's polars' job)
- Not chasing full sklearn API parity for parity's sake
