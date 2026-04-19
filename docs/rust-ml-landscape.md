# Rust ML/Clustering Landscape

Research conducted 2026-04-18 to inform rustcluster architecture decisions.

## Major Rust ML Libraries

### linfa
- The scikit-learn of Rust (`linfa` crate, github.com/rust-ml/linfa)
- Modular design with sub-crates like `linfa-clustering` for K-means, DBSCAN, Gaussian Mixture, OPTICS
- K-means lives in `linfa-clustering` — implements both Lloyd's algorithm and mini-batch variant
- Built on `ndarray` for all matrix operations
- Uses `Fit`/`Predict` trait pattern (similar to sklearn's fit/predict API)
- Initialization: K-means++ and random
- Parallelism via `rayon` feature flag

### smartcore
- Broader ML library (SVM, trees, linear models, clustering)
- K-means uses flat `Vec<f64>` internally with a custom `DenseMatrix` abstraction rather than `ndarray`
- Supports K-means++ init
- Less modular than linfa; monolithic design
- Less actively maintained as of 2025

### rusty-machine
- Legacy; largely abandoned
- Had basic K-means but no SIMD or parallelism

### `kmeans` crate (crates.io/crates/kmeans)
- Purpose-built, performance-oriented
- Uses flat `Vec<f32/f64>` in row-major order for cache-line-friendly sequential point access
- Parallel assignment via `rayon`'s `par_chunks`
- K-means++ and K-means|| (scalable parallel init)
- Convergence via centroid delta norm below epsilon

## K-means Implementation Patterns Across Libraries

### Memory Layout
- Row-major universally preferred
- Each row is a data point, so iterating over points hits contiguous memory
- Distance computation (the hot inner loop) streams through L1 cache efficiently

### linfa-clustering K-means Internals
- `ndarray::Array2<f64>` in row-major (C order)
- Assignment step parallelized with rayon (`par_axis_iter` over rows)
- Convergence: checks if no point changed assignment OR centroid shift < tolerance
- Supports incremental/mini-batch via `IncrementalFit` trait

### SIMD
- Most rely on LLVM auto-vectorization rather than explicit intrinsics
- Tight iterator loops over `&[f64]` slices with `.zip()` and `.map().sum()` auto-vectorize well with `-C target-cpu=native`
- The `pulp` crate offers portable explicit SIMD if needed

### Parallelism
- `rayon` is universal across the ecosystem
- Assignment step (point-to-centroid distances) is embarrassingly parallel — gets the biggest speedup
- Update step (recompute centroids) is typically fast enough single-threaded or uses `rayon`'s `reduce`

### Allocation
- Pre-allocate centroid and assignment buffers outside the main loop; reuse across iterations
- Avoid `collect()` in hot paths; prefer in-place mutation

## Patterns That Make Rust Numerical Code Fast

1. **Iterator chains that auto-vectorize**: `a.iter().zip(b).map(|(x,y)| (x-y).powi(2)).sum::<f64>()` — LLVM unrolls and vectorizes into SIMD instructions with no manual intrinsics
2. **Slice-based APIs over iterators**: Passing `&[f64]` ensures contiguous memory; compiler knows alignment and can emit vector loads
3. **`#[inline]` on small distance functions**: Prevents function-call overhead in the inner loop, enabling cross-function vectorization
4. **Avoiding bounds checks**: Using `.get_unchecked()` in critical loops (unsafe) or structuring iteration so the compiler elides checks (e.g., `for i in 0..len` after asserting lengths match)
5. **`-C target-cpu=native`** in build config unlocks AVX2/AVX-512

## Recommended Crate Stack

| Crate | Purpose |
|-------|---------|
| `ndarray` | N-dimensional arrays, row/col-major, slicing, BLAS integration |
| `rayon` | Data parallelism (par_iter, par_chunks) |
| `pyo3` | Rust <-> Python bindings |
| `numpy` (`rust-numpy`) | Zero-copy ndarray <-> NumPy conversion |
| `rand` + `rand_distr` | RNG for initialization (WeightedIndex for K-means++) |
| `num-traits` | Generic numeric traits (Float, Zero) |
| `pulp` | Portable explicit SIMD (only if auto-vectorization isn't enough) |
| `openblas-src` or `blis-src` | BLAS backend for ndarray (high-d path, future) |

## Architecture Recommendation

Follow linfa's pattern:
- Use `ndarray::Array2` as the core data type
- Expose via `rust-numpy` for zero-copy Python interop through `pyo3`
- Parallelize assignment with `rayon`
- Rely on LLVM auto-vectorization for the distance kernel (only drop to `pulp` if profiling shows it's needed)
