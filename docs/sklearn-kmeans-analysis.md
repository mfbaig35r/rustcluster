# scikit-learn K-Means: Internals and Where Rust Can Win

Research conducted 2026-04-18 to understand our competition and identify realistic performance advantages.

## scikit-learn K-Means Implementation

scikit-learn's K-Means is written in **Cython** (not pure C), with **OpenMP** parallelism for the assignment step. It supports two algorithms:

### Lloyd's Algorithm (`algorithm="lloyd"`)
- The classic iterate-assign-update loop
- Inner distance computations use **BLAS** (`gemm`) to compute squared Euclidean distances as `||x||^2 + ||c||^2 - 2*x*c^T`
- This turns the bottleneck into a matrix multiply, which is extremely fast for high-dimensional data because it hits optimized BLAS (MKL/OpenBLAS)
- Default since scikit-learn 1.1

### Elkan's Algorithm (`algorithm="elkan"`)
- Uses triangle inequality to skip distance computations
- Maintains per-point lower bounds for each cluster
- Faster when `k` is large and dimensions are low
- Uses `O(n*k)` extra memory, which kills it for large `k` or large `n`
- BLAS-backed distance computation made Elkan's memory overhead not worth it for most cases

### Parallelism
- OpenMP threads parallelize across chunks of data points in the assignment step
- GIL is released during Cython execution, so multi-core utilization is real during compute
- Python overhead exists in initialization (k-means++), convergence checks, and orchestration between iterations

## Where scikit-learn Is Hard to Beat

### High-dimensional dense data (d > 50)
The BLAS `gemm` trick makes distance computation essentially a matrix multiply. Competing with MKL/OpenBLAS on matmul is extremely difficult — a naive Rust implementation doing point-by-point distance will be **slower**.

### Large n, moderate k
The BLAS approach scales well here. The assignment step is memory-bandwidth-bound, and BLAS handles this optimally.

## Where scikit-learn IS Beatable

### 1. Low-dimensional data (d < 20)
The BLAS gemm trick has overhead (forming full distance matrices) that dominates when `d` is small. A Rust implementation with **SIMD-vectorized direct distance computation** (AVX2/AVX-512) can avoid the matrix intermediary and win.

**This is a huge real-world sweet spot**: customer segmentation, geospatial clustering, low-d embeddings.

### 2. Small to medium n (1K-100K)
Python/Cython startup overhead, memory allocation for intermediate arrays, and k-means++ initialization overhead are proportionally large. Rust's zero-overhead startup and stack allocation win here.

### 3. Memory efficiency with large k
Elkan's `O(n*k)` bounds arrays cause cache thrashing. A Rust implementation can use **Hamerly's algorithm** (one bound per point, `O(n)` memory) that scikit-learn doesn't offer.

### 4. k-means++ initialization
scikit-learn's k-means++ is partially Python. Doing it fully in Rust saves 10-30% of total time on large datasets.

### 5. Mini-batch K-Means
scikit-learn's mini-batch implementation has Python-level batch orchestration overhead. Rust can keep the hot loop tight.

### 6. Streaming / incremental use cases
scikit-learn allocates fresh arrays each call. A Rust library can reuse buffers across calls via a persistent struct.

## Benchmarking Strategy

### Dataset Parameters
| Parameter | Values |
|-----------|--------|
| n_samples | 10K, 100K, 1M, 10M |
| n_features | 2, 8, 32, 128, 512 |
| n_clusters | 8, 64, 256 |

### Methodology
- Wall-clock time excluding import/load
- Median of 5 runs after 1 warmup
- Compare `sklearn.cluster.KMeans` with `n_init=1` (single run) vs Rust
- Same initial centroids or fixed seed for both
- Report both total time and per-iteration time

### What We Need to Prove
Not dominance across all regimes. Credible advantage in our target regime (low-d, medium-n) and competitive elsewhere.

## Rust Optimization Roadmap

### MVP (v1)
1. Lloyd's algorithm with auto-vectorized distance kernel
2. k-means++ initialization fully in Rust
3. `rayon` for parallel assignment step
4. `ndarray::Array2<f64>` row-major layout
5. Build with `lto="fat"`, `codegen-units=1`, `target-cpu=native`

### Future (v2+)
1. **Hamerly's algorithm** for low-to-moderate `k` (one upper + one lower bound per point — cache-friendly)
2. **Explicit SIMD distance** via `std::arch` or `pulp` for low-d cases
3. **SoA (struct-of-arrays) layout** for cache-line efficiency in specific d ranges
4. **BLAS binding** (`faer` or `blas-sys`) for the high-d path, matching sklearn's `gemm` trick
5. Thread-local accumulators for the update step (avoids atomic contention)

## Bottom Line

Target the **low-dimensional (d < 20) and small-to-medium-n regime** where BLAS overhead hurts scikit-learn. This is where a lot of real clustering happens (customer segmentation, geospatial, embeddings). For high-dimensional data, we'll eventually need BLAS ourselves or we'll lose — but that's a v2 concern.
