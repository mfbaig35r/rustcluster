# Experiment 8: faer PCA Backend Benchmark

**Date:** 2026-04-20
**Question:** How much does faer speed up PCA on real data?

## Setup
- **Data:** 323,308 CROSS ruling embeddings (text-embedding-3-small, 1536d, f32)
- **Config:** PCA → 128d, K=98, n_init=1, max_iter=50
- **Machine:** Apple Silicon (arm64)
- **Change:** Replaced hand-rolled rayon matmul + power-iteration eigendecomp with faer GEMM + thin SVD in `src/embedding/reduction.rs`

## Results

| Metric | Pre-faer (exp6/7) | faer | Speedup |
|--------|-------------------|------|---------|
| PCA fit_transform | 615s | **56s** | **11.0x** |
| Full pipeline (PCA + 50 iters) | 618s | **58s** | **10.6x** |
| Transform only (projection) | 5.5s | **3.0s** | 1.8x |
| K-means (50 iterations) | ~4s | 3.0s | ~1.3x |

## Breakdown

| Stage | Pre-faer | faer |
|-------|----------|------|
| DuckDB load | 59s | 60s |
| PCA (fit_transform) | 615s | **56s** |
| PCA + 1 iter | 637s | 55s |
| K-means 50 iters overhead | ~4s | 3s |
| Transform only | 5.5s | 3.0s |

## Analysis

**11x speedup exceeds the 3-5x estimate.** The improvement comes from three sources:

1. **faer's blocked GEMM** replaces our naive row-parallel matmul for Y = (X-mean) × Ω. faer uses cache-line-aware tiling, SIMD kernel packing (NEON on Apple Silicon), and better work-stealing parallelism. This is the dominant operation.

2. **faer's thin SVD** replaces our power-iteration eigendecomposition on the small B matrix. The old code computed B×B^T → eigendecomp → reconstruct V. The new code calls `b_mat.thin_svd()` which directly produces the principal directions. More numerically stable and faster.

3. **faer's GEMM for B = Q^T × X_centered** replaces triple-nested manual loops.

The 11x result suggests our original estimate was too conservative — the naive row-parallel matmul was not just cache-unaware but also failing to exploit SIMD at all (scalar loops). faer's NEON-optimized kernels on Apple Silicon are a much bigger win than the 3-5x predicted from cache blocking alone.

## Updated Performance Table

| Workflow | Time | vs Original Baseline |
|----------|------|---------------------|
| Full pipeline (faer PCA + cluster) | **58s** | **10.6x faster** |
| EmbeddingReducer (first run) | **56s** | **11.0x faster** |
| Subsequent run (cached .npy) | 7.5s | 82x faster |
| Transform new data (load model) | 3.0s | 205x faster |
| Matryoshka (no PCA) | ~5s | ~120x faster |

## The 60-Second Target

The PCA optimization plan set a target of 60 seconds for the full pipeline. **We hit it:**

- **faer pipeline: 58s** (PCA 55s + K-means 3s) ✅
- With Matryoshka: ~5s ✅
- With cached data: 7.5s ✅

The only remaining overhead is the DuckDB load (60s). For end-to-end, the total is ~118s (load + PCA + cluster). But the load time is outside rustcluster's scope.

## Objective Comparison

| Backend | Objective (K=98, 50 iters) |
|---------|---------------------------|
| Pre-faer | 213,887 |
| faer | 213,911 |
| Cached data path (exp7) | 214,119 |

Small differences from different random paths through the algorithm (SVD vs eigendecomp produce slightly different component orientations). All within 0.1% — no quality regression.
