# Experiment 7: EmbeddingReducer Workflow — Pay Once, Cluster for Free

**Date:** 2026-04-20
**Question:** How much time does the new EmbeddingReducer save when iterating on clustering configs?

## Setup
- **Data:** 323,308 CROSS ruling embeddings (text-embedding-3-small, 1536d, f32)
- **Config:** K=98, PCA→128, n_init=1, max_iter=50
- **Machine:** Apple Silicon (arm64)

## Results

### End-to-End Timing

| Workflow | Time | Speedup |
|----------|------|---------|
| **A. Baseline** — EmbeddingCluster (PCA + cluster) | 618s | 1x |
| **B. First run** — EmbeddingReducer fit_transform + save | 615s | — |
| **C. Cached model** — load reducer + transform new data | 5.5s | 112x (transform only) |
| **D. Cached data** — np.load + cluster | 7.5s | **82x** |
| **E. 5 configs** — iterate on K, n_init, algorithm | 73.9s | **42x** vs 5x baseline |

### Breakdown: Where Time Goes

| Operation | Time |
|-----------|------|
| DuckDB load (323K embeddings) | 58.7s |
| PCA fit (eigendecomp + matmul) | ~610s |
| PCA transform only (projection) | 5.5s |
| Reducer save (1.5 MB binary) | 0.001s |
| Reducer load | 0.001s |
| np.save (316 MB reduced data) | 0.04s |
| np.load (316 MB reduced data) | 0.03s |
| EmbeddingCluster K=98 (on 128d) | 7.5s |

### Multi-Config Iteration (Section E)

All runs on cached 128d reduced data — zero PCA cost:

| Config | Time | Iters | Objective |
|--------|------|-------|-----------|
| EmbeddingCluster K=50 | 4.0s | 46 | 201,613 |
| EmbeddingCluster K=98 | 7.5s | 50 | 214,119 |
| EmbeddingCluster K=200 | 15.2s | 50 | 225,868 |
| KMeans K=98 cosine | 9.6s | 36 | 109,201 |
| EmbeddingCluster K=98 n_init=5 | 37.6s | 50 | 214,371 |
| **Total (5 configs)** | **73.9s** | | |

Without EmbeddingReducer, these 5 runs would cost 5 × 618s = **3,090s** (51 minutes).
With cached data: **73.9s** (1.2 minutes). **42x speedup.**

## Key Findings

1. **82x speedup on second run.** Once PCA is cached as a .npy file, clustering 323K embeddings drops from 618s to 7.5s.

2. **Transform-only is fast.** Loading the 1.5KB reducer model and projecting 323K × 1536 → 128 takes 5.5s. This is the projection matmul only (no eigendecomp, no random matrix generation).

3. **reduced_data_ works.** The cached reduced data from `EmbeddingCluster.reduced_data_` has correct shape (323308, 128), all rows unit-norm, and can be directly clustered with `reduction_dim=None`.

4. **Save/load is instant.** Reducer model: 1.5KB, loads in 0.001s. Reduced data: 316MB numpy, loads in 0.03s.

5. **PCA is still 99% of first-run cost.** 610s of 618s. faer integration remains the path to making the first run faster.

## Two Persistence Strategies

| Strategy | Save | Load | Re-cluster | Use When |
|----------|------|------|------------|----------|
| **Save reducer model** (1.5KB) | 0.001s | 0.001s + 5.5s transform | 7.5s | New data arrives, need to project with same PCA |
| **Save reduced data** (316MB) | 0.04s | 0.03s | 7.5s | Same data, iterating on clustering params |

## Recommendation

- For iterating on K, algorithm, n_init: save the reduced numpy array
- For production pipelines with new data: save the reducer model
- Both paths eliminate 610s of PCA from every subsequent run
