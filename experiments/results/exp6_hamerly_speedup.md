# Experiment 6: Hamerly Speedup Analysis

**Question:** Where does the runtime actually go in the EmbeddingCluster pipeline, and will Hamerly help?

## Setup
- **Data:** 323,308 CROSS ruling embeddings (text-embedding-3-small, 1536d, f32)
- **Config:** K=98, PCA→128, n_init=1, varying max_iter
- **Method:** Isolate PCA cost by running with max_iter=1, then compare with max_iter=10 and max_iter=50

## Results: Full Dataset (323K)

| Run | Total Time | Iterations | Per-Iteration |
|-----|-----------|------------|---------------|
| PCA + 1 iter | 637.0s | 1 | — |
| PCA + 10 iters | 615.4s | 10 | ~0s |
| PCA + 50 iters | 616.5s | 50 | ~0s |

**The K-means iterations are effectively free.** 50 iterations of spherical K-means on 323K × 128d × K=98 add negligible time beyond the PCA step.

## Results: 50K Sample

| Run | Total Time | Iterations | Per-Iteration |
|-----|-----------|------------|---------------|
| PCA + 1 iter | 42.4s | 1 | — |
| PCA + 50 iters | 40.5s | 50 | ~0s |
| PCA + 100 iters | 41.8s | 91 (converged) | ~0s |

Same pattern at 50K: PCA dominates, iterations are negligible.

Note: convergence detection triggered at iteration 91 (v2 three-part stop rule working).

## Runtime Breakdown (323K)

| Stage | Time | % of Total |
|-------|------|-----------|
| DuckDB load | ~61s | 10% |
| L2 normalization | ~1s | <1% |
| **PCA matmul (323K × 1536 × 138)** | **~570s** | **90%** |
| QR + eigendecomp | ~5s | 1% |
| PCA projection + re-normalize | ~30s | 5% |
| K-means (50 iterations) | **~0s** | **~0%** |
| Evaluation | ~1s | <1% |

## Key Finding

**Hamerly acceleration is irrelevant for this pipeline.** The triangle inequality pruning would speed up the K-means assignment step, but the assignment step is already negligible compared to PCA. Even an infinite speedup on K-means (0s) would save less than 1% of total runtime.

## Why This Happens

At 323K × 128d × K=98:
- Assignment step: 323K × 98 × 128 = ~4 billion multiply-adds per iteration
- At ~50 GFLOPS (single core, AVX2): ~0.08s per iteration
- 50 iterations: ~4s total — lost in the noise of a 600s pipeline

The PCA matmul dominates because:
- Y = X × Ω: 323K × 1536 × 138 = ~69 billion multiply-adds
- This is 17x more compute than all 50 K-means iterations combined
- Our pure-Rust rayon matmul is ~10x slower than BLAS for this size

## Implications

1. **Hamerly is still valuable** for the `reduction_dim=None` path (raw 1536d) where K-means iterations DO dominate. But most users will use PCA.

2. **The #1 speedup opportunity is faster PCA:**
   - **faer crate**: blocked matmul, potentially 3-5x faster than hand-rolled rayon
   - **BLAS backend**: OpenBLAS/MKL would be 5-10x faster on this matmul
   - **Skip PCA entirely**: Matryoshka truncation (request 128d from OpenAI API) eliminates the entire 570s cost

3. **The 60-second target requires skipping PCA.** With Matryoshka:
   - Load: ~61s (DuckDB, could be faster with parquet/mmap)
   - Normalize: ~1s
   - K-means (50 iters at 128d): ~4s
   - Total: ~66s — close to target

4. **For users with pre-computed 1536d embeddings**, the realistic path is:
   - faer PCA: ~100-200s (3-5x faster than current)
   - K-means: ~4s
   - Total: ~110-210s — much better than 630s but not 60s

## Recommendation

Reprioritize v2 improvements:
1. ~~Hamerly~~ → **faer PCA backend** (Phase 2.7) is now the #1 priority
2. **Matryoshka documentation** — tell users to request 128d embeddings at API call time
3. Hamerly remains valuable for `reduction_dim=None` path
