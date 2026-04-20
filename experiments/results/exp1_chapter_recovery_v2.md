# Experiment 1 (v2): Chapter-Level Cluster Recovery

**Question:** Does the v2 EmbeddingCluster (with convergence detection, chunked vMF, Matryoshka, Hamerly) change results on the CROSS ruling dataset?

## Setup
- **Data:** 323,308 CROSS ruling extraction embeddings (text-embedding-3-small, 1536d)
- **Ground truth:** 98 HTS chapters (2-digit codes)
- **Config:** EmbeddingCluster(K=98, PCA→128, n_init=3, max_iter=50)
- **Embedding:** A (description only)
- **Code version:** v2 (convergence detection, chunked vMF, Matryoshka, Hamerly, mini-batch, multi-view fusion, GIL interruptibility)

## Results

| Metric | v1 | v2 | Delta |
|--------|----|----|-------|
| Purity | 0.5864 | 0.5864 | 0 |
| NMI | 0.5384 | 0.5384 | 0 |
| Objective | 214,308.0 | 214,308.0 | 0 |
| Iterations | 50 | 50 | 0 |
| Time | 633s | 651.5s | +18.5s (noise) |
| Resultant (mean) | 0.647 | 0.647 | 0 |
| Resultant (min) | 0.507 | 0.507 | 0 |
| Resultant (max) | 0.822 | 0.822 | 0 |

### Top 5 Clusters by Size (unchanged)

| Rank | Size | Dominant Chapter | Purity |
|------|------|-----------------|--------|
| #1 | 7,675 | 61 (knit apparel) | 93.7% |
| #2 | 6,954 | 29 (organic chemicals) | 91.8% |
| #3 | 6,488 | 62 (woven apparel) | 72.3% |
| #4 | 6,398 | 62 (woven apparel) | 65.9% |
| #5 | 6,152 | 61 (knit apparel) | 95.8% |

## Interpretation

**Results are identical to v1.** This is expected and validates the v2 code:

1. **Same seed → same initialization → same results.** The v2 convergence detection, Hamerly bounds, and other features don't change the algorithm's output — they change how fast it gets there.

2. **Convergence detection did NOT trigger.** The three-part stop rule (relative objective < 1e-4 AND churn < 0.1% for 2 consecutive iterations) never fires on this dataset. With 323K points in 98 overlapping clusters, hundreds of borderline points flip every iteration — churn stays above 0.1% through all 50 iterations. The algorithm genuinely hasn't converged.

3. **The +18.5s difference is noise** — different system load, DuckDB read time, etc. The algorithm's per-iteration cost is unchanged since Hamerly is not yet wired into the EmbeddingCluster pipeline (it exists as a standalone function but the default `fit()` still calls Lloyd-style spherical K-means).

## Key Takeaway

On real-world data with overlapping categories, the v2 improvements don't change quality or runtime because:
- Convergence detection helps on **well-separated** data (6 iterations vs 50 on synthetic) but not on **overlapping** data
- The Hamerly accelerator will help by making **each iteration cheaper** (2-8x fewer distance computations), not by reducing iteration count
- Mini-batch would help by processing subsets per iteration, trading quality for speed

The next experiment should wire Hamerly into the `fit()` pipeline and re-run to measure the actual speedup.
