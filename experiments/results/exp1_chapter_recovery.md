# Experiment 1: Chapter-Level Cluster Recovery

**Question:** Does EmbeddingCluster recover HTS chapter groupings from product description embeddings?

## Setup
- **Data:** 323,308 CROSS ruling extraction embeddings (text-embedding-3-small, 1536d)
- **Ground truth:** 98 HTS chapters (2-digit codes)
- **Config:** EmbeddingCluster(K=98, PCA→128, n_init=3, max_iter=50)
- **Embedding:** A (description only)

## Results

| Metric | Value |
|--------|-------|
| Purity | 0.5864 |
| NMI | 0.5384 |
| Objective | 214,308.0 |
| Iterations | 50 |
| Time | 633s |
| Resultant (mean) | 0.647 |
| Resultant (min) | 0.507 |
| Resultant (max) | 0.822 |

### Top 5 Clusters by Size

| Rank | Size | Dominant Chapter | Purity |
|------|------|-----------------|--------|
| #1 | 7,675 | 61 (knit apparel) | 93.7% |
| #2 | 6,954 | 29 (organic chemicals) | 91.8% |
| #3 | 6,488 | 62 (woven apparel) | 72.3% |
| #4 | 6,398 | 62 (woven apparel) | 65.9% |
| #5 | 6,152 | 61 (knit apparel) | 95.8% |

## Interpretation

- **58.6% purity** is a strong result for 98 clusters on real-world data where categories have significant semantic overlap (e.g., "men's cotton shorts" vs "men's cotton sleep bottoms" differ only in intended use but fall in different HTS chapters).
- **Chapter 62 splits into two clusters** — the algorithm finds meaningful sub-structure within the large "woven apparel" category.
- **High-purity clusters exist** — chemicals (91.8%) and knit apparel (93.7-95.8%) are cleanly separated.
- **The algorithm converges to max_iter** — quality could improve with more iterations.
