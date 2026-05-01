# Supplier Clustering — Ultra Performance Methodology

## Overview

Same two-level clustering as the standard pipeline (commodity → sub-commodity) but optimized for maximum speed by exploiting Matryoshka representation learning. Eliminates PCA entirely, reducing pipeline time from minutes to ~60-90 seconds.

## Pipeline

```
312K embeddings (1536d, f32)
    │
    ▼
Matryoshka truncation        ← slice first 128 columns (instant)
    │
    ▼
L2 normalize                 ← unit norm for cosine geometry
    │
    ▼
Commodity clustering         ← spherical K-means
    │
    ▼
Sub-commodity clustering     ← per commodity cluster
    │
    ▼
Delta table output
```

## Why Matryoshka Instead of PCA

### What is Matryoshka Representation Learning?

OpenAI's text-embedding-3-small is trained with Matryoshka representation learning (MRL). During training, the model is jointly optimized so that the **first k dimensions** of the embedding form a valid k-dimensional representation for any k. The dimensions are ordered by importance — dimension 1 carries the most information, dimension 1536 the least.

This means you can simply slice the first 128 columns of a 1536d embedding and get a high-quality 128d representation — no matrix factorization needed.

### Matryoshka vs PCA on This Dataset

From experiment 9 (323K CROSS ruling embeddings, K=98):

| Method | Purity | NMI | Time |
|--------|--------|-----|------|
| PCA → 128d | 0.5739 | 0.5331 | 56s |
| Matryoshka → 128d | 0.5737 | 0.5097 | 0.01s |

Purity is virtually identical. NMI is slightly lower for Matryoshka — it finds different but equally valid cluster structures (ARI between the two clusterings is 0.48, meaning they agree on about half of point-pair groupings).

### When to Use Each

| Criterion | Matryoshka | PCA |
|-----------|-----------|-----|
| Embedding model | text-embedding-3-small/large, Cohere embed-v3 | Any model |
| Speed | Instant | ~56s for 312K × 1536 |
| Memory | Negligible | ~2-3 GB peak |
| Quality | Equivalent purity | Slightly higher NMI |
| Recommended when | Speed matters, Matryoshka model | Non-Matryoshka model, or maximum NMI needed |

## Performance Breakdown

### Expected Timing (312K Records, 28 GB Single Node)

| Stage | Matryoshka (ultra) | PCA (standard) |
|-------|-------------------|----------------|
| Load + stream embeddings | ~60s | ~60s |
| Dimensionality reduction | **<0.1s** | ~30-60s |
| Commodity clustering (K≤20) | ~3-5s | ~3-5s |
| Sub-commodity clustering | ~10-20s | ~10-20s |
| Save to Delta | ~10s | ~10s |
| **Total** | **~80-90s** | **~120-150s** |

Load time dominates — streaming 312K × 1536 embeddings row-by-row from Delta is I/O-bound. The actual clustering is seconds.

### Memory Profile

| Object | Size |
|--------|------|
| Metadata DataFrame (no embeddings) | ~50 MB |
| Embeddings f32 (312K × 1536) | 1.9 GB |
| Reduced f64 (312K × 128) | 320 MB |
| EmbeddingCluster internals | ~50 MB |
| **Peak Python memory** | **~2.3 GB** |

Well within the ~14 GB available to Python on a 28 GB Databricks node.

## Clustering Details

### Algorithm: Spherical K-means

Identical to the standard pipeline. Maximizes cosine similarity on the unit hypersphere. Centroids are re-normalized after each update.

- **n_init:** 3 (3 random restarts, best objective wins)
- **max_iter:** 100 per restart
- **Convergence:** angular shift < 1e-6

### K Selection

Commodity clusters:
```
K = clamp(distinct_commodities * 0.1, min=2, max=20)
```

Sub-commodity clusters (per commodity cluster):
```
K = clamp(distinct_sub_commodities * 0.1, min=2, max=25)
K = min(K, group_size // 2)
```

### Quality Metric

Silhouette score computed on a 10K random sample (full silhouette is O(n²), impractical at 312K).

## Differences from Standard Pipeline

| Aspect | Standard | Ultra |
|--------|----------|-------|
| Reduction | PCA (sample fit + chunked transform) | Matryoshka (column slice) |
| Reduction time | ~30-60s | <0.1s |
| Reduction quality | Data-adaptive projection | Model-pretrained prefix |
| Memory for reduction | ~2-3 GB | Negligible |
| Total time | ~2-3 min | ~60-90s |
| Output table columns | Same | Same + `reduction_method`, `reduction_dim` |

## Configuration Toggle

The notebook supports switching between methods via config:

```python
"reduction_method": "matryoshka",  # instant — for text-embedding-3
"reduction_method": "pca",         # sample-fit PCA — for any model
```

Both paths produce the same output schema. The choice is purely performance vs model compatibility.

## Limitations

- **Matryoshka only works for Matryoshka-trained models.** If the embeddings come from a model that wasn't trained with MRL (e.g., older ada-002, Voyage, some Cohere models), the first 128 dimensions are NOT a valid low-dimensional representation. Use PCA instead.
- **text-embedding-3-small and text-embedding-3-large are confirmed Matryoshka-trained** by OpenAI's documentation.
- **Quality vs PCA:** Purity is equivalent, NMI is slightly lower. For most production use cases this difference is negligible. If maximum clustering quality is critical, use PCA.
