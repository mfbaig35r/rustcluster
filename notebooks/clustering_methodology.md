# Supplier Clustering Methodology

## Overview

Two-level hierarchical clustering of 312K supplier embeddings by commodity similarity. Each supplier's `concatenated_text` (industry + commodity + sub_commodity + summary) was embedded using OpenAI's text-embedding-3-small (1536 dimensions). The clustering groups suppliers into commodity clusters, then further segments each commodity cluster into sub-commodity clusters.

## Pipeline

```
312K embeddings (1536d, f32)
    │
    ▼
PCA (1536d → 128d)          ← fit on 50K sample, transform all
    │
    ▼
L2 normalize                 ← all rows to unit norm
    │
    ▼
Commodity clustering         ← spherical K-means, K from data
    │
    ▼
Sub-commodity clustering     ← per commodity cluster, spherical K-means
    │
    ▼
Delta table output           ← with cluster assignments + run metadata
```

## Method Choices

### Dimensionality Reduction: Randomized PCA → 128d

1536 dimensions is too many for effective clustering — at high dimensions, distances between points converge (curse of dimensionality), making cluster boundaries fuzzy. PCA projects to 128 dimensions that capture the most variance, removing noise dimensions.

- **Algorithm:** Halko-Martinsson-Tropp randomized SVD, backed by faer's SIMD-accelerated GEMM
- **Target:** 128 dimensions (experimentally validated — below 64 loses information, above 256 has diminishing returns)
- **Fit on sample:** PCA is fitted on 50K rows (statistically identical to full data — covariance stabilizes well before 50K) and applied to all 312K
- **Post-PCA normalization:** rows are L2-normalized after projection to preserve cosine geometry

### Clustering Algorithm: Spherical K-means

Standard K-means minimizes squared Euclidean distance — it pulls centroids toward the mean of their assigned points. This is wrong for embeddings, which live on the unit hypersphere (their magnitude is meaningless, only direction matters).

Spherical K-means instead maximizes cosine similarity. After each centroid update, centroids are re-normalized to the unit sphere. This means the algorithm is finding groups of similar *directions*, not similar *positions* — which is the correct geometry for embedding vectors.

- **Initialization:** K-means++ (adapted for cosine)
- **n_init:** 3 restarts, best objective wins
- **max_iter:** 100 per restart
- **Convergence:** angular shift tolerance 1e-6

### K Selection: Heuristic (10% of Distinct Values)

For commodity clusters:
```
K = max(2, int(distinct_commodities * 0.1))
K = min(K, 20)
```

For sub-commodity clusters within each commodity cluster:
```
K = max(2, int(distinct_sub_commodities * 0.1))
K = min(K, 25, group_size // 2)
```

This heuristic targets roughly 10 suppliers per group as a starting segmentation. It's fast (no iterative search) and produces reasonable granularity for downstream use.

### Quality Metric: Sampled Silhouette Score

Silhouette score measures how well each point fits its assigned cluster vs the nearest other cluster. Range [-1, 1], higher is better.

Computed on a 10K random sample (full silhouette is O(n²) — 312K² = 97 billion distance computations, would take hours). Sampled silhouette is a reliable estimate.

## What Changed: sklearn vs rustcluster

### The sklearn Pipeline (Before)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

embeddings_scaled = StandardScaler().fit_transform(embeddings)  # z-score normalize
labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings_scaled)
```

**What it did:**
1. **StandardScaler** — z-score normalization (subtract mean, divide by std per dimension). This centers data at the origin and equalizes variance across dimensions.
2. **Euclidean K-means** — minimizes squared L2 distance to centroids.
3. **No PCA** — clustered in the full 1536d space.
4. **Distributed processing** — used `mapInPandas` and `groupBy.applyInPandas` to run sklearn on Spark partitions, with extensive OOM workarounds (repartitioning to 1000+ partitions, gc.collect, batch processing).

### The rustcluster Pipeline (After)

```python
from rustcluster.experimental import EmbeddingReducer, EmbeddingCluster

X_reduced = EmbeddingReducer(target_dim=128).fit_transform(embeddings)  # PCA + L2 normalize
model = EmbeddingCluster(n_clusters=k, reduction_dim=None).fit(X_reduced)  # spherical K-means
```

**What it does:**
1. **PCA → 128d** — randomized PCA reduces from 1536 to 128 dimensions, removing noise.
2. **L2 normalization** — projects to unit sphere (preserves cosine geometry).
3. **Spherical K-means** — maximizes cosine similarity, not Euclidean distance.
4. **Single-node** — fast enough to run entirely on the driver without Spark distribution.

### Key Differences

| Aspect | sklearn (before) | rustcluster (after) |
|--------|-----------------|-------------------|
| **Normalization** | StandardScaler (z-score) | L2 normalization (unit sphere) |
| **Distance metric** | Euclidean | Cosine (spherical K-means) |
| **Dimensionality** | Full 1536d | PCA → 128d |
| **Geometry** | Flat space | Unit hypersphere |
| **n_init** | 10 | 3 (faster, sufficient with better geometry) |
| **Distribution** | Spark mapInPandas + OOM workarounds | Single-node (fast enough) |
| **Sub-clustering** | Per-partition StandardScaler (inconsistent normalization) | Consistent — all data normalized once |

### Why These Changes Matter

**StandardScaler vs L2 normalization:** StandardScaler treats each dimension independently — it doesn't know that embedding dimensions are learned features that interact. L2 normalization preserves the angular structure that the embedding model was trained to produce. For cosine-similarity-based embeddings (which is what text-embedding-3-small produces), L2 normalization is the correct preprocessing.

**Euclidean vs cosine K-means:** Euclidean distance in high dimensions is dominated by the curse of dimensionality — all pairwise distances converge toward the same value. Cosine similarity measures angular separation, which remains discriminative even at high dimensions. Spherical K-means is the correct algorithm for unit-normalized embeddings.

**Full 1536d vs PCA 128d:** The 1536 dimensions contain redundancy and noise. PCA extracts the 128 most informative directions. Clustering in 128d is more effective (cleaner signal) and ~12x cheaper computationally.

**Inconsistent sub-cluster normalization:** The sklearn pipeline ran `StandardScaler().fit_transform()` independently inside each Spark partition during sub-commodity clustering. This means each partition learned its own mean and standard deviation — points in different partitions of the same commodity cluster were normalized differently. The rustcluster pipeline normalizes all data once before clustering, ensuring consistency.

### Expected Impact on Results

The cluster assignments will differ from the sklearn pipeline because:

1. Different normalization (L2 vs z-score) changes the distance geometry
2. Different algorithm (cosine vs Euclidean) finds different cluster boundaries
3. PCA reduction changes the feature space

These are intentional improvements. The clusters should be more semantically coherent because the algorithm now respects the cosine geometry that the embedding model was trained for.

Direct comparison of cluster quality between the two pipelines requires domain evaluation (e.g., do the commodity clusters group suppliers that procurement teams would consider interchangeable?). The silhouette score provides a geometry-based quality signal but is not a substitute for business validation.
