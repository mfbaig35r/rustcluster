# Clustering OpenAI Embeddings with rustcluster

A practical guide for clustering large sets of text embeddings (100K-400K+ vectors at 1536 dimensions).

## The Problem

OpenAI's text-embedding-3-small produces 1536-dimensional vectors. At scale, this creates memory and performance challenges:

| Dataset size | Raw f64 memory | Raw f32 memory |
|-------------|---------------|---------------|
| 100K embeddings | 1.2 GB | 600 MB |
| 200K embeddings | 2.3 GB | 1.2 GB |
| 400K embeddings | 4.7 GB | 2.3 GB |

scikit-learn's K-means adds intermediate matrices on top of this (copies, distance matrices, bounds arrays), often doubling or tripling the peak memory. This is where OOM errors come from.

## Recommended Pipeline

### Step 1: Load as float32

OpenAI embeddings are effectively float32 precision — the API returns float64 but the model produces float32. Casting to f32 halves your memory with no quality loss.

```python
import numpy as np

# If loading from a file/database
embeddings = np.array(your_embeddings, dtype=np.float32)  # (n, 1536)
```

### Step 2: Reduce dimensionality

1536 dimensions is too many for effective clustering. At high dimensions, Euclidean distance becomes less meaningful (the "curse of dimensionality" — all points become roughly equidistant). Reducing to 20-50 dimensions:

- Removes noise dimensions
- Improves cluster quality
- Massively reduces memory and compute

```python
from sklearn.decomposition import PCA

# Reduce from 1536 to 32 dimensions
reduced = PCA(n_components=32).fit_transform(embeddings).astype(np.float32)
# (400K, 32) × 4 bytes = 49 MB — down from 2.3 GB
```

**Alternatives to PCA:**
- **UMAP** — better at preserving local structure, slower to fit. Use `umap-learn` package.
- **t-SNE** — good for visualization (2-3d), not recommended for clustering input.

### Step 3: Cluster with rustcluster

With reduced f32 data, you're in rustcluster's sweet spot (d < 20, f32, low memory).

**K-Means** — when you know the number of clusters:

```python
from rustcluster import KMeans, silhouette_score

model = KMeans(n_clusters=50, random_state=42)
model.fit(reduced)

print(f"Clusters: {len(set(model.labels_))}")
print(f"Silhouette: {silhouette_score(reduced, model.labels_):.3f}")
```

**HDBSCAN** — when you don't know the number of clusters:

```python
from rustcluster import HDBSCAN

model = HDBSCAN(min_cluster_size=100)
model.fit(reduced)

n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
n_noise = sum(model.labels_ == -1)
print(f"Clusters: {n_clusters}, Noise: {n_noise}")
print(f"Probabilities: {model.probabilities_[:5]}")
```

**MiniBatchKMeans** — for very large datasets (1M+):

```python
from rustcluster import MiniBatchKMeans

model = MiniBatchKMeans(n_clusters=50, batch_size=1024, random_state=42)
model.fit(reduced)
```

### Step 4: Evaluate

```python
from rustcluster import silhouette_score, calinski_harabasz_score, davies_bouldin_score

labels = model.labels_

print(f"Silhouette:        {silhouette_score(reduced, labels):.3f}")     # [-1, 1], higher = better
print(f"Calinski-Harabasz: {calinski_harabasz_score(reduced, labels):.1f}")  # higher = better
print(f"Davies-Bouldin:    {davies_bouldin_score(reduced, labels):.3f}")  # lower = better
```

## Alternative: Cluster Raw Embeddings with Cosine Distance

If you don't want to reduce dimensions, you can cluster the raw 1536d embeddings with cosine distance. Cosine is more appropriate than Euclidean for normalized embeddings because it measures angular similarity.

```python
from rustcluster import KMeans

# No dimensionality reduction — cluster raw embeddings
model = KMeans(n_clusters=50, metric="cosine", random_state=42)
model.fit(embeddings)  # embeddings should be f32 for memory efficiency
```

This works and avoids sklearn's OOM issues (no intermediate distance matrix), but is slower per-iteration than the PCA + Euclidean approach because:
- d=1536 means more work per distance computation
- cosine distance forces Lloyd's algorithm (Hamerly's bounds assume Euclidean)
- No KD-tree acceleration for cosine

**When to use raw cosine clustering:**
- Small-to-medium datasets (< 50K embeddings)
- When PCA might lose important semantic structure
- When you need the simplest possible pipeline

**When to reduce first:**
- Large datasets (100K+)
- When you need fast iteration on cluster count
- When memory is constrained

## Memory Comparison

For 400K embeddings, k=50:

| Approach | Peak memory |
|----------|------------|
| sklearn K-means on raw f64 embeddings | ~6-10 GB |
| rustcluster cosine K-means on raw f32 embeddings | ~2.4 GB |
| rustcluster K-means on PCA-reduced f32 (d=32) | ~60 MB |

## Finding the Right Number of Clusters

If you don't know how many clusters to use, try:

**1. Elbow method with inertia:**

```python
from rustcluster import KMeans

inertias = []
for k in range(5, 105, 5):
    model = KMeans(n_clusters=k, random_state=42, n_init=3).fit(reduced)
    inertias.append((k, model.inertia_))
    print(f"k={k:3d}  inertia={model.inertia_:.0f}")

# Look for the "elbow" — where inertia stops decreasing sharply
```

**2. Silhouette sweep:**

```python
from rustcluster import KMeans, silhouette_score

for k in [10, 20, 30, 50, 75, 100]:
    model = KMeans(n_clusters=k, random_state=42, n_init=3).fit(reduced)
    score = silhouette_score(reduced, model.labels_)
    print(f"k={k:3d}  silhouette={score:.3f}")

# Higher silhouette = better-defined clusters
```

**3. Let HDBSCAN decide:**

```python
from rustcluster import HDBSCAN

# HDBSCAN automatically determines cluster count
model = HDBSCAN(min_cluster_size=50).fit(reduced)
n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
print(f"HDBSCAN found {n_clusters} clusters ({sum(model.labels_ == -1)} noise points)")
```

## Full Example

```python
import numpy as np
from sklearn.decomposition import PCA
from rustcluster import KMeans, HDBSCAN, silhouette_score

# 1. Load embeddings
embeddings = np.load("embeddings.npy").astype(np.float32)  # (200K, 1536)
print(f"Raw: {embeddings.shape}, {embeddings.nbytes / 1e6:.0f} MB")

# 2. Reduce dimensions
reduced = PCA(n_components=32).fit_transform(embeddings).astype(np.float32)
print(f"Reduced: {reduced.shape}, {reduced.nbytes / 1e6:.0f} MB")

# 3. Cluster
model = KMeans(n_clusters=30, random_state=42).fit(reduced)

# 4. Evaluate
score = silhouette_score(reduced, model.labels_)
print(f"Silhouette: {score:.3f}")

# 5. Inspect
for cluster_id in range(model.cluster_centers_.shape[0]):
    mask = model.labels_ == cluster_id
    print(f"Cluster {cluster_id}: {mask.sum()} items")
```

## Tips

- **Always use f32 for embeddings.** No quality loss, half the memory.
- **PCA to 20-50 dimensions is usually the right range.** Below 20 you lose information, above 50 you're not gaining much.
- **Start with HDBSCAN if you don't know the cluster count.** It handles noise (outliers get label -1) and finds natural cluster boundaries.
- **Use n_init=3-5 for K-means, not the default 10.** On large datasets, 3 initializations is usually enough and saves 2-3x runtime.
- **Pickle your fitted model** so you don't have to recluster:
  ```python
  import pickle
  pickle.dump(model, open("cluster_model.pkl", "wb"))
  model = pickle.load(open("cluster_model.pkl", "rb"))
  ```
