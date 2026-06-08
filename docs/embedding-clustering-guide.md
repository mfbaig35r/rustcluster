# Embedding Clustering Guide

A practical guide for clustering dense embedding vectors from OpenAI, Cohere, Voyage, and similar models using rustcluster.

## Quick Start

```python
from rustcluster.experimental import EmbeddingCluster

model = EmbeddingCluster(n_clusters=50, reduction_dim=128)
model.fit(embeddings)  # L2-normalize → PCA → spherical K-means

print(model.labels_)           # cluster assignments
print(model.cluster_centers_)  # unit-norm centroids in reduced space
print(model.objective_)        # sum of cosine similarities
```

`EmbeddingCluster` is the all-in-one pipeline optimized for embedding vectors. It handles normalization, dimensionality reduction, and spherical clustering in a single call.

## Choosing a Reduction Method

High-dimensional embeddings (768d–3072d) benefit from dimensionality reduction before clustering. You have three options:

### Option A: Matryoshka, Skip PCA Entirely (Fastest)

If you're generating **new** embeddings from a model that supports the `dimensions` parameter, request low-dimensional embeddings at API call time:

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small",
    dimensions=128  # ← returns 128d instead of 1536d
)
embeddings_128d = np.array([e.embedding for e in response.data], dtype=np.float32)
```

Then cluster directly; no PCA needed:

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=None)
model.fit(embeddings_128d)  # ~5s for 323K points
```

**Supported models:** OpenAI text-embedding-3-small/large, Cohere embed-v3 (with `input_type`), and other Matryoshka-trained models.

**When to use:** New projects where you control the embedding pipeline. Eliminates the PCA step entirely; reduction cost is 0s.

**Limitation:** Not all models support this. Can't retroactively request lower dimensions for existing embeddings stored in a database.

#### Truncating Existing Matryoshka Embeddings

If you already have high-dimensional embeddings from a Matryoshka-trained model, you can truncate them client-side:

```python
from rustcluster.experimental import EmbeddingReducer

reducer = EmbeddingReducer(target_dim=128, method="matryoshka")
X_reduced = reducer.fit_transform(embeddings)  # slices first 128 cols + L2-normalizes
```

This is instant (no fitting needed). It works because Matryoshka-trained models pack the most important information into the first dimensions. The L2 re-normalization preserves cosine geometry.

### Option B: PCA for Any Embeddings

For embeddings from models that don't support Matryoshka, or existing 1536d data you can't re-embed, use randomized PCA:

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=128)
model.fit(embeddings)  # PCA + cluster in one pipeline
```

PCA is the default. It works with any embedding model but the matrix factorization takes ~10 minutes for 323K × 1536d on Apple Silicon.

**Key optimization: separate reduction from clustering** (see next section).

### Option C: No Reduction

If your embeddings are already low-dimensional (≤256d), either from Matryoshka, a smaller model, or pre-reduced data:

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=None)
model.fit(embeddings_128d)  # direct clustering, no PCA
```

## EmbeddingReducer: Fit Once, Cluster Many Times

PCA is 99% of the EmbeddingCluster runtime. `EmbeddingReducer` separates reduction from clustering so you pay the PCA cost once:

```python
from rustcluster.experimental import EmbeddingReducer
import numpy as np

# Pay the PCA cost once (~10 min for 323K × 1536d)
reducer = EmbeddingReducer(target_dim=128, method="pca")
X_reduced = reducer.fit_transform(embeddings)
np.save("reduced_128.npy", X_reduced)

# Iterate on clustering for free
X_reduced = np.load("reduced_128.npy")  # 0.03s

EmbeddingCluster(n_clusters=50,  reduction_dim=None).fit(X_reduced)  #  4s
EmbeddingCluster(n_clusters=98,  reduction_dim=None).fit(X_reduced)  #  7s
EmbeddingCluster(n_clusters=200, reduction_dim=None).fit(X_reduced)  # 15s
```

### Save/Load the Reducer Model

For production pipelines where new data arrives:

```python
# Save the fitted PCA model (1.5 KB)
reducer.save("pca_128.bin")

# Later: load and project new embeddings (~5s for 323K points)
reducer = EmbeddingReducer.load("pca_128.bin")
X_new_reduced = reducer.transform(new_embeddings)
```

### Access Reduced Data from EmbeddingCluster

If you used the convenience pipeline, access the intermediate PCA output:

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=128)
model.fit(embeddings)

X_reduced = model.reduced_data_  # (n, 128) numpy array, L2-normalized
np.save("reduced_128.npy", X_reduced)
```

Returns `None` if `reduction_dim=None`.

## Using Reduced Data with Any Algorithm

Reduced embeddings work with all rustcluster algorithms:

```python
from rustcluster import KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering

X_reduced = np.load("reduced_128.npy")

# Use metric="cosine"; the data is L2-normalized after reduction
KMeans(n_clusters=50, metric="cosine").fit(X_reduced)
DBSCAN(eps=0.5, metric="cosine").fit(X_reduced)
HDBSCAN(min_cluster_size=100, metric="cosine").fit(X_reduced)
AgglomerativeClustering(n_clusters=50, linkage="average", metric="cosine").fit(X_reduced)
```

## Workflow Decision Tree

```
Do you control the embedding pipeline?
├── YES: Does your model support Matryoshka / dimensions parameter?
│   ├── YES → Request 128d at API time (Option A). Zero reduction cost.
│   └── NO  → EmbeddingReducer(method="pca") (Option B).
└── NO (existing embeddings):
    ├── Already ≤256d? → Skip reduction (Option C).
    ├── From a Matryoshka model? → EmbeddingReducer(method="matryoshka").
    └── Other → EmbeddingReducer(method="pca") (Option B).

Running multiple clustering configs?
→ Always save reduced data and cluster with reduction_dim=None.
```

## Performance Reference

Measured on 323,308 CROSS ruling embeddings (text-embedding-3-small, 1536d → 128d, K=98, Apple Silicon):

| Workflow | Time | vs Baseline |
|----------|------|-------------|
| EmbeddingCluster end-to-end (PCA every run) | 618s | 1x |
| EmbeddingReducer first run (fit_transform + save) | 615s | n/a |
| Subsequent run (load cached .npy + cluster) | **7.5s** | **82x faster** |
| 5 clustering configs on cached data | 74s | 42x vs 5× baseline |
| Matryoshka 128d from API (no PCA at all) | **~5s** | **~120x faster** |

### Persistence Strategy

| Strategy | File Size | Load Time | Best For |
|----------|-----------|-----------|----------|
| Save reduced data (`np.save`) | 316 MB | 0.03s | Same data, iterating on K/algorithm |
| Save reducer model (`.save()`) | 1.5 KB | 0.001s + 5.5s transform | New data, same PCA projection |

## Evaluation and Refinement

### Evaluation Metrics

```python
print(model.intra_similarity_)    # per-cluster avg cosine similarity to centroid
print(model.resultant_lengths_)   # per-cluster directional concentration [0, 1]
print(model.representatives_)     # index of most representative point per cluster
```

### vMF Refinement (Soft Clustering)

For soft cluster probabilities and model selection:

```python
model.refine_vmf()
print(model.probabilities_)   # (n, k) soft membership matrix
print(model.concentrations_)  # per-cluster κ (higher = tighter)
print(model.bic_)             # Bayesian Information Criterion
```

### Finding the Right K

Use BIC from vMF to compare different cluster counts:

```python
X_reduced = np.load("reduced_128.npy")

for k in [20, 50, 98, 150, 200]:
    m = EmbeddingCluster(n_clusters=k, reduction_dim=None).fit(X_reduced)
    m.refine_vmf()
    print(f"K={k:3d}  obj={m.objective_:.0f}  BIC={m.bic_:.0f}")
```

Lower BIC is better (penalizes model complexity).

## Memory-Constrained Workflows (v0.7.0+)

For large embedding datasets on memory-constrained drivers (e.g. a 28 GB Databricks node), three EmbeddingReducer features compose to bring peak Python memory under ~3 GB on the 312K x 1536d workload.

### `chunk_size`: bounded per-call working set

```python
X = reducer.transform(embeddings, chunk_size=50_000)
X = reducer.fit_transform(embeddings, chunk_size=50_000)
```

Per-chunk centered matrix is the only one live at any moment. For 312K x 1536, a 50K chunk caps the centered matrix at ~300 MB (f32) instead of ~1.9 GB. Matryoshka skips chunking (it's a column slice; overhead would dominate).

### `fit_sample_size`: sample-based PCA fit

```python
reducer = EmbeddingReducer(target_dim=128, fit_sample_size=50_000)
reducer.fit(embeddings)  # internally samples to 50K rows
```

PCA's principal directions stabilize well before n_samples reaches dataset size. Sampling uses `random_state` for reproducibility. Pass through to `fit_transform` as well; the chunked transform still operates on the full input.

### `dtype=float32`: f32 fast path

Pass a float32 array and the reducer runs PCA entirely in f32. The n*d centered matrix stays in f32, halving working-set memory. **Output dtype tracks input dtype** as of v0.7.0:

```python
embeddings_f32 = embeddings.astype(np.float32)
X = reducer.fit_transform(embeddings_f32)  # X is float32
```

(Prior to v0.7.0, output was always upcast to float64. If you relied on the upcast, add an explicit `.astype(np.float64)` after `.transform()`.)

### Loading from Spark with no list-of-lists overhead

`toPandas()` on a DataFrame with a 1536d embedding column materializes each row as a Python list of Python floats before NumPy conversion, costing ~3 GB on 312K rows. `extract_embeddings_from_spark` streams via `toLocalIterator()` directly into a pre-allocated NumPy array:

```python
from rustcluster.utils import extract_embeddings_from_spark

embeddings, pdf = extract_embeddings_from_spark(
    df,
    embedding_col="embedding",
    metadata_cols=["supplier_id", "commodity"],
    dtype=np.float32,
    sample_n=None,
)
```

### End-to-end memory-conscious workflow

```python
from rustcluster.utils import extract_embeddings_from_spark
from rustcluster.experimental import EmbeddingReducer, EmbeddingCluster

embeddings, pdf = extract_embeddings_from_spark(
    df, "embedding", ["supplier_id", "commodity"], dtype=np.float32
)
reducer = EmbeddingReducer(target_dim=128, fit_sample_size=50_000)
X = reducer.fit_transform(embeddings, chunk_size=50_000)
model = EmbeddingCluster(n_clusters=20, reduction_dim=None).fit(X)
```

Total Python peak on 312K x 1536d: ~1.5-3 GB, down from ~8-9 GB before v0.7.0.

## Tips

- **Always use f32 for embeddings.** No quality loss vs f64, half the memory, and the f32 fast path in v0.7.0+ keeps it f32 end-to-end.
- **128d is a good default reduction target.** Below 64 you lose information, above 256 PCA gains diminish.
- **Use `metric="cosine"` when clustering reduced data.** The data is L2-normalized after reduction, so cosine distance is the natural metric.
- **Start with n_init=1 for exploration, increase to 3-5 for final results.** Each init re-runs K-means from scratch.
- **Use HDBSCAN if you don't know K.** It finds natural cluster boundaries and labels outliers as -1.
