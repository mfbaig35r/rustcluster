# Cluster Memory: Turning `fit()` into a Persistent Classification System

Most clustering libraries treat clustering as a one-shot computation. You call `fit()`, you get labels, you move on. If new data shows up tomorrow, you refit from scratch. The cluster topology -- the thing you spent minutes or hours computing -- is trapped inside a transient Python object that dies when your process exits.

I built Cluster Slotting in rustcluster to fix this. The idea is simple: after you fit a clustering model, you should be able to freeze the cluster topology, save it to disk, load it on a different machine, and assign new points in milliseconds -- with confidence scores, rejection logic, and drift detection. Clustering should work like a trained classifier, not a disposable computation.

This post walks through the complete feature set: how it works, why it's designed the way it is, and what the numbers look like on real embedding data.

## The Problem

I work with high-dimensional embedding data -- 1536-dimensional vectors from OpenAI models, hundreds of thousands of them. Clustering these embeddings is the foundation for things like tariff classification and supplier categorization. The clustering step itself takes 30-60 seconds. That's fine for a batch job, but it breaks down in three scenarios:

1. **New data arrives continuously.** You can't refit every time a new supplier or ruling shows up.
2. **The cluster topology is the artifact.** You want to assign new items to *the same* clusters, not whatever a fresh `fit()` decides today.
3. **You need confidence.** Not every point belongs cleanly to a cluster. Some are boundary cases. Some are genuinely novel. You need to know which is which.

sklearn's `KMeans.predict()` handles scenario 1, but it has no persistence format, no confidence scoring, and no way to detect when the world has shifted enough that you should refit. HDBSCAN has `approximate_predict()` but no save/load. FAISS does centroid routing internally but doesn't expose it as a first-class assignment API.

## ClusterSnapshot: The Core Primitive

After fitting any centroid-based model in rustcluster, you call `snapshot()` to freeze the cluster state:

```python
from rustcluster import KMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster

model = EmbeddingCluster(n_clusters=20, reduction_dim=128).fit(X_train)
snapshot = model.snapshot()
```

A snapshot captures everything needed to reproduce the assignment decision: centroids, the preprocessing pipeline (L2 normalization, PCA projection), the distance metric, and metadata about the training run. For `EmbeddingCluster`, this means the full pipeline -- L2-normalize raw embeddings, project through PCA, L2-normalize again, assign via spherical (max dot product) search -- is baked into the snapshot. Users pass raw embeddings and get labels back.

```python
labels = snapshot.assign(X_new)
```

The snapshot is a Rust struct with `Arc<Vec<f64>>` centroids, which means it's `Send + Sync` -- safe to share across threads. Assignment runs in Rust with the GIL released, parallelized via rayon. The Python wrapper is thin: validate the array shape, hand it to Rust, return the result.

### What Gets Frozen

- **Centroids** (flat row-major f64, shape k x d)
- **Preprocessing pipeline** (None, L2Normalize, or EmbeddingPipeline with PCA projection)
- **Distance metric and spherical flag** (min-distance vs. max-dot-product)
- **Training metadata** (cluster sizes, sample count, per-cluster mean distances)
- **Calibration data** (optional -- more on this below)

What doesn't get frozen: the training data itself. A snapshot for 20 clusters in 128 dimensions is 21 KB. The training data that produced it was 1.5 GB. That's the point.

## Persistence: safetensors + JSON

```python
snapshot.save("clusters/")
snapshot = ClusterSnapshot.load("clusters/")
```

On disk, a snapshot is a directory with two files:

```
clusters/
  centroids.safetensors   # float arrays (centroids, PCA components, variances)
  metadata.json           # human-readable config + calibration statistics
```

I chose safetensors because it's zero-copy, doesn't execute arbitrary code on load (unlike pickle), and has native Rust and Python support. The JSON sidecar is there so you can `cat metadata.json` and see what's inside -- the algorithm, dimensionality, cluster count, calibration status. It's versioned (currently version 2), and version 1 snapshots load fine with all calibration fields defaulting to None.

Snapshot creation is sub-millisecond. Save and load are also sub-millisecond for typical sizes. A 20-cluster, 128-dimensional snapshot is 21 KB. Even a 98-cluster snapshot with PCA projection is only 1.6 MB.

## Confidence Scoring

Plain `assign()` returns labels. `assign_with_scores()` returns labels, distances, confidences, and rejection flags:

```python
result = snapshot.assign_with_scores(X_new)
result.labels_        # cluster indices (or -1 if rejected)
result.distances_     # distance to nearest centroid
result.confidences_   # [0, 1) confidence score
result.rejected_      # boolean mask
```

Confidence is computed from the top-2 margin ratio. For each point, I find both the nearest and second-nearest centroid in a single pass using `assign_nearest_two_with()`. The confidence formula depends on the metric:

- **Distance metrics** (Euclidean, Manhattan): `1 - (nearest / second_nearest)`
- **Dot product** (spherical/cosine): `1 - (second_best / best)`

The result is bounded [0, 1), where 0 means the point sits exactly on the decision boundary between two clusters, and values approaching 1 mean the assignment is decisive.

The boundary behavior is clean. If you walk a point from centroid A at x=0 to centroid B at x=10:

| Position | Confidence | Note |
|----------|------------|------|
| x=0.0 | 1.000 | On centroid A |
| x=4.5 | 0.333 | Approaching boundary |
| x=5.0 | 0.000 | Exact boundary |
| x=5.5 | 0.333 | Other side |
| x=10.0 | 1.000 | On centroid B |

Perfect symmetry, no discontinuities. Ties go to the lower-indexed cluster.

## The Rejection Sweep Problem

Confidence scores are useful, but turning them into accept/reject decisions reveals a fundamental problem with global thresholds. Here's real data from a supplier embedding dataset (312K x 1536, K=20):

| Method | Rejected | Commodity Purity | Kept |
|--------|----------|-----------------|------|
| No rejection | 0% | 0.782 | 62,477 |
| Global 0.10 | 54% | 0.820 | 28,640 |
| Global 0.30 | 98.5% | 0.940 | 909 |

With a global threshold of 0.30, you reject 98.5% of your data. You get beautiful purity on the 909 survivors, but you've thrown away the dataset. Drop to 0.10, and you keep half, but purity only improves modestly. Global thresholds force a binary choice between keeping everything (low purity) or keeping almost nothing (high purity).

The problem is that clusters have different natural spreads. A tight, well-separated cluster might have training confidences centered around 0.6. A diffuse cluster in a crowded region might peak at 0.15. A global threshold of 0.30 keeps almost everything from the tight cluster and rejects almost everything from the diffuse one. You're not filtering by quality -- you're filtering by cluster geometry.

## Calibration and Adaptive Thresholds

This is what `calibrate()` solves. Pass it the training data (or a representative sample), and it computes three things in a single pass:

1. **Per-cluster confidence quantiles** (P5, P10, P25, P50)
2. **Per-cluster per-dimension variance** (diagonal covariance)
3. **Per-cluster vMF concentration parameter kappa** (spherical metrics only)

```python
snapshot.calibrate(X_train)
```

Calibration is optional. An uncalibrated snapshot works perfectly for basic assignment. But once calibrated, you can use adaptive thresholds:

```python
result = snapshot.assign_with_scores(
    X_new,
    adaptive_threshold=True,
    adaptive_percentile="p10"
)
```

With `adaptive_threshold=True`, each point is compared against its *own cluster's* training distribution. A point assigned to cluster 7 is rejected if its confidence falls below cluster 7's P10 -- not some global number. The results on the same supplier data:

| Method | Rejected | Commodity Purity | Kept |
|--------|----------|-----------------|------|
| Adaptive P5 | 5% | 0.785 | 59,343 |
| Adaptive P10 | 10% | 0.788 | 56,159 |
| Adaptive P50 | 50% | 0.814 | 31,116 |

Adaptive P10 rejects 10% of points and keeps 56,159 items instead of 909. The purity improvement is modest at P10 (0.782 to 0.788), but the coverage is dramatically better. You're rejecting the genuine outliers from each cluster rather than wiping out entire clusters because their geometry happens to produce lower confidence scores.

This is the most important design decision in the feature: per-cluster thresholds that respect cluster geometry, computed from one calibration call and persisted through save/load.

## Diagonal Mahalanobis Boundaries

Calibration also enables Mahalanobis-distance boundaries:

```python
labels = snapshot.assign(X_new, boundary_mode="mahalanobis")
```

Standard Voronoi assignment draws straight-line boundaries between centroids. Mahalanobis boundaries account for each cluster's per-dimension variance -- elongated clusters get elongated decision regions. The storage cost is O(k * d), just the diagonal of the covariance matrix per cluster.

I should be honest about the results here. On embedding data, Mahalanobis boundaries are marginal. On supplier embeddings (312K, K=20), commodity purity went from 0.782 to 0.792 -- a real but small improvement. Embeddings tend to be roughly isotropic after L2 normalization, so the diagonal covariance doesn't capture much that Voronoi misses. Mahalanobis would matter more on raw feature data with very different scales per dimension. I included it because the implementation cost is negligible once you have per-cluster variances from calibration, and it does help on non-embedding data.

## Drift Detection

A snapshot is a promise that the cluster topology from training time is still valid. Drift detection tells you when that promise is breaking down.

```python
report = snapshot.drift_report(X_recent)
```

The drift report computes per-cluster statistics against the training baseline. For Euclidean metrics, that's mean distance drift and rejection rate. For spherical metrics (EmbeddingCluster), it adds two signals:

- **vMF kappa drift**: how the concentration of points around each centroid has changed. If kappa drops, the cluster is becoming more diffuse -- new data isn't as tightly grouped.
- **Centroid direction drift**: `1 - dot(old_centroid, new_mean_direction)`. If the centroid of new data has rotated away from the training centroid, the cluster meaning has shifted.

The multi-signal approach matters because no single drift metric is reliable on its own. A cluster could maintain its kappa (same spread) while its direction drifts (different content), or vice versa.

On CROSS ruling embeddings (323K, K=98), the vMF kappa discrimination ratio between in-distribution test data and random data was 124x. On supplier embeddings, it was 1,268x. These are large enough margins that you can set thresholds with confidence -- in-distribution data looks nothing like random noise to the drift detector.

## Hierarchical Slotting

Single-level clustering has an inherent resolution limit. With K=20 commodity clusters, you get commodity-level purity of 0.782 but can't distinguish sub-commodities within each cluster. Increasing K to 200 fragments the clusters without improving the hierarchy.

`HierarchicalSnapshot` chains a root snapshot with per-cluster child snapshots:

```python
from rustcluster.experimental import HierarchicalSnapshot

hier = HierarchicalSnapshot.build(X_train, model, n_sub_clusters=10)
root_labels, child_labels = hier.assign(X_new)
```

The `build()` convenience method fits a sub-clustering model within each root cluster. Assignment cascades: first assign to the root, then route each point to its root cluster's child snapshot. When combined with adaptive rejection, points rejected at the root level are short-circuited -- they don't get passed to child snapshots.

```python
result = hier.assign_with_scores(
    X_new,
    adaptive_threshold=True,
    adaptive_percentile="p10"
)
result.root_labels      # coarse cluster
result.child_labels     # fine cluster (-1 if root-rejected)
result.rejected_        # True if rejected at either level
```

Persistence mirrors the hierarchy:

```
clusters/hierarchy/
  root/
    centroids.safetensors
    metadata.json
  children/
    0/
      centroids.safetensors
      metadata.json
    1/
      ...
  hierarchy.json
```

The implementation is pure Python orchestration over Rust-backed snapshots. Each individual assignment call drops into Rust with the GIL released, so you get the performance benefit of Rust for the expensive part (centroid distance computation) and the flexibility of Python for the routing logic.

On supplier embeddings (312K, K=20 root x 10 sub), hierarchical slotting boosted commodity purity from 0.782 to 0.837 and sub-commodity purity from 0.044 to 0.092. The combined configuration -- hierarchical + adaptive P10 at both levels -- achieved commodity purity of 0.846 on 50K kept items with 20% total rejection. The entire hierarchical snapshot is 240 KB.

## Real Numbers

I validated Cluster Slotting on two production-scale embedding datasets. Both use 80/20 train/test splits.

### CROSS Ruling Embeddings (tariff classification)

323K x 1536 embeddings of customs rulings, clustered into K=98 (one per HTS chapter), PCA to 128 dimensions.

- **Speedup**: Slotting 64K points takes 676ms vs. 56s for a full refit -- **113x faster**
- **Fidelity**: 99.86% of training points land back in their original cluster when re-assigned through the snapshot
- **Quality**: Slotted purity 0.580 vs. full refit 0.574 -- equivalent (the snapshot is as good as refitting)
- **Snapshot size**: 1.6 MB on disk vs. 1.5 GB of training data

### Supplier Embeddings (commodity classification)

312K x 1536 embeddings of supplier catalog items, K=20, matryoshka truncation to 128 dimensions.

- **Speedup**: Slotting 62K points takes 66ms vs. 30s for a full refit -- **453x faster**
- **Fidelity**: 99.94% training fidelity (153 mismatches out of 250K, max confidence on mismatches: 0.002)
- **Quality**: Slotted commodity purity 0.782 vs. full refit 0.781 -- identical
- **Snapshot size**: 21 KB (flat), 240 KB (hierarchical 20x10)

The fidelity numbers deserve a comment. When 99.94% of training points re-assign to their original cluster, the 0.06% that don't are exclusively boundary points with near-zero confidence. These are points that were ambiguous during training and remain ambiguous during assignment. The snapshot isn't losing information -- it's faithfully reproducing the uncertainty.

### Throughput

At the raw level, snapshot assignment sustains 3.5 million points per second at 100K points in 128 dimensions. For EmbeddingCluster -- including the full preprocessing pipeline (L2-norm, PCA projection, L2-norm, spherical assignment) -- throughput is 948K points per second. These are wall-clock numbers on a single machine, not theoretical peaks.

## What This Replaces

Here's the workflow most clustering libraries give you:

1. Load training data
2. `fit()` -- wait 30-60 seconds
3. Get labels
4. Discard the model (or pickle it, if you're brave)
5. New data arrives -- go to step 1

Here's the workflow with Cluster Slotting:

1. **Fit once** on your corpus
2. **Snapshot** the cluster topology
3. **Calibrate** with training data (optional, enables adaptive rejection and drift)
4. **Save** to disk -- KB, not GB
5. **Load anywhere** -- different process, different machine, no training data needed
6. **Assign** new items in milliseconds with confidence scores
7. **Reject** uncertain items for human review (adaptive, per-cluster)
8. **Monitor** for drift -- know when to refit
9. **Hierarchical** -- multi-level classification in a single `assign()` call

The snapshot becomes the artifact. Not the training data, not the model object, not a pickle file -- a 21 KB directory with two human-inspectable files that encodes everything you need to classify new data against your cluster topology.

## The API

```python
from rustcluster import KMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster, HierarchicalSnapshot

# Fit and snapshot
model = EmbeddingCluster(n_clusters=20, reduction_dim=128).fit(X_train)
snapshot = model.snapshot()
snapshot.calibrate(X_train)
snapshot.save("clusters/")

# Load and assign (different process, no training data)
snapshot = ClusterSnapshot.load("clusters/")
labels = snapshot.assign(X_new)

# With confidence and adaptive rejection
result = snapshot.assign_with_scores(
    X_new,
    adaptive_threshold=True,
    adaptive_percentile="p10"
)
# result.labels_        -> cluster indices, -1 for rejected
# result.confidences_   -> [0, 1) confidence
# result.rejected_      -> boolean mask

# Drift monitoring
report = snapshot.drift_report(X_recent)
# report.kappa_drift_          -> vMF concentration shift per cluster
# report.direction_drift_      -> centroid direction shift per cluster

# Hierarchical
hier = HierarchicalSnapshot.build(X_train, model, n_sub_clusters=10)
hier.save("clusters/hierarchy/")

hier = HierarchicalSnapshot.load("clusters/hierarchy/")
root_labels, child_labels = hier.assign(X_new)
```

## Architecture Notes

A few decisions worth explaining:

**f64-only snapshots.** Even if you trained with f32, the snapshot converts to f64. Centroid arithmetic in reduced dimensions is cheap, and f64 eliminates any precision concerns during assignment. The training data can be whatever dtype you want.

**`Arc<Vec<f64>>` centroids.** The Arc means multiple threads can read centroids simultaneously without copying. Rayon's parallel iterator hands each row its own slice of the work, and all threads share the same centroid array. Combined with GIL release, this means Python doesn't block during assignment.

**Confidence from a single pass.** The `assign_nearest_two_with()` function finds both the nearest and second-nearest centroid in one loop over the centroid array. No separate pass for the runner-up. The confidence computation adds zero extra cost to assignment.

**Hierarchical is pure Python.** The root/child routing logic is trivial -- a loop over cluster IDs with numpy boolean indexing. The expensive part (centroid distance computation) is in Rust. Putting the routing in Python means it's easy to extend (custom routing, asymmetric sub-cluster counts) without touching the Rust core.

**Calibration is optional and additive.** Version 1 snapshots (no calibration) work fine. Calling `calibrate()` bumps the version to 2 and adds the calibration fields. Loading a v1 snapshot sets all calibration fields to None. Nothing breaks, you just don't get adaptive thresholds or Mahalanobis boundaries until you calibrate.

## When to Use This

Cluster Slotting makes sense when:

- You have a stable cluster topology that new data should be assigned into
- You need sub-second assignment latency (API serving, streaming pipelines)
- You care about confidence -- not all assignments are equal
- You want to detect when the world has shifted enough to justify refitting
- You need a tiny deployment artifact (KB, not GB)

It does not make sense when:

- Your clusters change every batch (just refit)
- You have fewer than a few thousand points (refitting is already fast)
- You need non-centroid-based clustering (DBSCAN, HDBSCAN -- no centroids to snapshot)

## Getting Started

```bash
pip install rustcluster
```

rustcluster is a Rust+PyO3 library with a sklearn-compatible API. Six algorithms (KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Agglomerative, EmbeddingCluster), all compiled to native code. Cluster Slotting works with KMeans, MiniBatchKMeans, and EmbeddingCluster -- the three algorithms that produce centroids.

The source is on GitHub. The snapshot implementation lives in `src/snapshot.rs` (Rust core), `src/snapshot_io.rs` (persistence), `python/rustcluster/snapshot.py` (Python wrapper), and `python/rustcluster/hierarchical.py` (hierarchical orchestration).

---

*rustcluster is published on PyPI and supports Python 3.9+ on Linux, macOS, and Windows.*
