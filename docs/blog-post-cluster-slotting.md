# Building Cluster Memory: Making Clustering Work Beyond `fit()`

Most clustering libraries assume your data holds still. You call `fit()`, get labels, and move on. But data in production doesn't hold still. New suppliers show up. New product descriptions appear. New embeddings get generated daily. And the question is always the same: which cluster does this new point belong to?

I built Cluster Slotting for rustcluster to answer that question properly -- with confidence scoring, rejection thresholds, persistence, and drift detection. This post covers the design process, the architecture decisions, the bugs, and the real-world benchmarks.

## The problem

rustcluster ships six clustering algorithms: KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Agglomerative, and EmbeddingCluster (a purpose-built pipeline that L2-normalizes, PCA-projects via faer, and runs spherical K-means). It's a Rust+PyO3 library, sklearn-compatible, published on PyPI. The clustering is fast. But it was stateless.

In two production use cases -- supplier classification (312K embeddings) and tariff ruling categorization (323K embeddings) -- I needed to assign new data to existing clusters continuously. The naive solution is to re-cluster every time. That's expensive and unnecessary. If you have 300K embeddings clustered into 98 groups, and 500 new embeddings arrive, you shouldn't redo 26 seconds of compute. You should slot them into the existing structure in milliseconds.

No major Python clustering library ships this as a first-class feature with the three things you actually need in production: confidence scores (how sure are we?), rejection (what should a human review?), and persistence (save and load the cluster topology without the training data).

## Starting from overengineered requirements

I started with a requirements document that I'd put together after researching incremental clustering patterns, production vector systems, and taxonomy stability problems. The core concept was right, but the spec was overengineered. It proposed:

- Explicit hyperplane boundaries between clusters
- A classifier-based boundary mode (train a model to approximate the boundaries)
- A hybrid assignment mode combining centroid distance with boundary proximity
- Soft membership (probability of belonging to multiple clusters)
- von Mises-Fisher mixture snapshots
- A vague `boundary_strictness` parameter

I did a deep research pass to validate against production implementations: FAISS IVF, Milvus, Weaviate HFresh, sklearn's `predict()`, HDBSCAN's `approximate_predict`, serialization formats, and drift detection methods. The research was useful for filling blind spots -- especially around PCA persistence subtleties, vMF distributions for spherical data, and multi-signal drift detection.

But the real value came from synthesizing the research against the requirements and cutting scope.

### What I killed and why

**Hyperplane boundaries.** For centroid-based algorithms, the Voronoi tessellation IS the decision boundary. Computing explicit hyperplanes between every pair of centroids gives you mathematically identical assignments at O(k^2) cost instead of O(k). The research confirmed this: every production vector system (FAISS, Milvus, Weaviate) uses frozen centroid assignment. Nobody builds explicit boundaries.

**Classifier-based boundaries.** This is model distillation -- training a second model to approximate the first. It's a legitimate technique, but it's a different feature entirely. Scope creep.

**Hybrid mode.** Once I understood that Voronoi IS the boundary, the "hybrid" mode collapsed to nearest-centroid assignment plus a confidence score derived from the gap between the two nearest centroids. One mode, not three.

**Soft membership.** Useful for topic modeling. Not useful for supplier classification. Deferred to v2.

**vMF mixture snapshots.** The research correctly pointed to von Mises-Fisher distributions as the right model for spherical clusters. But storing full mixture parameters when we just need assignment is premature. I store a concentration proxy (intra-cluster similarity) instead, enough for drift detection without the complexity.

I replaced the vague `boundary_strictness` with two concrete, measurable parameters: `distance_threshold` and `confidence_threshold`. One controls absolute proximity. One controls relative decisiveness.

## Architecture

### The ClusterSnapshot struct

The central type is `ClusterSnapshot` in Rust:

```rust
pub struct ClusterSnapshot {
    pub algorithm: SnapshotAlgorithm,
    pub metric: Metric,
    pub spherical: bool,
    pub centroids: Arc<Vec<f64>>,  // flat row-major
    pub k: usize,
    pub d: usize,
    pub input_dim: usize,
    pub preprocessing: Preprocessing,
    pub fit_mean_distances: Vec<f64>,
    pub fit_cluster_sizes: Vec<usize>,
    pub fit_n_samples: usize,
    pub version: u32,
}
```

A few deliberate choices here:

**`Arc<Vec<f64>>` for centroids.** Flat row-major storage. The `Arc` makes it `Send+Sync` for free, so we can release the GIL and run parallel assignment with rayon. For a 98-cluster, 128-dimensional snapshot, the centroid matrix is 98 * 128 * 8 = 100 KB. Negligible.

**f64-only snapshots.** Even when the model was fit on f32 data, the snapshot stores f64 centroids. Centroid storage is tiny -- there's no reason to double all the assignment code for dtype dispatch. The f32 factory constructor just widens on creation.

**Preprocessing baked in.** This was the single most important design decision. The `Preprocessing` enum is:

```rust
pub enum Preprocessing {
    None,
    L2Normalize,
    EmbeddingPipeline {
        input_dim: usize,
        pca: PcaProjection,
    },
}
```

For EmbeddingCluster, the snapshot encodes the entire preprocessing pipeline: L2-normalize the raw embeddings, PCA-project them, L2-normalize again, then do spherical assignment. Users pass raw embeddings and the snapshot handles everything. Without this, users would silently get wrong results -- passing unnormalized data to a spherical snapshot, or forgetting to PCA-project, or normalizing before PCA instead of after. I've seen every one of these mistakes in other libraries where preprocessing is the caller's responsibility.

**Scoped to three algorithms.** KMeans, MiniBatchKMeans, and EmbeddingCluster only. DBSCAN, HDBSCAN, and Agglomerative don't produce centroids -- their cluster definitions are fundamentally different (density-connected components, hierarchical merges). I didn't fake it. HDBSCAN has `approximate_predict` in the original library, which I may add later, but it's a different mechanism.

### Confidence scoring

I needed a scalar confidence score in [0, 1) that answers: "How decisively did this point get assigned?"

The answer reuses machinery I already had. The `assign_nearest_two_with<F, D>()` function from `utils.rs` finds the nearest AND second-nearest centroid in a single pass. Confidence falls out naturally:

For distance metrics (lower is better):
```
confidence = 1 - (nearest_distance / second_nearest_distance)
```

For dot product (higher is better):
```
confidence = 1 - (second_best_similarity / best_similarity)
```

This gives 0.0 when the point is equidistant between two centroids (pure boundary -- no confidence) and approaches 1.0 when the nearest centroid is much closer than any alternative.

For k=1, confidence is always 0.0. There's no meaningful confidence with only one cluster -- there's nowhere else the point could go. This seems obvious in retrospect, but I hit a bug during implementation: the single-cluster case was returning confidence 1.0. The second distance was infinity, so `1 - (best / infinity) = 1 - 0 = 1`. Fixed by special-casing k < 2 to return 0.0.

I validated the scoring with a boundary walkthrough experiment: start a point at centroid 0 (x=0), walk it step by step toward centroid 1 (x=10). Confidence drops perfectly symmetrically -- 1.0 at the centroids, smooth decline to exactly 0.0 at the midpoint (x=5). No discontinuities, no edge cases. The math is clean.

### Persistence format

The snapshot serializes to a directory with two files:

```
clusters/
  centroids.safetensors   # float arrays (centroids, PCA components, PCA mean)
  metadata.json           # everything else
```

**safetensors** for the numeric arrays -- zero-copy deserialization, no arbitrary code execution risk, good Rust and Python interop. This was a no-brainer. The research surveyed ONNX, pickle, protobuf, and custom binary. Pickle is a non-starter for anything that crosses a trust boundary (arbitrary code execution). ONNX is overkill. safetensors is the right tool.

**JSON** for the metadata sidecar -- human-readable, inspectable, versionable. You can `cat metadata.json` and immediately understand what you're looking at: algorithm, metric, k, dimensions, preprocessing type, fit-time statistics.

A 98-cluster, 128-dimensional EmbeddingCluster snapshot with full PCA persistence is 1.6 MB. A 20-cluster, 128-dimensional KMeans snapshot is 21 KB. Compare that to shipping the training data (1.5 GB of embeddings for the tariff case).

### Drift detection

The `DriftReport` computes how new data compares to what the model saw at fit time:

- Per-cluster mean distances in the new data
- Per-cluster relative drift: `(new_mean - fit_mean) / fit_mean`
- Global mean distance
- Rejection rate (fraction of points exceeding 2x the fit-time global mean)

The research recommended a multi-signal approach -- requiring at least two alarm conditions before triggering a refit. A single metric can spike from benign distributional shifts; correlated signals are more reliable.

One honest limitation: drift detection calibration for spherical metrics (cosine similarity) is less well-studied than for Euclidean. The thresholds I use work empirically, but I don't have theoretical guarantees for them. This is a v2 problem.

## Implementation

I built Cluster Slotting in 8 sequential, self-contained commits:

1. Dependencies (`safetensors`, `serde`, `serde_json`) + error types
2. Core `ClusterSnapshot` struct + `AssignmentResult` + assignment logic with tests
3. Snapshot I/O (save/load with safetensors + JSON)
4. Factory constructors (`from_kmeans`, `from_minibatch_kmeans`, `from_embedding_cluster`)
5. PyO3 bindings + `.snapshot()` methods on all three algorithms
6. `DriftReport` struct + `drift_report` method
7. Python wrapper class (`snapshot.py`) + integration into existing model classes
8. Integration tests

Total: roughly 3,000 lines across 14 files. 197 Rust tests, 264 Python tests, zero regressions.

### Bugs worth mentioning

**The k=1 confidence bug.** Already described above. The fix is two lines but the lesson is general: edge cases in distance/similarity ratios show up as infinities and NaNs before they show up as wrong answers. Every ratio needs a denominator guard.

**PcaProjection derives.** The `Preprocessing` enum needs `Debug + Clone`. The inner `PcaProjection` struct didn't have those derives because it was only used by value before. Adding `#[derive(Debug, Clone)]` is trivial, but the compiler error pointed at the wrong location (the enum definition, not the missing derives on the struct). Took longer to diagnose than to fix.

**Circular import in Python.** The `snapshot.py` module needed `_prepare_array` from `__init__.py`, which imported `snapshot.py`. Classic. I fixed it by inlining `_prepare_array` directly in `snapshot.py` -- it's five lines, not worth the import gymnastics.

## The API

After fitting any supported model:

```python
from rustcluster import KMeans, EmbeddingCluster
from rustcluster.snapshot import ClusterSnapshot

# Fit
model = EmbeddingCluster(n_clusters=98, target_dim=128)
model.fit(X_train)

# Snapshot
snapshot = model.snapshot()
snapshot.save("clusters/")

# Later, in a different process, without the training data
snapshot = ClusterSnapshot.load("clusters/")

# Simple assignment
labels = snapshot.assign(X_new)

# Assignment with confidence and rejection
result = snapshot.assign_with_scores(
    X_new,
    confidence_threshold=0.3,
)
# result.labels_       -> int64 array (-1 for rejected)
# result.distances_    -> float64 array
# result.confidences_  -> float64 array, [0, 1)
# result.rejected_     -> bool array

# Drift monitoring
report = snapshot.drift_report(X_recent)
# report.relative_drift_       -> per-cluster drift
# report.global_mean_distance_ -> aggregate signal
# report.rejection_rate_       -> fraction that would be rejected
```

## Benchmarks

All numbers are real, from the actual experiments. Not cherry-picked.

### Synthetic data

| Scenario | Speedup | Throughput |
|---|---|---|
| KMeans (k=10, d=32) | 142x | 6.6M pts/sec |
| EmbeddingCluster (k=20, d=256->32) | 93x | 980K pts/sec |
| Scaling test (100K points) | -- | 3.5M pts/sec sustained |

The EmbeddingCluster number is lower because it includes L2-normalize + PCA-project + L2-normalize before assignment. The preprocessing is the bottleneck, not the assignment itself.

### CROSS ruling embeddings (323K x 1536, real data)

This is a tariff classification dataset -- 323K customs ruling embeddings at 1536 dimensions, clustered into 98 groups (matching HTS chapter count), PCA-compressed to 128 dimensions.

- **Fit time:** 56.1 seconds
- **Slot 64K new points:** 676 milliseconds (113x speedup)
- **Slot purity:** 0.580 vs full refit 0.574 -- equivalent (slotting slightly higher, within noise)
- **Training fidelity:** 99.86% -- only 369 out of 259K points got different labels than full refit, all with confidence below 0.04
- **Snapshot size:** 1.6 MB vs 1.5 GB of training data

The rejection sweep is where it gets interesting:

| Confidence threshold | Purity | Coverage |
|---|---|---|
| None | 0.580 | 100% |
| 0.10 | 0.630 | ~85% |
| 0.30 | 0.710 | ~60% |
| 0.50 | 0.860 | ~30% |

### Supplier embeddings on Databricks (312K x 1536, real data)

Production supplier classification. Matryoshka truncation to 128 dimensions.

- **Fit time:** 26.7 seconds
- **Slot 62K new points:** 65.9 milliseconds (453x speedup)
- **Slot commodity purity:** 0.782 vs full refit 0.781 -- identical to three decimal places
- **Training fidelity:** 99.94% -- 153 mismatches, maximum confidence 0.002
- **Throughput:** 948K pts/sec
- **Snapshot size:** 21 KB

The rejection sweep on suppliers:

| Confidence threshold | Purity |
|---|---|
| None | 0.782 |
| 0.10 | 0.820 |
| 0.20 | 0.870 |
| 0.30 | 0.940 |

At a confidence threshold of 0.30, you get 94% commodity purity on the auto-classified items, and the rest get flagged for human review. That's the actual value proposition.

### Separation sweep

I ran a synthetic separation sweep varying Gaussian noise from 0.1 to 3.0 on well-separated clusters. Accuracy holds stable at 74-75% for noise up to 1.0, then degrades gracefully. The 10th-percentile confidence tracks the degradation cleanly -- dropping from 0.10 to 0.01 as noise increases. This means you can use the confidence distribution itself as a signal for when your clusters are losing coherence.

## What didn't work, or isn't done

**Drift detection calibration.** The spherical drift thresholds are empirical. I don't have a formal statistical test for "has this cluster drifted significantly?" on cosine similarity. The Euclidean case maps to standard hypothesis testing; the spherical case needs vMF concentration parameter estimation, which I deferred.

**Sub-commodity purity.** The 0.78 commodity purity on supplier data is solid for top-level categories. But within-cluster sub-commodity purity is lower -- the clusters are too coarse for fine-grained classification. This points toward hierarchical slotting (cluster at commodity level, then sub-cluster), not a limitation of the slotting mechanism itself.

**Confidence distribution skew.** In practice, confidence scores are heavily right-skewed. Most points have high confidence; the interesting ones are in the long left tail. This means a single global threshold is a blunt instrument. Per-cluster adaptive thresholds would be better, but that's additional complexity.

**HDBSCAN support.** The original library has `approximate_predict`, which uses a different mechanism (mutual reachability distance to the nearest core point). It's not centroid-based, so it doesn't fit the current architecture. Worth adding, but needs its own design.

## What's next

The research pointed clearly at a few v2 directions:

- **Diagonal Mahalanobis distance** for anisotropic clusters -- clusters that are elongated in some dimensions. This is the right upgrade over Voronoi for non-spherical cluster shapes.
- **Hierarchical slotting** -- slot to commodity first, then sub-commodity within. Cascading snapshots.
- **vMF concentration parameters** stored in snapshot metadata, enabling proper statistical drift tests for spherical data.
- **Per-cluster adaptive thresholds** -- different clusters have different natural spreads; the rejection threshold should reflect that.

## The takeaway

The algorithm here isn't novel. Nearest-centroid assignment has been in textbooks for decades. Every production vector database does it internally. The contribution is packaging it as a first-class library feature with the things that make clustering usable in production: confidence scoring that tells you how certain the assignment is, rejection that lets you route uncertain items to human review, persistence that decouples the snapshot from the training data, and drift detection that tells you when to refit.

The rejection sweep is the real differentiator -- not the 453x speedup. Being able to say "auto-classify these 312K suppliers at 94% purity, flag the rest for review" is what turns clustering from an exploratory tool into a production system.

The speedup is nice. The confidence is what makes it useful.
