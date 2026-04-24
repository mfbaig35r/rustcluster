# Cluster Slotting v2: Fixing What Production Broke

The [first version of Cluster Slotting](blog-post-cluster-slotting.md) shipped with frozen centroid assignment, confidence scoring, rejection thresholds, persistence via safetensors, and drift detection. I validated it on 323K CROSS ruling embeddings (113x speedup, 99.86% fidelity) and 312K supplier embeddings on Databricks (453x speedup, 99.94% fidelity). The v1 blog post ended with a list of known limitations and a clear roadmap for v2.

Then I actually used it in production. The limitations weren't theoretical anymore.

## Four problems

### 1. Global confidence threshold rejected 98.5% of data

On the supplier dataset (312K embeddings, k=20), setting `confidence_threshold=0.30` rejected 98.5% of points -- 61,568 out of 62,477. Only 909 items survived. The v1 blog post showed this as the headline result: "94% commodity purity on the auto-classified items, flag the rest for review." What I glossed over was that "the rest" was almost everything.

The issue is straightforward. Confidence distributions vary dramatically across clusters. A tight, well-separated cluster produces high confidence for every member. A diffuse cluster sitting close to its neighbors produces low confidence even for points that are clearly in the right place. A single global threshold treats these identically, which means it's calibrated for neither.

I flagged this in the v1 post: "a single global threshold is a blunt instrument." Turns out "blunt" was generous. On this dataset, it was unusable.

### 2. Drift detection was broken for spherical metrics

v1's drift detection computed `fit_mean_distances` during training and used a 2x threshold heuristic at inference time. For spherical snapshots (cosine or dot product metrics), this showed 100% rejection rate for both in-distribution data and random noise. The heuristic was designed for Euclidean distance, where larger values mean farther away. Dot product similarity values don't behave like distances -- higher means closer -- and the 2x threshold logic was meaningless.

I'd called this out explicitly in v1: "drift detection calibration for spherical metrics is less well-studied than for Euclidean. The thresholds I use work empirically, but I don't have theoretical guarantees." That was too charitable. They didn't work empirically either. On spherical data, the drift detector couldn't distinguish a new batch of real supplier embeddings from Gaussian noise. That's not "less well-studied" -- that's broken.

### 3. Sub-commodity purity was 0.04 at k=20

The supplier pipeline clusters at the commodity level (k=20), then needs sub-clusters within each commodity. v1 snapshots only handle a single level. At k=20, sub-commodity purity was 0.044 -- barely above random. The original pipeline gets better results by doing hierarchical clustering: cluster into commodities first, then cluster each commodity's points into sub-commodities. But v1 snapshots had no mechanism for this.

### 4. Voronoi assumes spherical clusters

Nearest-centroid assignment (Voronoi tessellation) treats every cluster as having equal spread in all dimensions. Real clusters can be elongated -- high variance in some dimensions, low in others. A point might be closer to the wrong centroid in Euclidean space but obviously belong to a different cluster if you account for the cluster's shape. This was the theoretical concern. Whether it mattered on real embedding data was an open question.

## Four fixes

### Feature 1: Per-cluster adaptive thresholds

The fix starts with a new method: `calibrate(X_train)`. It runs assignment on the training data and computes per-cluster confidence quantiles -- P5, P10, P25, P50. At assignment time, `adaptive_threshold=True` rejects a point only if its confidence falls below the Nth percentile of its assigned cluster's training distribution.

```python
snap = model.snapshot()
snap.calibrate(X_train)
result = snap.assign_with_scores(X_new, adaptive_threshold=True, adaptive_percentile="p10")
```

The semantics of `adaptive_percentile="p10"` are: "reject this point if its confidence is lower than 90% of training points that were assigned to this same cluster." A diffuse cluster with naturally low confidence scores gets a low threshold. A tight cluster with high confidence scores gets a high threshold. Each cluster's rejection boundary is calibrated to its own distribution.

In Rust, `calibrate()` is a `&mut self` method on `ClusterSnapshot`. It calls `assign_batch()` internally, collects per-cluster confidence scores, sorts them, and computes quantiles via linear interpolation. A new `apply_adaptive_rejection()` method on `AssignmentResult` looks up the assigned cluster's threshold and rejects accordingly.

**Results on CROSS rulings (323K, k=98):**

| Method | Rejected | Purity |
|--------|----------|--------|
| Global 0.30 | 85.4% | 0.706 |
| Adaptive P10 | 10.1% | 0.594 |
| Adaptive P25 | 25.2% | 0.616 |
| Adaptive P50 | 49.8% | 0.658 |

**Results on supplier embeddings (312K, k=20):**

| Method | Rejected | Commodity Purity | Kept |
|--------|----------|-----------------|------|
| Global 0.30 | 98.5% | 0.940 | 909 |
| Adaptive P5 | 5.0% | 0.785 | 59,343 |
| Adaptive P10 | 10.1% | 0.788 | 56,159 |
| Adaptive P50 | 50.2% | 0.814 | 31,116 |

From 909 kept items to 56,159. The rejection rate went from "reject everything" to something you can actually tune. The purity at P10 (0.788) is close to the no-rejection baseline (0.782), which makes sense -- you're only cutting the bottom 10% of each cluster's confidence distribution, not the bottom 10% globally.

The feature works.

### Feature 2: vMF drift detection

For spherical snapshots, `calibrate()` estimates per-cluster von Mises-Fisher concentration parameter kappa from the mean resultant length R:

```
kappa = R * (d - R^2) / (1 - R^2)
```

This is the Banerjee et al. (2005) approximation for the vMF MLE. The mean resultant length R measures how tightly concentrated the data is around the mean direction -- R near 1 means all points cluster tightly on the hypersphere, R near 0 means they're spread uniformly.

The `drift_report()` method then computes kappa for new data and reports two metrics:

- `kappa_drift_`: per-cluster `(new_kappa - fit_kappa) / fit_kappa`. Negative means the new data is more dispersed than training data.
- `direction_drift_`: per-cluster `1 - dot(old_centroid, new_mean_direction)`. Higher means the cluster's center of mass has rotated on the hypersphere.

Non-spherical snapshots return `None` for both fields. Existing behavior is completely unchanged.

**Results on CROSS rulings:**

| Data | kappa_drift mean | direction_drift mean |
|------|-----------------|---------------------|
| In-distribution | 0.003 | 0.002 |
| Random noise | 408,000,000 | 0.21 |

Direction drift discrimination ratio: 124x.

**Results on supplier embeddings:**

| Data | kappa_drift mean | direction_drift mean |
|------|-----------------|---------------------|
| In-distribution | 0.001 | 0.0003 |
| Random noise | -0.82 | 0.43 |

Direction drift discrimination ratio: **1,268x**.

The v1 bug is completely fixed. In-distribution data produces drift values near zero. Random noise produces values orders of magnitude higher. The system can now clearly tell the difference between "new suppliers that look like existing suppliers" and "garbage embeddings."

The kappa_drift numbers are interesting: on CROSS rulings, random noise produces an enormous positive kappa_drift (the random embeddings are paradoxically concentrated in some clusters), while on supplier embeddings it's negative (more dispersed). Direction drift is the more reliable signal -- it's consistently high for random data and near zero for real data on both datasets.

### Feature 3: Diagonal Mahalanobis boundaries

`calibrate()` computes per-cluster per-dimension variance -- a diagonal covariance matrix, O(kd) storage instead of O(kd^2) for full covariance. Assignment with `boundary_mode="mahalanobis"` uses:

```
distance = sum((x_i - mu_i)^2 / var_i)
```

instead of standard Euclidean:

```
distance = sum((x_i - mu_i)^2)
```

Dimensions with high variance contribute less. Dimensions with low variance contribute more. A point far from the centroid along a high-variance dimension is penalized less than a point equally far along a low-variance dimension. A variance floor of 1e-12 prevents division by zero.

```python
snap.calibrate(X_train)
labels = snap.assign(X_new, boundary_mode="mahalanobis")
result = snap.assign_with_scores(X_new, boundary_mode="mahalanobis")
```

Implementation: a new `assign_nearest_two_mahalanobis()` function mirrors `assign_nearest_two_with()` but divides each dimension's squared difference by the stored variance. The variance computation is a standard two-pass algorithm (means, then squared deviations) with an unbiased estimator (n-1 denominator).

**Results on CROSS rulings:** Voronoi purity 0.580, Mahalanobis purity 0.566. Slightly worse. 83% agreement between methods.

**Results on supplier embeddings:** Voronoi purity 0.782, Mahalanobis purity 0.792. Slightly better. 85% agreement between methods.

The honest assessment: Mahalanobis is a wash on real embedding data. The disagreements are exclusively low-confidence boundary points with mean confidence around 0.03. After PCA projection and L2 normalization, embedding clusters are roughly spherical -- there isn't much anisotropy for diagonal variance to exploit.

This is the feature I'd call correctly opt-in. It exists, it's tested, it doesn't hurt anything, and it'll matter when someone has genuinely elongated clusters in non-normalized data. On the two embedding datasets I tested, it doesn't move the needle. I could have cut it, but the implementation cost was low (it piggybacks on the variance computation that `calibrate()` already does for its own purposes) and the theoretical case for it is sound.

### Feature 4: Hierarchical slotting

`HierarchicalSnapshot` chains a root snapshot with per-cluster child snapshots. Assignment cascades: assign to root first, then route to the child snapshot for the assigned root cluster. Rejection at root level short-circuits -- no child assignment is attempted.

```python
from rustcluster.experimental import HierarchicalSnapshot

hier = HierarchicalSnapshot.build(X_train, root_model, n_sub_clusters=10)
root_labels, child_labels = hier.assign(X_new)
hier.save("clusters/hierarchy/")

# Later
loaded = HierarchicalSnapshot.load("clusters/hierarchy/")
```

This is pure Python -- no Rust changes needed. The class wraps a root `ClusterSnapshot` and a dict of child `ClusterSnapshot` instances. `assign()` calls `root.assign()`, groups points by root label, then calls each child's `assign()` on its group. `save()`/`load()` uses a directory tree: `root/` + `children/0/`, `children/1/`, etc. + `hierarchy.json` metadata.

The `build()` convenience method fits an `EmbeddingCluster` per root cluster on its assigned data. It handles edge cases: clusters with too few points get skipped (no child snapshot created).

**Results on CROSS rulings (k_root=98, k_sub=10):**

| Metric | Flat | Hierarchical | Delta |
|--------|------|-------------|-------|
| Chapter purity | 0.580 | **0.706** | +0.126 |
| Heading purity | 0.370 | **0.536** | +0.165 |

**Results on supplier embeddings (k_root=20, k_sub=10):**

| Metric | Flat | Hierarchical | Delta |
|--------|------|-------------|-------|
| Commodity purity | 0.782 | **0.837** | +0.054 |
| Sub-commodity purity | 0.044 | **0.092** | +0.048 |

Sub-commodity purity doubled on both datasets. More importantly, the sub-clusters are meaningful. The cluster profile table from the supplier experiment tells the story: root cluster 10 (Components/Metal) splits into "Precision Machined Parts" (900 items), "Aluminum Extrusions" (468 items), "Metal Stampings" (723 items). Those are real categories that a procurement team would use. They're not just statistical artifacts.

**Combined result (hierarchical + adaptive P10):**

- 20% total rejection (10% at root, additional 10% at child level)
- Commodity purity: 0.846 on 49,924 kept items
- Total snapshot size: 240 KB
- Assignment speed: still sub-second on 62K points

That's a production supplier classification system in a quarter-megabyte file.

## Why `calibrate()` is one method

The most important design decision in v2 was making `calibrate(X_train)` the single entry point for all calibration data. One call computes three things:

1. Per-cluster confidence quantiles (for adaptive thresholds)
2. Per-cluster per-dimension variance (for Mahalanobis boundaries)
3. Per-cluster vMF kappa (for spherical drift detection)

I considered alternatives:

**Pass data to `snapshot(X)`.** This breaks when snapshotting from a deserialized model. If you load a snapshot from disk, you don't have a model to call `snapshot()` on. Calibration needs to work on loaded snapshots, not just fresh ones.

**Compute during `fit()`.** This would mean changing every algorithm's fit path to store additional statistics. Invasive, and it couples the calibration contract to the fitting contract. If I add a new algorithm, it would need to know about calibration during fit.

**Separate methods per feature -- `calibrate_thresholds()`, `calibrate_variances()`, `calibrate_drift()`.** Three passes over the data instead of one, and three separate API decisions for the user. "Did I calibrate for thresholds or just drift? Do I need to call all three?"

`calibrate()` is explicit, optional, backward compatible, and survives save/load. Uncalibrated snapshots behave identically to v1. All new fields are `Option<T>` with `None` default.

The one cost: calibration preprocesses data twice -- once for its own variance/kappa computation, once inside the `assign_batch()` call it makes internally. I considered fusing these but decided against it. `calibrate()` is a one-time setup call, not a hot path. Optimizing it would have meant breaking the clean separation between "compute assignments" and "compute statistics on those assignments." Not worth it.

## Backward compatibility

Snapshot version bumps from 1 to 2 when `calibrate()` is called. The loader accepts both:

- v1 snapshots load with all calibration fields as `None`
- v2 snapshots without calibration also work
- `adaptive_threshold=True` without calibration raises a clear `RuntimeError`

New JSON metadata fields use `#[serde(skip_serializing_if = "Option::is_none", default)]` -- v1 JSON files parse cleanly into v2 structs. Cluster variances are stored in safetensors as an additional `"cluster_variances"` tensor with shape `[k, d]`. Old safetensors files without this tensor load fine.

I tested the cross-version path explicitly: save with v1, load with v2, calibrate, save again, load again. No issues. The `Option` pattern in Rust makes this almost impossible to get wrong -- you can't accidentally access an uncalibrated field without matching on `None`.

## What worked, what didn't

Three out of four features delivered measurable value on real data.

**Adaptive thresholds** solved the exact problem they were built for. The supplier dataset went from 909 usable items with global thresholding to 56,159 with adaptive P10, at nearly the same purity. This is the difference between a feature that works in a blog post and one that works in production.

**vMF drift detection** went from broken (100% rejection on both real and random data) to a 1,268x discrimination ratio on the supplier dataset. The direction drift metric is the reliable signal -- kappa drift is noisier and dataset-dependent, but direction drift is consistently near zero for in-distribution data and consistently large for noise.

**Hierarchical slotting** doubled sub-commodity purity and produced meaningful cluster profiles. The combined configuration (hierarchical + adaptive P10) is the answer I'd actually deploy.

**Mahalanobis boundaries** were the honest miss. Not harmful, not broken, just marginal. On both real embedding datasets, it moved purity by about one percentage point in either direction. The explanation is clear: after PCA projection and L2 normalization, clusters don't have dramatic anisotropy. The feature is correctly positioned as opt-in. It will matter for someone working with raw, un-normalized feature vectors where clusters genuinely differ in shape. That wasn't my use case.

## Stats

- 1,392 lines across 7 files
- 208 Rust tests + 284 Python tests (20 new v2 tests), zero regressions
- Validated on two real datasets: 323K CROSS rulings, 312K supplier embeddings
- All features backward compatible with v1 snapshots
- Built in a single session, 11 commits collapsed to one

The v1 post ended with four items on the roadmap. v2 built all four. Three of them mattered. The combined result -- hierarchical slotting with adaptive rejection, a quarter-megabyte snapshot, sub-second assignment on 62K points -- is what I'd actually put behind an API.

The lesson is the same one from v1, just reinforced: the algorithm isn't the hard part. Nearest-centroid assignment is trivial. What makes it useful is everything around it -- the confidence calibration that adapts to each cluster's natural distribution, the drift detection that actually works on your metric space, the hierarchy that matches how your domain is structured, and the persistence format that keeps all of it in a file you can inspect with a text editor and a hex dump.
