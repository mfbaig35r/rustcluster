# Cluster Slotting v2 Requirements

**rustcluster 0.6.0** | April 2026

## Background

Cluster Slotting v1 shipped in rustcluster 0.5.0 with `ClusterSnapshot` — frozen centroid assignment with confidence scoring, rejection, persistence via safetensors, and drift detection. It was validated on two production datasets: 323K CROSS rulings (113x speedup, 99.86% fidelity) and 312K supplier embeddings on Databricks (453x speedup, 99.94% fidelity).

The architecture works. The performance works. But v1 exposed four concrete limitations that blocked or degraded real use cases. This document specifies four features that address them.

---

## Feature 1: Per-Cluster Adaptive Thresholds

### What

Replace the single global `confidence_threshold` with per-cluster thresholds derived from the training data's confidence distribution.

### Why

On the supplier dataset (k=20), setting `confidence_threshold=0.30` rejected 98.5% of points. The problem isn't the metric — it's that confidence distributions vary dramatically across clusters. Tight clusters produce high confidence for all members; diffuse clusters produce low confidence even for correct assignments. A global threshold either rejects nearly everything (too strict for diffuse clusters) or accepts nearly everything (too lenient for tight clusters).

### Technical approach

At snapshot creation time, compute per-cluster confidence statistics from training assignments:

- For each cluster: mean, standard deviation, and quantiles (P5, P10, P25, P50) of confidence scores
- Store as `fit_confidence_stats: Vec<ClusterConfidenceStats>` in the snapshot
- At assignment time, reject a point if its confidence falls below a configurable percentile of its assigned cluster's training distribution — for example, below P10 means "this point is less confident than 90% of training points in this cluster"
- The global threshold remains available as a fallback; adaptive thresholds are opt-in

The key dependency is that confidence statistics require the training data at snapshot time. Currently `from_kmeans()` receives only a `KMeansState` (centroids and labels), not the original data. See the shared `calibrate()` approach below.

### API

```python
snap = model.snapshot()
snap.calibrate(X_train)  # computes per-cluster confidence stats

result = snap.assign_with_scores(
    X_new,
    adaptive_threshold=True,       # use per-cluster P10 by default
    adaptive_percentile=10,        # configurable: reject below this percentile
)

# inspect the per-cluster thresholds
stats = snap.confidence_stats  # list of dicts: {cluster, mean, std, p5, p10, p25, p50}
```

### What changes in the codebase

- **`snapshot.rs`**: Add `fit_confidence_stats: Option<Vec<ClusterConfidenceStats>>` to `ClusterSnapshot`. New struct `ClusterConfidenceStats { mean: f64, std: f64, p5: f64, p10: f64, p25: f64, p50: f64 }`.
- **`snapshot.rs`**: New method `calibrate()` that accepts training data, runs assignment, computes per-cluster confidence quantiles, and populates `fit_confidence_stats`.
- **`snapshot.rs`**: Extend `assign_with_scores()` — when adaptive mode is active, compare each point's confidence against its assigned cluster's percentile threshold instead of the global threshold.
- **`snapshot_io.rs`**: Serialize `fit_confidence_stats` in the JSON sidecar. Old snapshots load with `None` (backward compatible).
- **`python/rustcluster/snapshot.py`**: Expose `calibrate()`, `confidence_stats` property, and `adaptive_threshold`/`adaptive_percentile` parameters on `assign_with_scores()`.

### Acceptance criteria

- On the supplier dataset (k=20), adaptive thresholds with P10 reject approximately 10-30% of points (not 98.5%).
- Rejected points have measurably lower assignment quality (wrong cluster) compared to accepted points.
- Per-cluster thresholds are accessible via `snap.confidence_stats` and visible in `metadata.json`.
- Snapshots without calibration behave identically to v1 (global threshold only).
- Round-trip: `save()` then `load()` preserves calibration data.

---

## Feature 2: Hierarchical Slotting

### What

A `HierarchicalSnapshot` that chains multiple `ClusterSnapshot` instances in a cascade: assign at level 0, then route to a level-1 snapshot within the assigned cluster.

### Why

The supplier classification pipeline clusters at commodity level (k=20), then sub-clusters within each commodity (k=2-25 per commodity). v1 snapshot slotting only handles a single level. Sub-commodity purity on the supplier data was 0.04 at k=20 — the commodity-level clusters are too coarse for sub-commodity classification. The original pipeline achieves usable sub-commodity purity by doing hierarchical clustering. Snapshot slotting needs to support this pattern.

### Technical approach

- `HierarchicalSnapshot` is a Python-only class (no new Rust types in v1 of this feature) that wraps a tree of `ClusterSnapshot` instances.
- Level 0: one snapshot covering the full embedding space (e.g., commodity clusters, k=20).
- Level 1: one snapshot per level-0 cluster, each trained on the subset of data assigned to that cluster.
- Assignment cascades: assign to level 0 first, then route to the level-1 snapshot keyed by the level-0 label.
- Rejection at any level short-circuits: if a point is rejected at level 0, it does not proceed to level 1.
- Persistence uses a directory tree: `hierarchy/level_0.safetensors`, `hierarchy/level_1/cluster_0.safetensors`, etc.

### API

```python
from rustcluster import HierarchicalSnapshot, ClusterSnapshot

# Build from fitted models
commodity_snap = commodity_model.snapshot()
sub_snaps = {
    cluster_id: sub_model.snapshot()
    for cluster_id, sub_model in sub_models.items()
}

hier = HierarchicalSnapshot(root=commodity_snap, children=sub_snaps)
hier.save("clusters/hierarchy/")

# Load and assign
hier = HierarchicalSnapshot.load("clusters/hierarchy/")
labels = hier.assign(X_new)
# labels: list of tuples [(commodity_label, sub_label), ...]

result = hier.assign_with_scores(X_new, confidence_threshold=0.2)
# result.labels_: list of (commodity_label, sub_label)
# result.confidences_: list of (commodity_confidence, sub_confidence)
# result.rejected_: True if rejected at either level
```

### What changes in the codebase

- **`python/rustcluster/snapshot.py`**: New `HierarchicalSnapshot` class with `__init__(root, children)`, `assign()`, `assign_with_scores()`, `save()`, `load()`.
- **`python/rustcluster/__init__.py`**: Export `HierarchicalSnapshot`.
- No Rust changes. Hierarchy is orchestration logic — the inner assignment calls use existing Rust-backed `ClusterSnapshot` instances.

### Acceptance criteria

- Reproduce the supplier pipeline's two-level clustering: assign to commodity, then to sub-commodity, via a single `hier.assign()` call.
- Sub-commodity purity via hierarchical slotting matches what the original pipeline achieves (significantly above the 0.04 from flat k=20).
- Rejection at level 0 prevents level-1 assignment (short-circuit).
- `save()` and `load()` round-trip correctly. A loaded hierarchy produces identical assignments to the original.
- Level-1 snapshots are optional — a hierarchy with only a root snapshot degrades to a regular `ClusterSnapshot`.

---

## Feature 3: vMF-Based Drift Detection for Spherical Metrics

### What

Replace the distance-based drift heuristic with von Mises-Fisher concentration parameter (kappa) comparison for spherical (cosine/dot product) snapshots.

### Why

The current `drift_report()` uses `fit_mean_distances` and a 2x threshold heuristic. On CROSS rulings (spherical metric), this produced a 100% rejection rate for both in-distribution data and random embeddings. The bug: `intra_similarity` values (dot products in the 0.3-0.6 range) don't translate to distance-style thresholds. The heuristic was designed for Euclidean distance and is meaningless for spherical metrics.

### Technical approach

For spherical snapshots, store per-cluster vMF concentration parameters (kappa) computed at fit time. Kappa measures how tightly a cluster's points concentrate around the centroid on the unit hypersphere. Higher kappa = tighter cluster.

- **Fit time**: estimate kappa per cluster from the mean resultant length (R-bar). `EmbeddingCluster` already computes `resultant_lengths` (passed in as `intra_similarity`). The MLE for kappa in d dimensions: `kappa = R_bar * (d - R_bar^2) / (1 - R_bar^2)`.
- **Drift detection**: compute R-bar for new data per cluster, estimate kappa_new, compare to kappa_fit. A cluster has drifted if `|kappa_new - kappa_fit| / kappa_fit` exceeds a threshold. Additionally, check centroid direction shift: dot product between old and new mean direction vectors. A direction shift below a threshold (e.g., 0.95) signals drift.
- **Aggregation**: global drift flag is raised if a configurable fraction of clusters show drift.
- Non-spherical snapshots continue using the existing distance-based heuristic unchanged.

### API

```python
snap = embedding_model.snapshot()
report = snap.drift_report(X_new)

# Existing fields (unchanged for non-spherical)
report.mean_distances           # per-cluster mean distances
report.rejection_rate           # fraction that would be rejected

# New fields (spherical snapshots only)
report.fit_concentrations       # per-cluster kappa at fit time
report.new_concentrations       # per-cluster kappa for new data
report.concentration_drift      # per-cluster |kappa_new - kappa_fit| / kappa_fit
report.centroid_direction_drift # per-cluster dot product of old vs new mean direction
report.drifted_clusters         # list of cluster IDs flagged as drifted
```

### What changes in the codebase

- **`snapshot.rs`**: Add `fit_concentrations: Option<Vec<f64>>` to `ClusterSnapshot`. Populated from `intra_similarity` via the kappa MLE formula in `from_embedding_cluster()`.
- **`snapshot.rs`**: Extend `drift_report()` — for spherical snapshots, compute kappa_new from new data's mean resultant lengths, compare to `fit_concentrations`. Compute centroid direction drift. Populate new fields on `DriftReport`.
- **`snapshot.rs`**: Add new fields to `DriftReport`: `concentration_drift: Option<Vec<f64>>`, `centroid_direction_drift: Option<Vec<f64>>`, `drifted_clusters: Option<Vec<usize>>`.
- **`snapshot_io.rs`**: Serialize `fit_concentrations` in the JSON sidecar. Backward compatible — old snapshots have `None`.
- **Python bindings**: Expose new `DriftReport` fields.

### Acceptance criteria

- On CROSS rulings: in-distribution data shows low concentration drift (<0.2); random embeddings show high drift (>1.0). This is the direct fix for the v1 bug where both showed 100% rejection.
- Kappa values are inspectable in `metadata.json` after save.
- Non-spherical snapshots produce `None` for all new fields — existing behavior is unchanged.
- Centroid direction drift correctly detects when a cluster's center has rotated on the hypersphere.

---

## Feature 4: Diagonal Mahalanobis Boundaries

### What

An optional assignment mode that uses per-cluster per-dimension variance (diagonal Mahalanobis distance) instead of raw Euclidean distance, accounting for clusters with different shapes and spreads.

### Why

Voronoi assignment (nearest centroid by Euclidean distance) assumes all clusters are spherical with equal spread. In practice, some clusters are elongated — high variance in some embedding dimensions, low in others. A point equidistant from two centroids in Euclidean space may clearly belong to the cluster whose high-variance axis it lies along. This is most relevant for non-spherical metrics (Euclidean, Manhattan) where cluster shapes are not constrained by normalization.

### Technical approach

- Store per-cluster diagonal variance: a flat `Vec<f64>` of shape `(k, d)` — per-dimension variance, not full covariance. Storage is O(kd), not O(kd^2).
- Mahalanobis distance: `d_mahal^2 = sum((x_i - mu_i)^2 / sigma_i^2)` instead of `d_eucl^2 = sum((x_i - mu_i)^2)`.
- Assignment mode is controlled by `boundary_mode`: `"voronoi"` (default, backward compatible) or `"mahalanobis"`.
- Confidence scoring uses the same top-2 margin ratio formula, applied to Mahalanobis distances instead of Euclidean.
- Variance computation requires training data at snapshot time. Uses the same `calibrate()` mechanism as adaptive thresholds.

### API

```python
snap = model.snapshot()
snap.calibrate(X_train)  # computes per-cluster variances (and confidence stats)

# Use Mahalanobis for assignment
labels = snap.assign(X_new, boundary_mode="mahalanobis")

result = snap.assign_with_scores(
    X_new,
    boundary_mode="mahalanobis",
    confidence_threshold=0.3,
)
```

### What changes in the codebase

- **`snapshot.rs`**: Add `fit_variances: Option<Vec<f64>>` and `boundary_mode: BoundaryMode` (enum: `Voronoi`, `Mahalanobis`) to `ClusterSnapshot`.
- **`snapshot.rs`**: New function `assign_nearest_two_mahalanobis()` — identical structure to `assign_nearest_two_with()` but divides each dimension's squared difference by the stored variance. Minimum variance floor to avoid division by near-zero (e.g., `max(sigma_i^2, 1e-10)`).
- **`snapshot.rs`**: Extend `calibrate()` — compute per-cluster per-dimension variance from training assignments and store in `fit_variances`.
- **`snapshot_io.rs`**: Serialize variances in safetensors as tensor `"variances"` with shape `[k, d]`. Serialize `boundary_mode` in JSON sidecar.
- **`utils.rs`**: The new assignment function may live here alongside `assign_nearest_two_with`.
- **Python bindings**: Expose `boundary_mode` parameter on `assign()` and `assign_with_scores()`.

### Acceptance criteria

- On synthetic anisotropic clusters (10:1 axis ratio), Mahalanobis assignment accuracy exceeds Voronoi by at least 5 percentage points.
- On real data (CROSS/supplier), results are comparable or slightly better — Mahalanobis is not expected to be transformative when clusters are roughly spherical.
- Voronoi remains the default. Existing code without `boundary_mode` is unchanged.
- Error if `boundary_mode="mahalanobis"` is requested but `calibrate()` was not called.
- Storage overhead is O(kd) additional floats in safetensors. For k=100, d=768, that's 600KB — negligible.

---

## Shared Dependency: Training Data at Snapshot Time

Features 1 and 4 both require training data that is not currently available in the snapshot creation path. `from_kmeans()` receives only `KMeansState` (centroids + labels), not the original data matrix.

### Recommended approach: `calibrate(X_train)`

A two-step API where snapshot creation works without data (v1 behavior) and calibration is an explicit optional step:

```python
snap = model.snapshot()        # works today, no data needed
snap.calibrate(X_train)        # optional: adds per-cluster stats, variances, concentrations
snap.save("clusters/")         # persists everything including calibration data
```

**Why this approach over alternatives:**

- **Pass data to `snapshot(X)`**: Clean, but fails when snapshotting from a deserialized model where training data is unavailable.
- **Compute during `fit()` and store in state**: Most efficient, but changes the fit path for every algorithm. Invasive.
- **`calibrate()`**: Backward compatible, doesn't change `fit()`, makes the dependency on training data explicit. Calibration survives `save()`/`load()`.

In Rust, `calibrate()` runs assignment on `X_train`, groups points by cluster, and computes: confidence quantiles (feature 1), per-dimension variances (feature 4), and kappa from mean resultant length (feature 3, though this is already available from `intra_similarity`). One pass over the data produces all calibration statistics.

---

## Priority Order

1. **Per-cluster adaptive thresholds** — highest impact. Directly fixes the 98.5% rejection rate that makes confidence-based rejection unusable on real data.
2. **Hierarchical slotting** — unlocks the supplier sub-commodity pipeline, the primary multi-level use case.
3. **vMF drift detection** — fixes a known bug. Without this, drift monitoring is broken for all spherical (embedding) snapshots.
4. **Diagonal Mahalanobis** — lowest priority. Largest implementation effort, smallest expected impact on the datasets tested so far. Most valuable for future use cases with genuinely anisotropic clusters.

## Estimated Effort

| Feature | Rust | Python | Tests | Size |
|---|---|---|---|---|
| Adaptive thresholds | ~200 lines | ~50 lines | ~15 tests | Medium |
| Hierarchical slotting | 0 (Python-only) | ~200 lines | ~15 tests | Medium |
| vMF drift detection | ~100 lines | ~20 lines | ~10 tests | Small |
| Diagonal Mahalanobis | ~300 lines | ~30 lines | ~15 tests | Large |

All four features are additive — no breaking changes to v1 APIs. Snapshots saved by v1 load and work identically in v2. New fields default to `None`/absent when not calibrated.
