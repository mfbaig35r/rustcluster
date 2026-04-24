# Cluster Slotting v2 Build Log

Built in a single session. Four features, 11 commits collapsed into one, 1,392 lines added across 7 files.

## What we built

### 1. Per-Cluster Adaptive Thresholds

**Problem**: v1's global `confidence_threshold=0.30` rejected 98.5% of supplier embeddings. A single threshold can't handle clusters with different natural spreads — tight clusters produce high confidence for everyone, diffuse clusters produce low confidence even for correct assignments.

**Solution**: `calibrate(X_train)` runs assignment on training data and computes per-cluster confidence quantiles (P5, P10, P25, P50). At assignment time, `adaptive_threshold=True` rejects a point only if its confidence falls below its assigned cluster's P10 — meaning "this point is less confident than 90% of training points in this cluster."

```python
snap = model.snapshot()
snap.calibrate(X_train)
result = snap.assign_with_scores(X_new, adaptive_threshold=True, adaptive_percentile="p10")
```

### 2. vMF Drift Detection for Spherical Metrics

**Problem**: v1's drift detection used a 2x distance threshold heuristic. For spherical metrics (cosine/dot product), this produced 100% rejection rate for both in-distribution AND random data. The heuristic was designed for Euclidean distance and is meaningless for similarity values.

**Solution**: For spherical snapshots, `calibrate()` estimates per-cluster von Mises-Fisher concentration parameter kappa from the mean resultant length: `kappa = R * (d - R^2) / (1 - R^2)`. `drift_report()` then computes kappa for new data and reports the shift.

Two new fields on `DriftReport`:
- `kappa_drift_`: per-cluster `(new_kappa - fit_kappa) / fit_kappa`. Negative = more dispersed.
- `direction_drift_`: per-cluster `1 - dot(old_centroid, new_mean_direction)`. Higher = centroid has rotated.

Validated: in-distribution data shows kappa_drift ≈ 0 and direction_drift ≈ 0. Random embeddings show kappa_drift < -0.1 and direction_drift > 0.1. Non-spherical snapshots return None for both fields.

### 3. Diagonal Mahalanobis Boundaries

**Problem**: Voronoi (nearest centroid by Euclidean distance) assumes all clusters are spherical with equal spread. Some clusters are elongated — high variance in some dimensions, low in others.

**Solution**: `calibrate()` computes per-cluster per-dimension variance (diagonal covariance, O(kd) not O(kd^2)). Assignment with `boundary_mode="mahalanobis"` divides each dimension's squared difference by the cluster's variance for that dimension, with a floor at 1e-12 to prevent division by zero.

```python
snap.calibrate(X_train)
labels = snap.assign(X_new, boundary_mode="mahalanobis")
result = snap.assign_with_scores(X_new, boundary_mode="mahalanobis")
```

Voronoi remains the default. Mahalanobis is opt-in and requires calibration.

### 4. Hierarchical Slotting

**Problem**: The supplier pipeline clusters at commodity level (k=20), then sub-clusters within each commodity. v1 snapshots only handle one level. Sub-commodity purity at k=20 was 0.04.

**Solution**: `HierarchicalSnapshot` chains a root snapshot with per-cluster child snapshots. Pure Python — no Rust changes needed. Assignment cascades: root first, then route to the child snapshot for the assigned root cluster. Rejection at the root level short-circuits (no child assignment attempted).

```python
from rustcluster.experimental import HierarchicalSnapshot

hier = HierarchicalSnapshot.build(X_train, root_model, n_sub_clusters=5)
root_labels, child_labels = hier.assign(X_new)

hier.save("clusters/hierarchy/")
loaded = HierarchicalSnapshot.load("clusters/hierarchy/")
```

Directory structure: `root/` + `children/0/`, `children/1/`, etc. + `hierarchy.json`.

## Architecture

### The `calibrate()` design

The key decision: `calibrate(X_train)` is a single method that computes everything in one pass over the data:
- Per-cluster confidence quantiles (P5/P10/P25/P50) for adaptive thresholds
- Per-cluster per-dimension variance for Mahalanobis boundaries
- Per-cluster vMF kappa for spherical drift detection

This was chosen over alternatives:
- Passing data to `snapshot(X)` — breaks when snapshotting from a deserialized model
- Computing during `fit()` — invasive, changes every algorithm's fit path
- `calibrate()` — backward compatible, explicit, optional, survives save/load

### Backward compatibility

All new fields are `Option<T>`, defaulting to `None`. The snapshot version field bumps from 1 to 2 when `calibrate()` is called. The loader accepts both v1 and v2:
- v1 snapshots load with all calibration fields as None
- v2 snapshots without calibration also work (calibration is optional)
- `adaptive_threshold=True` without calibration raises a clear RuntimeError

### Serialization

Calibration metadata (confidence quantiles, kappa, resultant lengths) goes in the JSON sidecar via `#[serde(skip_serializing_if = "Option::is_none", default)]`. Cluster variances go in the safetensors file as a `"cluster_variances"` tensor with shape `[k, d]`.

## Files changed

| File | Changes |
|------|---------|
| `src/snapshot.rs` | +608 lines: `ClusterConfidenceStats`, `ClusterVariances`, `calibrate()`, `apply_adaptive_rejection()`, `assign_batch_mahalanobis()`, vMF kappa in `drift_report()` |
| `src/snapshot_io.rs` | +142 lines: v2 metadata fields, cluster_variances tensor, v1 backward compat |
| `src/lib.rs` | +78 lines: `calibrate()` PyO3 binding, `adaptive_threshold`/`boundary_mode` params, `kappa_drift_`/`direction_drift_` getters |
| `python/rustcluster/snapshot.py` | +41 lines: `calibrate()`, `is_calibrated`, `boundary_mode` param |
| `python/rustcluster/hierarchical.py` | 263 lines (new): `HierarchicalSnapshot`, `HierarchicalAssignmentResult` |
| `python/rustcluster/experimental.py` | +3 lines: export `HierarchicalSnapshot` |
| `tests/test_snapshot_v2.py` | 270 lines (new): 20 tests across all 4 features |

## Test results

- **208 Rust tests** — all pass (11 new snapshot tests)
- **284 Python tests** — all pass (20 new v2 tests)
- **Zero regressions** on v1 tests

## What didn't change

- `assign()` and `assign_with_scores()` without new parameters behave identically to v1
- v1 snapshots load and work without modification
- No changes to `fit()` on any algorithm
- No new Rust dependencies
- HierarchicalSnapshot is pure Python orchestration — Rust handles the heavy lifting in individual ClusterSnapshot instances
