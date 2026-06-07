# Snapshot drift_report — two correctness bugs

`ClusterSnapshot::drift_report` returns two output fields that are unreliable
across all currently-supported snapshot algorithms:

- `relative_drift` — NaN on every cluster for KMeans / MiniBatchKMeans snapshots
- `rejection_rate` — pinned to `0.0` for KMeans / MiniBatchKMeans snapshots and
  pinned to `1.0` for `EmbeddingCluster` snapshots

Both fields are exposed via PyO3 (`src/lib.rs:2364`, `src/lib.rs:2374`) and
shown in the `DriftReport.__repr__` (`src/lib.rs:2401`), so downstream users
see them as authoritative output.

## Bug 1 — `relative_drift` is NaN for non-embedding snapshots

### Where

- Factories: `src/snapshot.rs:136, 162, 190, 219`
- Drift math: `src/snapshot.rs:580-594`
- Calibration (does not backfill): `src/snapshot.rs:704-812`

### Root cause

`from_kmeans`, `from_kmeans_f32`, `from_minibatch_kmeans`, and
`from_minibatch_kmeans_f32` all initialize:

```rust
fit_mean_distances: vec![0.0; k],
```

The drift calculation in `drift_report` then computes:

```rust
let fit_mean = self.fit_mean_distances[c];     // = 0.0
let new_mean = new_mean_distances[c];          // typically > 0
if fit_mean.abs() < 1e-30 {
    if new_mean.abs() < 1e-30 { 0.0 } else { f64::NAN }
} else {
    (new_mean - fit_mean) / fit_mean.abs()
}
```

Because `fit_mean` is always `0.0` for KMeans-family snapshots, every cluster
returns `NaN` whenever the new data has any points assigned to it.

`calibrate()` populates `confidence_stats`, `cluster_variances`, and
`fit_kappa`, but **does not** backfill `fit_mean_distances`. So calling
`calibrate()` first does not fix the issue.

`EmbeddingCluster.snapshot()` is unaffected — `from_embedding_cluster`
(`src/snapshot.rs:266`) populates `fit_mean_distances` from
`intra_similarity` produced by `evaluation::intra_cluster_similarity` during
fit.

### Reproduction

```python
import numpy as np
from rustcluster import KMeans

rng = np.random.default_rng(42)
X = rng.standard_normal((200, 8)).astype(np.float64)

model = KMeans(n_clusters=4, random_state=42).fit(X)
snap = model.snapshot()
report = snap.drift_report(X)
print(report.relative_drift_)   # array of NaN, length k
```

### Impact

- Any `KMeans` or `MiniBatchKMeans` user relying on `relative_drift_` sees NaN
  and has no signal.
- The same Python API works correctly for `EmbeddingCluster`, which makes the
  bug easy to miss in cross-algorithm code.

### Why tests didn't catch it

`tests/test_snapshot.py::TestDriftReport` asserts only:

- `report.n_samples_`
- `report.global_mean_distance_`
- `report.new_cluster_sizes_`

It never reads `report.relative_drift_`. `tests/test_snapshot_v2.py` asserts
`kappa_drift_` and `direction_drift_` but also skips `relative_drift_`.

### Fix direction

Two options, both straightforward:

1. **Populate `fit_mean_distances` at snapshot construction** — for KMeans
   factories, compute per-cluster mean squared (or Euclidean) distance from
   `state.labels` + `state.centroids` + the training data. This requires
   passing the training data into the snapshot factory or computing the
   per-cluster mean during fit and storing it on `KMeansState`. The latter is
   cleaner since the same loop already iterates the data for inertia.

2. **Have `calibrate()` backfill `fit_mean_distances`** from calibration data.
   Cheaper to implement (no API change), but it silently shifts the meaning of
   the field from "fit-time" to "calibration-time" and makes the field's
   semantics depend on whether `calibrate()` was called.

Option 1 is more honest about what the field represents.

## Bug 2 — `rejection_rate` is meaningless for every snapshot algorithm

### Where

- `src/snapshot.rs:600-617`

### Root cause

The relevant block:

```rust
let fit_global_mean = if self.fit_n_samples > 0 {
    self.fit_mean_distances.iter().sum::<f64>() / k as f64
} else {
    0.0
};
let rejection_rate = if fit_global_mean.abs() > 1e-30 && n > 0 {
    let threshold = 2.0 * fit_global_mean;
    let rejected = if self.spherical {
        result.distances.iter().filter(|&&d| d < threshold).count()
    } else {
        result.distances.iter().filter(|&&d| d > threshold).count()
    };
    rejected as f64 / n as f64
} else {
    0.0
};
```

Two separate failure modes, depending on the snapshot's algorithm:

**Case A — KMeans / MiniBatchKMeans snapshots**

`fit_mean_distances` is the all-zero vector (see Bug 1). So
`fit_global_mean = 0`, the outer guard `fit_global_mean.abs() > 1e-30` is
false, and `rejection_rate` falls into the `else` branch and is always `0.0`,
regardless of how badly the new data has drifted.

**Case B — `EmbeddingCluster` snapshots (spherical)**

`fit_mean_distances` holds per-cluster mean cosine similarity, typically
`~0.5–0.9`. So `fit_global_mean ≈ 0.7` and `threshold = 2.0 * 0.7 = 1.4`. But
in spherical mode `result.distances` contains cosine similarities, which are
bounded above by `1.0`. The spherical filter `|&&d| d < threshold` therefore
matches **every** point, and `rejection_rate` is always `1.0`.

The threshold heuristic `2.0 * fit_global_mean` was presumably designed for
Euclidean distances (where doubling the typical fit-time distance is a
reasonable outlier cutoff), but it does not transfer to the cosine-similarity
units used in the spherical branch.

### Reproduction

```python
import numpy as np
from rustcluster import KMeans
from rustcluster.experimental import EmbeddingCluster

rng = np.random.default_rng(42)

# Case A — KMeans
X = rng.standard_normal((200, 8))
snap = KMeans(n_clusters=4, random_state=42).fit(X).snapshot()
report = snap.drift_report(X + 1000.0)  # massively shifted
print(report.rejection_rate_)            # 0.0

# Case B — EmbeddingCluster
E = rng.standard_normal((200, 64)).astype(np.float32)
snap_e = EmbeddingCluster(n_clusters=5, reduction_dim=16).fit(E).snapshot()
report_e = snap_e.drift_report(E)         # *same* data as fit
print(report_e.rejection_rate_)           # 1.0
```

### Impact

- The number printed in `DriftReport.__repr__` (`src/lib.rs:2401`,
  `"rejection_rate={:.2}%"`) is always `0.00%` or `100.00%` and conveys no
  information about the data.
- For embedding-clustering users (the documented headline use case for
  snapshots), the field is actively misleading — it suggests catastrophic
  drift on in-distribution data.

### Why tests didn't catch it

No test in `tests/test_snapshot.py` or `tests/test_snapshot_v2.py` reads
`report.rejection_rate_`. The two pinned outputs (`0.0` for KMeans, `1.0` for
EmbeddingCluster) are deterministic, so even a single sanity test (e.g.,
"rejection_rate on the same data used to fit should be near zero") would have
flagged the bug immediately.

### Fix direction

The right cutoff is mode-dependent and should not use a hardcoded `2.0×`
multiplier:

- **Euclidean / Manhattan** — a cutoff like `mean + 3 * std` of the fit-time
  per-point distance distribution, or a configurable percentile of the
  fit-time distance distribution stored at snapshot time.
- **Spherical** — invert the comparison. The right "outlier" criterion is
  *low* similarity relative to the cluster's typical similarity, e.g.
  `sim < fit_mean_sim - 3 * fit_std_sim`, or a configurable lower percentile.

Both branches need per-cluster (not global) thresholds for the rate to be
meaningful when clusters have very different tightness — which is exactly what
`calibrate()` already collects in `confidence_stats`. A clean fix would
deprecate the heuristic and route `rejection_rate` through the calibration
data when it exists, returning `None` (or an explicit "not calibrated" status)
when it does not.

## Suggested minimum-viable tests

Add to `tests/test_snapshot.py::TestDriftReport`:

```python
def test_kmeans_relative_drift_finite(self, well_separated_data):
    m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
    snap = m.snapshot()
    report = snap.drift_report(well_separated_data)
    rd = np.asarray(report.relative_drift_)
    assert np.all(np.isfinite(rd)), f"relative_drift has non-finite entries: {rd}"

def test_rejection_rate_low_for_in_distribution(self, well_separated_data):
    m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
    snap = m.snapshot()
    report = snap.drift_report(well_separated_data)
    assert report.rejection_rate_ < 0.10

def test_rejection_rate_high_for_shifted(self, well_separated_data):
    m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
    snap = m.snapshot()
    report = snap.drift_report(well_separated_data + 50.0)
    assert report.rejection_rate_ > 0.50

def test_embedding_rejection_rate_low_for_in_distribution(self):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 64)).astype(np.float32)
    snap = EmbeddingCluster(n_clusters=5, reduction_dim=16).fit(X).snapshot()
    report = snap.drift_report(X)
    assert report.rejection_rate_ < 0.10
```

Each of these tests currently fails on `main` for at least one of the two bugs
above; together they pin down the contract that any fix should preserve.

## Severity summary

| Field | KMeans / MiniBatchKMeans snapshot | EmbeddingCluster snapshot |
|---|---|---|
| `relative_drift_` | **broken** (always NaN) | works |
| `rejection_rate_` | **broken** (always 0.0) | **broken** (always 1.0) |
| `global_mean_distance_` | works | works |
| `new_cluster_sizes_` | works | works |
| `new_mean_distances_` | works | works |
| `kappa_drift_` / `direction_drift_` | N/A | works (requires `calibrate()`) |
