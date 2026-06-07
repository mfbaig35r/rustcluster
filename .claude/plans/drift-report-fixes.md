# Plan — fix `ClusterSnapshot::drift_report` bugs

Bug doc: `docs/snapshot-drift-bugs.md`.

Two correctness bugs:

1. `relative_drift_` is NaN for KMeans / MiniBatchKMeans snapshots because
   `from_kmeans*` / `from_minibatch_kmeans*` initialize
   `fit_mean_distances: vec![0.0; k]` and nothing ever populates it.
2. `rejection_rate_` is meaningless: pinned to `0.0` for KMeans
   (downstream of bug 1) and pinned to `1.0` for `EmbeddingCluster`
   (the `2.0 × fit_global_mean` heuristic was designed for Euclidean
   distances but is applied to cosine similarities in spherical mode —
   every similarity ≤ 1.0 falls below `2.0 × 0.7 ≈ 1.4`, so every point
   is "rejected").

## Decisions (locked in before implementation)

| Question | Decision | Rationale |
|---|---|---|
| Bug 1 fix shape | Populate `fit_mean_distances` at fit time on `KMeansState` / `MiniBatchKMeansState`. Factories copy from state. | Alternative (`calibrate()` backfill) silently shifts the field's semantic to "calibration-time mean" and makes the field's meaning depend on whether `calibrate()` was called. Computing it at fit time is honest and free — the existing fit loops already touch every per-point distance. |
| Bug 2 threshold strategy | Per-cluster `mean ± 3·std` from calibration data. Spherical inverts the comparison (low similarity = outlier). Stored in a new `Option<ClusterDistanceStats>` on `ClusterSnapshot`, populated by `calibrate()`. | Per-cluster (not global) lets it work when clusters have very different tightness. Calibration is the existing distributional fingerprint — no need for a parallel mechanism. |
| `rejection_rate` API when uncalibrated | Return `f64::NAN`. `__repr__` prints `rejection_rate=N/A`. | Non-breaking — keeps `report.rejection_rate_: float` in Python. Matches `relative_drift`'s existing NaN convention for under-defined entries. `Option<f64>` is cleaner but breaks every PyPI consumer doing `report.rejection_rate_ * 100`, and v0.6.x doesn't need a breaking change for this. |
| MiniBatchKMeans extra pass | None needed. The fit already does a final full-data pass to compute labels + inertia. Extend that same loop. | Zero performance regression. |
| Lloyd extra pass | None needed. Lloyd already computes inertia from the last-iter `assignments` (label, dist) tuples. Compute `fit_mean_distances` from the same data — stale-by-1 convention is what Lloyd already does for inertia. | Consistent with existing inertia semantic. |
| Hamerly extra pass | None needed. Already has a final per-point distance pass after the loop. Extend it. | Zero performance regression. |
| `fit_mean_distances` rename | Keep the name. Document the dual semantic (Euclidean distance for kmeans-family, cosine similarity for embedding). The `spherical` field already signals units. | Lowest churn. Bigger correctness win is bug 1 itself. |
| Version bump | Stay v0.6.x. | No breaking API change. |

## Ordered implementation (5 commits)

### Step 1 — Failing tests pin the contract
Add to `tests/test_snapshot.py::TestDriftReport`:
- `test_kmeans_relative_drift_finite` — fits KMeans, snapshots, checks `np.all(np.isfinite(report.relative_drift_))`. Fails on main (bug 1).
- `test_rejection_rate_low_for_in_distribution` — fits KMeans, **calls `snap.calibrate(X)`**, asserts `report.rejection_rate_ < 0.10` on the fit data. Fails on main (bug 2 case A — currently 0.0 regardless; passes by accident, but after step 3 we want it to still pass).
- `test_rejection_rate_high_for_shifted` — same setup, shift data by `+50.0`, assert `report.rejection_rate_ > 0.50`. Fails on main (bug 2 — pinned to 0.0).
- `test_embedding_rejection_rate_low_for_in_distribution` — fits `EmbeddingCluster`, calibrates, asserts `< 0.10`. Fails on main (bug 2 case B — pinned to 1.0).
- `test_rejection_rate_nan_uncalibrated` — fits KMeans, **does not** calibrate, asserts `np.isnan(report.rejection_rate_)`. Pins the new "uncalibrated → NaN" semantic.

Verify they fail on main *before* moving to step 2 (run once, confirm the failure modes, then proceed).

### Step 2 — Bug 1: populate `fit_mean_distances` at fit time
- `src/kmeans.rs`: add `pub fit_mean_distances: Vec<f64>` to `KMeansState<F>`. In `run_lloyd_iterations` (`src/kmeans.rs:248`), in the loop body that already iterates `(label, dist)` from `assignments` (lines 276-279) to accumulate `inertia`, also accumulate `cluster_dist_sums[label] += dist` and `cluster_counts[label] += 1`. At the end, divide. Store in returned state.
- `src/hamerly.rs`: extend the final inertia pass at `src/hamerly.rs:158-167` to also accumulate per-cluster sums and counts, then derive `fit_mean_distances`. Set on returned `KMeansState`.
- `src/minibatch_kmeans.rs`: add `pub fit_mean_distances: Vec<f64>` to `MiniBatchKMeansState<F>`. In the final-pass loop at `src/minibatch_kmeans.rs:237-242` (already iterating final_assignments), accumulate per-cluster sums + counts. Divide. Store.
- `src/snapshot.rs`: replace `vec![0.0; k]` at lines 136, 162, 190, 219 with `state.fit_mean_distances.clone()`. For the f32 factories (lines 162, 219), the field is already `Vec<f64>` (we'll store it as f64 in state for both f32 and f64 variants to avoid a conversion step at snapshot time).
- Verify `test_kmeans_relative_drift_finite` now passes.

### Step 3 — Calibration stats infrastructure
- `src/snapshot.rs`: add
  ```rust
  #[derive(Debug, Clone)]
  pub struct ClusterDistanceStats {
      pub mean: Vec<f64>,
      pub std: Vec<f64>,
  }
  ```
  and add `pub fit_distance_stats: Option<ClusterDistanceStats>` to `ClusterSnapshot`. Initialize to `None` in all five factories.
- Extend `calibrate()` (`src/snapshot.rs:704`): the existing loop already aggregates per-cluster means/variances over `work_data`. The simpler thing is to additionally aggregate per-cluster sum and sum-of-squares over `result.distances[i]` (the assignment distance/similarity), then derive `mean[c]` and `std[c]` per cluster. Store in `self.fit_distance_stats`.
- `src/snapshot_io.rs`:
  - Add `fit_distance_mean: Option<Vec<f64>>` and `fit_distance_std: Option<Vec<f64>>` to `SnapshotMetadata` with `#[serde(skip_serializing_if = "Option::is_none", default)]` (matches the existing v2 calibration fields' pattern).
  - Populate in `save_snapshot` from `snapshot.fit_distance_stats`.
  - Reconstruct in `load_snapshot` into `Some(ClusterDistanceStats { mean, std })` when both fields are present, else `None`.
  - Update the two test fixture metadata blobs at `src/snapshot_io.rs:313` and `:340` if they're constructed as struct literals (verify — they may use `..Default::default()` already).

### Step 4 — Bug 2: rewire `rejection_rate`
- `src/snapshot.rs`: replace the rejection block at `src/snapshot.rs:599-617` with:
  ```rust
  let rejection_rate = match self.fit_distance_stats.as_ref() {
      Some(stats) if n > 0 => {
          let k_sigma = 3.0_f64;
          let mut rejected = 0usize;
          for i in 0..n {
              let label = result.labels[i];
              if label < 0 { continue; }
              let c = label as usize;
              if c >= self.k { continue; }
              let mean = stats.mean[c];
              let std = stats.std[c].max(1e-12);
              let d = result.distances[i];
              let is_out = if self.spherical {
                  d < mean - k_sigma * std       // low similarity = outlier
              } else {
                  d > mean + k_sigma * std       // high distance = outlier
              };
              if is_out { rejected += 1; }
          }
          rejected as f64 / n as f64
      }
      _ => f64::NAN,
  };
  ```
  Delete the `2.0 * fit_global_mean` heuristic entirely.
- `src/lib.rs`: the `rejection_rate_` getter (`src/lib.rs:2373-2376`) already returns `f64`, no change needed — NaN passes through to Python as `float('nan')`.
- `src/lib.rs:2399-2406`: update `__repr__` — when `rejection_rate.is_nan()`, print `rejection_rate=N/A` instead of `0.00%`.
- Verify the four bug-2 tests now pass.

### Step 5 — Docs + sanity
- Update doc comment on `ClusterSnapshot::fit_mean_distances` to explain the dual semantic.
- Update doc comment on `DriftReport::rejection_rate` to explain "NaN when snapshot has not been calibrated".
- Add `debug_assert!(self.fit_mean_distances.len() == self.k, "factory forgot to populate fit_mean_distances")` to `drift_report` to catch future regressions.
- Run `cargo test`, `cargo clippy --all-targets`, `maturin develop`, `pytest tests/test_snapshot.py -v`.

## Risks

- **`EmbeddingCluster` `relative_drift_` regression** — Step 2 touches only the four KMeans-family factories. `from_embedding_cluster` (`src/snapshot.rs:266`) keeps populating from `intra_similarity`. Confirm `tests/test_snapshot.py::TestEmbeddingClusterSnapshot` still green.
- **Snapshot on-disk format** — Step 3 adds two optional fields. Backward-compat via `serde default`. Forward-compat fine since `deny_unknown_fields` is not set. Confirmed by reading `src/snapshot_io.rs:22-52`.
- **In-tree tests that construct `KMeansState` literals** — none found in initial grep, but `cargo test` in step 5 will fail if any exist with the new field. Trivial fix (add `fit_mean_distances: vec![]`).
- **`DriftReport.__repr__` string change** — `rejection_rate=N/A` instead of `0.00%` when uncalibrated. Anyone parsing the repr (bad idea) will break.
- **`fit_mean_distances` semantic change for serialized v1 snapshots on disk** — old snapshots were saved with `vec![0.0; k]` for kmeans-family. They'll continue to load that way and `relative_drift_` will be NaN on them until re-fit. This is consistent with what users see today, so no regression.

## Critical files

- `src/kmeans.rs` — `KMeansState`, Lloyd loop
- `src/hamerly.rs` — Hamerly final pass
- `src/minibatch_kmeans.rs` — `MiniBatchKMeansState`, final pass
- `src/snapshot.rs` — `ClusterSnapshot`, factories, `drift_report`, `calibrate`
- `src/snapshot_io.rs` — `SnapshotMetadata`, save/load
- `src/lib.rs` — PyO3 `DriftReport.__repr__`
- `tests/test_snapshot.py` — `TestDriftReport`
