# rustcluster.index v1.1 Performance Plan

## Goal

Close the gap to FAISS on the operations that v1 left slow: `search()` and
`range_search()`. The fused `similarity_graph()` already wins on the
production workload at d=128. This plan does **not** try to match FAISS at
d=1536 — that's a BLAS-vs-faer ceiling that needs a separate decision
(linking BLAS or hand-tuned kernels).

## Diagnosis

Bench at n=100k, d=128, nq=1000, k=10 (Apple Silicon M-series, threshold
where most thresholds emit zero edges so we measure the kernel, not the
output):

| Operation | Time |
|---|---|
| numpy `Q @ X.T` (parallel BLAS) | 44ms |
| numpy GEMM + per-row Python top-k | 699ms |
| **faiss `IndexFlatIP.search(k=10)`** | **22ms** ← fused |
| **rustcluster `IndexFlatIP.search(k=10)`** | **444ms** |
| rustcluster `similarity_graph()` (n×n compute) | 1,764ms — **parallel via rayon over tiles** |

Decomposition:
- Our GEMM is single-threaded (faer with `Par::Seq`). Roughly 8× off
  parallel BLAS in wall-clock — most of the 444ms.
- Our `materialize_topk` runs the per-query top-k in a sequential
  `for i in 0..nq` loop. Roughly 80–100ms of the 444ms.
- FAISS at 22ms beats parallel BLAS at 44ms, which means it's using a
  **fused** GEMM-with-top-k kernel that doesn't even materialize the score
  matrix.

`similarity_graph` does not have this problem because it parallelizes over
tile pairs via rayon — we have CPU parallelism, we just don't use it in
the other entry points.

## Predicted impact

For `search` at n=100k, d=128, nq=1000, current 444ms:

| Change | Predicted | Confidence |
|---|---|---|
| Parallel matmul (`Par::Rayon`) | ~80ms | high (matches numpy parallel BLAS) |
| + Parallel per-query top-k (rayon over queries) | ~60ms | high (top-k is independent across queries) |
| + Fused GEMM+top-k kernel | ~30ms | medium (matching FAISS requires care) |

The first two changes are **30 minutes of code** and should land us at
~3× off FAISS. The fused kernel is real engineering and is a separate
milestone.

## Scope

**In scope for v1.1:**
1. Configurable `Par` in `ip_batch` so it parallelizes the GEMM when called
   from non-rayon contexts (`search`, `range_search`) and stays sequential
   when called from `similarity_graph`'s already-parallel tile loop.
2. Parallel per-query loop in `materialize_topk` and `materialize_range`.
3. A small Criterion bench in `benches/` so we can detect regressions.
4. A pytest performance regression test (skipped by default, runnable via
   `pytest -m perf`) that asserts wall-clock budgets vs FAISS at known
   shapes.

**Out of scope for v1.1:**
- Fused GEMM+top-k kernel (next milestone if v1.1 isn't enough).
- Optional BLAS backend (Accelerate / OpenBLAS feature flag) — would close
  the d=1536 gap but breaks the "pip install just works" promise.
- Hand-tuned f32 GEMM tiles for Apple Silicon AMX / x86 AVX-512.
- Anything that changes the public API.

## Milestones

### M1: Parallelize the matmul (highest impact, lowest risk)

**Change:** `ip_batch` takes a `Par` parameter (or selects internally based
on `rayon::current_thread_index()`).

```rust
// src/index/kernel.rs
pub fn ip_batch(queries: ArrayView2<f32>, data: ArrayView2<f32>, par: Par) -> Array2<f32> {
    // ...
    matmul(out_mut, Accum::Replace, q_ref, x_ref.transpose(), 1.0, par);
}
```

Callers:
- `search`, `range_search` in `flat.rs` → pass `Par::Rayon`
- `similarity_graph` (already parallel over tiles) → pass `Par::Seq`

**Rationale for explicit param over `current_thread_index()` detection:**
explicit is auditable and avoids surprising behavior when users call
`search` from inside their own rayon pool. Hide the choice behind two
small helpers (`ip_batch_parallel`, `ip_batch_seq`) so call sites read
naturally.

**Expected outcome at n=100k, d=128, nq=1000:** 444ms → ~80ms (5-6×).

**Risk:** None to correctness — same compute, different scheduler.

**Effort:** 1 hour including tests.

### M2: Parallelize per-query top-k and range emit

**Change:** `materialize_topk` and `materialize_range` use `par_iter` over
queries. Top-k is trivially independent. Range needs per-query buffers
that get concatenated at the end (similar to `similarity_graph`'s tile
chunks pattern).

```rust
// materialize_topk
let results: Vec<(Vec<f32>, Vec<i64>)> = (0..nq)
    .into_par_iter()
    .map(|i| {
        let row = scores.row(i);
        let skip = self_position(opts.exclude_self, queries.row(i), data);
        topk(row.as_slice().unwrap(), k, direction, skip)
    })
    .collect();
// Then write results into Array2 sequentially.
```

For `materialize_range`, the per-query buffers must be merged in order to
preserve the `lims` CSR semantics. Trivially:

```rust
let chunks: Vec<(Vec<f32>, Vec<i64>)> = (0..nq).into_par_iter().map(...).collect();
let mut lims = Vec::with_capacity(nq + 1);
lims.push(0);
let mut distances = Vec::new();
let mut labels = Vec::new();
for (d, l) in chunks {
    distances.extend(d);
    labels.extend(l);
    lims.push(distances.len() as i64);
}
```

**Expected outcome on top of M1:** 80ms → ~60ms (additional 1.3×).

**Risk:** Range_search's `lims` array has order semantics — must merge
chunks in query order, not in completion order. Easily handled by
collecting into a `Vec` (rayon preserves order with `par_iter().collect()`).

**Effort:** 2 hours including tests.

### M3: Lock in the gains with benches and perf tests

**Criterion bench in `benches/index.rs`:**
```rust
fn bench_search_n100k_d128(c: &mut Criterion) {
    // Pre-build index once outside the benchmark loop.
    c.bench_function("flat_ip_search_100k_128_nq1000_k10", |b| {
        b.iter(|| index.search(queries.view(), 10, opts));
    });
}
```
Configurations matching the v1 bench table. Run via `cargo bench`.

**Perf-test in pytest:**
```python
@pytest.mark.perf  # skip by default; run with `pytest -m perf`
def test_search_within_3x_of_faiss():
    # n=100k, d=128, nq=1000, k=10
    # Assert rustcluster within 3× of FAISS wall-clock.
```

This gives us regression coverage without slowing down the default test
suite.

**Effort:** 2 hours.

## Verification plan

After each milestone:
1. Run `cargo test` and `pytest` — must stay green.
2. Run the bench script (`/tmp/rcbench.py` from the v1 walkthrough).
3. Compare against the predicted-impact table above; investigate if off
   by more than ~30%.
4. Update the speedup table in `docs/vector-indexing-v1-plan.md` so
   future readers see the v1.1 numbers, not the v1 ones.

## After v1.1

If 3× off FAISS isn't acceptable for some workload, the next options in
priority order:

1. **Fused GEMM + top-k kernel.** Avoid materializing the (nq, n) score
   matrix; compute distances and maintain a per-query heap inline. Real
   engineering — probably a week of focused work.
2. **Optional BLAS backend** behind a Cargo feature. Link Accelerate on
   macOS, OpenBLAS on Linux. Closes the d=1536 gap but breaks
   pip-install-just-works for the BLAS-enabled wheel.
3. **Hand-tuned f32 GEMM tile** for the small-tile shapes used by
   `similarity_graph` at d≥384. Highly architecture-specific.

These don't need to be decided now — measure v1.1 first.
