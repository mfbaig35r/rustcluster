# Plan: Optimize how rustcluster.index calls faer

## Goal

Close some of the d=1536 gap to FAISS without taking on the
distribution complexity of a BLAS backend. Specifically: reduce
overhead in how we call faer's `matmul` from `ip_batch` and
`similarity_graph`, so more of the wall-clock time is actual
floating-point work rather than allocation, packing, and setup.

This is the **"low-cost middle path"** between Option 1 (tile-tuning,
already shipped) and Option 2 (BLAS backend, being researched).
Realistic ambition: **5-15% improvement** at d=1536. If the actual
gain after profiling looks lower than that, the plan is to abandon
the work and let the BLAS-backend research drive the next decision.

## Hypothesis (to be verified by profiling)

The bench shows our `similarity_graph` at d=1536 is ~30% behind
FAISS. Three sources of overhead inside our current `ip_batch`:

1. **Per-tile output allocation.** Every call to `ip_batch` does
   `vec![0.0; nq * n]`. In `similarity_graph` that's a fresh
   `tile × tile` allocation per tile pair. At n=20k, d=1536,
   tile=384, that's ~1,431 tiles × ~590 KB = **~840 MB of
   allocation churn** over a single similarity_graph run.
2. **Per-tile packing.** faer / `gemm` pack input matrices into
   cache-friendly panels before the microkernel runs. Each input
   strip gets packed every time it appears — and each strip
   appears in O(n_tiles) tile pairs. At d=1536, packing a
   `tile × d` strip is ~2.3 MB of memcpy. We do it twice per
   tile (xi and xj). Over a full run that's **gigabytes of
   redundant pack work**.
3. **GEMM setup overhead.** Each `matmul` call has fixed-cost
   thread coordination + dispatch. With 1,431 small calls, this
   adds up — though it's the smallest of the three.

Numbers above are upper bounds. Actual cost depends on what faer's
allocator and packing implementation are doing under the hood; we
need to profile before committing.

## Phase 1: Measurement (mandatory before any code change)

**Tools:** `cargo flamegraph` for the hot path, `samply` or
`Instruments.app` for allocation profiling on macOS.

**Scenarios to profile:**
- `similarity_graph` at n=20k, d=1536, threshold=0.5 (the
  production-shaped workload).
- `search` at n=100k, d=128, nq=1000, k=10 (the operation that
  benefited most from M1+M2).

**What to look for:**
- Top of the flamegraph — is faer's `matmul` itself hot, or are
  we spending time around it?
- Allocator weight in the profile (`malloc` / `__rust_alloc`).
- Time spent in faer's packing functions vs the SIMD microkernel.

**Stop criteria for Phase 1:**
- If faer's microkernel is >85% of the time, this plan is
  pointless — the gap is fundamentally faer-vs-BLAS and only the
  BLAS backend can close it. Abandon and wait for the BLAS
  research.
- If the microkernel is <70% (i.e., overhead is significant),
  proceed to Phase 2.

**Effort:** half a day.

## Phase 2: Easy wins (if Phase 1 justifies them)

### 2a. Output buffer reuse

Today every `ip_batch` call allocates a fresh output Vec. In the
similarity_graph hot loop, allocate one max-size buffer per rayon
worker and reuse it across tiles.

```rust
// Pseudocode — sketch only.
let buffers: Vec<Mutex<Vec<f32>>> = (0..rayon::current_num_threads())
    .map(|_| Mutex::new(Vec::with_capacity(max_tile * max_tile)))
    .collect();

pairs.par_iter().for_each_init(
    || buffers[rayon::current_thread_index().unwrap_or(0)].lock().unwrap(),
    |buf, &(i0, j0)| {
        buf.clear();
        buf.resize(h * w, 0.0);
        // Use buf as the matmul output; emit edges; release.
    }
);
```

**Expected gain:** 5-10% at d=1536 if Phase 1 confirms allocator
churn is meaningful. Could be 0% if the allocator is already
caching tile-shaped allocations efficiently (jemalloc-style
behavior).

**Risk:** low. The semantics don't change.

**Effort:** 1 day.

### 2b. Direct `gemm` calls (skip faer's wrapper)

`faer::linalg::matmul::matmul` is a thin wrapper around the
underlying `gemm` crate. We could call `gemm::gemm()` directly
with the same inputs. Saves one layer of indirection and lets
us pass parameters faer doesn't expose (e.g., explicit
microkernel hints for f32 / our specific shape).

**Expected gain:** 0-5%. If faer's wrapper does anything load-
bearing (trait dispatch, dtype checks), it's tiny. Mostly
worth doing if we ALSO want lower-level control for §2c.

**Risk:** medium. We'd be bypassing faer's invariants — easy to
get the leading-dimension or stride wrong and miscompute.
Existing parity tests would catch this.

**Effort:** 2-3 days.

## Phase 3: The interesting one — pre-pack the database matrix

This is where the real win lives, IF `gemm` exposes panel-packing
as an external operation. Status uncertain; needs investigation.

### Research first

`gemm` crate API audit:
- Does it expose `gemm_pack_a` / `gemm_pack_b` as public functions?
- Or are they pub(crate)?
- If hidden, is there an issue tracker or RFC requesting them be
  exposed? (Other consumers of `gemm` would also want this.)
- Failing that, is there a fork / copy of the relevant kernels we
  could pull in?

### If exposed

Pre-pack the entire `data` matrix once at index construction (or
on first call to `similarity_graph`). Each tile then calls a
"GEMM with one pre-packed operand" path that skips re-packing.

**Expected gain at d=1536:** 10-20% — packing cost is roughly
proportional to GEMM cost at our shape, and we currently re-pack
every strip many times.

**Risk:** medium-high. Pre-packed format is implementation-
specific; if `gemm` updates the layout in a future version we
break silently. Need a version-pin and test fixtures to detect
layout drift.

**Effort:** 3-5 days IF the API is exposed. Indeterminate if not.

### If not exposed

Skip Phase 3. The win isn't worth forking `gemm` for a solo
project.

## Phase 4: Bench, decide, ship (or shelve)

After whichever subset of Phases 2-3 we land:

1. Re-run the bench from `docs/index-performance-plan.md` at all
   v1.1 shapes (n × d combinations).
2. Re-run `experiments/exp_index_databricks_port.py` for end-to-
   end notebook simulation.
3. Run `pytest -m perf` to verify nothing regressed.

**Ship criteria:**
- d=1536 `similarity_graph` improves by ≥10% over the v1.1+tile
  baseline (currently 0.77x of FAISS).
- No regression at d=128 (currently 2.32x faster than FAISS).
- All correctness tests still pass.

**Shelve criteria:**
- d=1536 improves by <5% — not worth the maintenance burden.
- d=128 regresses — the changes hurt more than they help.

If shelving, revert the working branch; the perf plan stays in
`docs/` as a record of what we tried.

## What's NOT in scope

- Hand-written SIMD microkernels (would beat faer's choice of
  `pulp` SIMD wrapper — different conversation, different plan).
- Apple AMX direct access (Apple Silicon only; warrants its own
  evaluation).
- Fused GEMM+threshold kernel (closes more gap but overlaps with
  the "SimSIMD-style Rust kernels" path).
- Any change that requires a new dependency or feature flag.

## Effort summary

| Phase | Effort | Dependency |
|---|---|---|
| 1. Profile | 0.5 day | none — start here |
| 2a. Buffer reuse | 1 day | Phase 1 says it's worth it |
| 2b. Direct gemm calls | 2-3 days | optional |
| 3. Pre-pack data | 3-5 days | `gemm` exposes packing |
| 4. Bench + ship/shelve | 1 day | always |
| **Total best case** | **~1 week** | |
| **Total worst case** | abandon after Phase 1 (0.5 day) | |

## Decision tree

```
Profile (Phase 1)
├── Microkernel >85% of time
│   └── ABANDON. The gap is BLAS-vs-faer kernel quality.
│       Wait for the BLAS-backend research.
└── Overhead is meaningful (microkernel <70%)
    └── Implement Phase 2a (buffer reuse).
        ├── Bench shows ≥10% improvement at d=1536
        │   └── SHIP. Stop here unless Phase 3 is cheap.
        ├── Bench shows 5-10% improvement
        │   └── Try Phase 2b + Phase 3 if `gemm` exposes packing.
        │       Re-bench. Ship if combined ≥10%.
        └── Bench shows <5% improvement
            └── ABANDON. Faer's overhead is small; the gap is
                fundamentally kernel quality. Wait for BLAS.
```

## Phase 1 result (April 2026)

Profiled via two custom harnesses (committed under `examples/`):

- `cargo run --release --example profile_similarity_graph --no-default-features`
  measures wall-clock time inside the `similarity_graph_ip` hot loop,
  splitting into `gemm` (the `ip_batch` call), `emit` (threshold
  filter + edge push), and `concat` (final merge of per-tile chunks).
- `cargo run --release --example profile_ip_batch --no-default-features`
  drills inside `ip_batch` itself, splitting alloc / wrap / matmul.

**Result at n=20k, d=1536, tile=384, on Apple Silicon (M-series):**

```
similarity_graph_ip wall fraction:
  gemm (ip_batch):  99%
  emit:              1%
  concat:            0%
  remainder:         0%

inside ip_batch (1431 reps):
  matmul (faer):  99.9%   ← the SIMD kernel
  alloc:            0.1%
  wrap (MatRef):    0.0%
  other:            0.0%
```

**Conclusion: ABANDON Phase 2 and Phase 3.**

There is no measurable overhead to optimize. The d=1536 gap to FAISS
is the SIMD microkernel itself — pure-Rust faer/`gemm` running 20-30%
slower than hand-tuned BLAS on Apple AMX / Intel AVX-512 / OpenBLAS-
optimized paths.

Phase 2a (output buffer reuse) would save under 1%. Phase 2b (direct
`gemm` calls) would save 0%. Phase 3 (pre-packing) is bounded by
whatever fraction of faer's matmul is packing vs SIMD — even an
optimistic 10% upper bound is below the 5% shelve criterion.

**The only path to closing the d=1536 gap is the BLAS backend** (or
SimSIMD-style hand-tuned distance kernels). Decision deferred to the
result of `docs/blas-backend-research-prompt.md`.

The profiling harnesses stay in `examples/` so we can re-verify on
any future hardware (especially Linux x86_64 with AVX-512 — the
balance might shift there).

## Why this plan over going straight to BLAS

The BLAS-backend research will probably take a week to digest and
the implementation is 1-2 weeks after that. This plan caps at
~1 week total, has a clear abandon-early option (Phase 1 alone is
half a day), and has zero distribution risk — no new dependencies,
no wheel-size growth, no ABI clash, no per-platform CI matrix.

If the gain is meaningful it's a free improvement landed before
the BLAS decision. If the gain isn't meaningful, the profile data
is itself useful — it tells us whether the BLAS path is the right
one (because the kernel itself, not the overhead, is the bottleneck).
