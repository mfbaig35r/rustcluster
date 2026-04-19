# Lessons from Building a Rust-Backed Python Library

What we learned building rustcluster — a Rust-backed Python clustering library that ships K-means, DBSCAN, cosine distance, and evaluation metrics. These aren't abstract principles. They're things we got right, things we almost got wrong, and things we'd do the same way again.

---

## 1. Start with a narrow, benchmarkable claim

We didn't start with "build a clustering framework." We started with "build K-means in Rust and see if it beats scikit-learn." That's a claim you can prove or disprove in a single benchmark run. If the benchmark fails, you stop. If it succeeds, you have a foundation.

Our first version was four Rust files, one Python wrapper, and 31 tests. It did one thing: K-means with Lloyd's algorithm. It beat sklearn 1.5-6.4x on low-dimensional data. That was enough to justify everything that came after.

The temptation is to design the abstraction layer, the trait hierarchy, the plugin system, the metric framework — all before you know whether the core kernel is actually fast. Resist that. Prove the kernel first. The architecture can follow.

## 2. Research the competition before writing code

We spent more time reading sklearn source code than writing our own. This felt like procrastination. It was the best decision we made.

What we learned from the source:

- sklearn's K-means uses a BLAS matrix-multiply trick for distance computation. On high-dimensional data, this is nearly impossible to beat.
- But on low-dimensional data (d < 20), the BLAS overhead — intermediate buffer allocation, matrix formation — dominates the actual computation.
- sklearn's k-means++ initialization is partially Python. Doing it fully in Rust saves 10-30% of total runtime on large datasets.
- sklearn has known OpenMP/BLAS thread contention issues. Single-threaded benchmarks isolate the kernel from the threading policy.

This research shaped every architectural decision. Without it, we might have built a generic BLAS-backed implementation and lost to sklearn on our own target regime. Instead, we specialized where sklearn is weakest: direct row-wise kernels on contiguous memory for low-dimensional data.

**The lesson**: "Rust is fast" is not a strategy. "Rust can avoid this specific overhead that sklearn pays" is a strategy.

## 3. The three-layer kernel architecture

This is the single design decision we're most satisfied with. Every piece of rustcluster follows this pattern:

```
Layer 1: PyO3 boundary    — validates input, releases GIL, returns NumPy arrays
Layer 2: Algorithm logic   — ndarray types, iteration management, convergence
Layer 3: Hot kernel         — raw &[f64] slices, #[inline(always)], auto-vectorized
```

Why three layers instead of two? Because the concerns are genuinely different:

**Layer 1** deals with Python types, the GIL, dtype dispatch, contiguity checks. It's the only layer that knows PyO3 exists. When we added f32 support, only this layer changed (dtype dispatch).

**Layer 2** deals with algorithm flow: initialization, iteration, convergence, n_init orchestration. It uses ndarray for readable array manipulation. When we added Hamerly's algorithm, this is where the new logic lived.

**Layer 3** deals with raw numerical throughput. It sees `&[f64]` and a loop count. It doesn't know what algorithm is calling it or what Python type the data came from. When we added cosine distance, this is where the new kernel lived.

Each layer can change independently. We added DBSCAN without touching K-means. We added cosine distance without touching the algorithm logic. We added f32 without touching the hot kernels (generics handled it via monomorphization).

**What we'd do differently**: nothing. This decomposition held up through every feature addition.

## 4. Don't abstract until you have two concrete implementations

We knew from the start that we'd want a Distance trait. We also knew that designing it from one algorithm (K-means with Euclidean distance) would give us the wrong trait.

So we waited. We built K-means with direct `squared_euclidean()` calls. Then we built Hamerly's algorithm with the same calls. Only after shipping two algorithms that both used distance functions did we extract the `Distance<F>` trait.

The result was a trait that actually fit:

```rust
pub trait Distance<F: Scalar>: Send + Sync + Clone + Copy + 'static {
    fn distance(a: &[F], b: &[F]) -> F;
}
```

Stateless. Static dispatch via `D::distance(a, b)` — no vtable. Monomorphization produces identical code to the pre-trait version. When we later added `CosineDistance`, the trait was ready because it was designed from real usage, not hypothetical requirements.

We did the same thing with the `Scalar` trait for f32/f64 generics, and with `ClusterError` (originally `KMeansError`, renamed when DBSCAN needed shared error variants). Each abstraction was extracted from two or more concrete uses.

**Generalizing from one example gives you the wrong abstractions. Generalizing from two gives you the right ones.**

## 5. The PyO3 boundary is where most performance is won or lost

The hot kernel gets all the attention, but the PyO3 boundary decisions had a bigger impact on our benchmark numbers.

**Zero-copy input**: `PyReadonlyArray2<f64>::as_array()` borrows NumPy's memory directly. No copy, no allocation. But it works on any strided layout, including slow Fortran-order views. We made a deliberate choice: **reject non-C-contiguous arrays at the Rust boundary.** The Python wrapper handles normalization with `np.ascontiguousarray()`. This means the Rust kernel always gets a flat, cache-friendly buffer.

**One boundary crossing**: A PyO3 function call costs ~68ns. Trivial for a `fit()` call, catastrophic if you cross per-point or per-iteration. We pass data in once, run the entire algorithm (all iterations, all n_init runs) in Rust, and return results once. Two boundary crossings per fit.

**GIL release**: `py.allow_threads()` for all computation. One line that makes the library a good citizen in async and threaded Python applications. Every serious PyO3 library does this. Skipping it blocks the entire Python process during fit.

**Strict dtype fast path**: The Rust boundary requires C-contiguous float64 (or float32). No silent conversion, no fallback to strided iteration. The Python wrapper is permissive (accepts integers, Fortran order, any dtype) and normalizes before calling Rust. The cost of normalization is explicit and measurable. The Rust kernel never pays for it.

## 6. Auto-vectorization is enough (until it isn't)

Our hot distance kernel is a simple counted loop:

```rust
#[inline(always)]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}
```

No `unsafe`. No explicit SIMD intrinsics. No `packed_simd`. LLVM auto-vectorizes this into SIMD on both x86 (AVX2) and ARM (NEON) from the same source code.

Criterion benchmarks confirmed: `squared_euclidean` at d=8 runs in **1.36 nanoseconds**. That's ~5.9 GFlops/s on a single core. At d=8, one point fits in a single 64-byte cache line. The auto-vectorized kernel reads one cache line per point per centroid comparison.

Public benchmarks (SimSIMD) show explicit AVX2 only modestly beats good auto-vectorization. AVX-512 roughly doubles throughput through better tail handling — but that's a v3 optimization behind runtime ISA dispatch, not something you need in an MVP.

**The lesson**: get out of the compiler's way. Simple counted loops over contiguous slices. `#[inline(always)]` on the distance function. Raw `&[f64]` slices, not `ArrayView2` in the hot path. LLVM does the rest.

## 7. Benchmark methodology matters more than benchmark numbers

Our v1 showed Lloyd's at 39ms for one configuration. Our v2 showed 299ms for the same configuration. We thought we had a regression. We didn't.

The issue: v1's benchmark varied `random_state` per run. Most seeds converged in 2 iterations. The median was a lucky seed. v2's benchmark used the same seed for all runs — which happened to be a hard seed (26 iterations).

Once we fixed the methodology (vary seeds, measure median), both versions showed ~63ms. No regression at all.

What we learned to do right:

- **Vary random_state per run** — don't measure a single lucky/unlucky initialization
- **Benchmark single-threaded first** — isolate kernel performance from threading policy (sklearn has known OpenMP contention issues)
- **Use fixed initial centroids** when comparing algorithms — otherwise you're benchmarking initialization quality, not clustering speed
- **Three benchmark layers**: Criterion for Rust kernels, pytest-benchmark for Python API, manual timing for end-to-end
- **Separate sweeps**: vary d, n, k independently, don't conflate them

The benchmark that tells you the truth is not one benchmark. It's a systematic sweep with controlled confounders.

## 8. Hamerly beats Elkan — and the research told us so

Our initial instinct was to add Elkan's algorithm as the "fast" K-means variant. It's the textbook answer. sklearn supports it. The papers cite it.

The research said otherwise:

- Elkan stores k lower bounds per point — O(n*k) memory. At n=100K, k=100, that's 80MB just for bounds.
- Hamerly stores one lower bound per point — O(n) memory. A few MB for the same configuration.
- The Hamerly paper says it's fastest for dimensions up to 50. That's our entire target regime.
- sklearn's own maintainers note that GEMM-boosted Lloyd now beats Elkan in many cases.

We shipped Hamerly instead. On our benchmarks, it beats Lloyd by 27-38% at higher k (k=32) — exactly where the bound-filtering pays off. At low k, Lloyd is already fast enough that Hamerly's bookkeeping doesn't help.

The auto-dispatch heuristic we settled on:
- d ≤ 16 and k ≤ 32 → Lloyd (minimal overhead)
- Otherwise → Hamerly (bound filtering pays off)

**The lesson**: the "standard" algorithm isn't always the right one for your regime. Research the tradeoffs for your specific data shape.

## 9. f32 support is a cache-line argument

Adding f32 wasn't about precision tradeoffs. It was about cache lines.

At d=8 with f64, one point is 64 bytes — exactly one cache line. At d=8 with f32, one point is 32 bytes — **two points per cache line**. The distance kernel's throughput is memory-bandwidth-bound at these dimensions. Halving the memory footprint per point directly improves throughput.

The implementation used Rust generics with a `Scalar` marker trait:

```rust
pub trait Scalar: Float + Send + Sync + Sum + Debug + 'static { ... }
impl Scalar for f32 {}
impl Scalar for f64 {}
```

Monomorphization generates separate f32 and f64 codegen — no runtime overhead. The `Distance<F: Scalar>` trait composes with it cleanly. One refactor touched every Rust file, but the generated code for f64 was identical to before.

The key precision decision: accumulate centroid sums and kmeans++ sampling weights in f64 regardless of input type. The hot distance kernel runs in native precision (f32 or f64), but the reduction steps that are sensitive to floating-point accumulation error use f64. This matches the pattern used by serious numerical libraries.

## 10. DBSCAN validated the architecture

The real test of a modular architecture is the second algorithm. DBSCAN was ours.

What we had to touch:
- `error.rs` — rename `KMeansError` → `ClusterError`, add two DBSCAN-specific variants
- `dbscan.rs` — new file, entire algorithm
- `lib.rs` — register new module, add `Dbscan` pyclass
- `__init__.py` — add `DBSCAN` class

What we didn't have to touch:
- `kmeans.rs` — zero changes
- `hamerly.rs` — zero changes
- `utils.rs` — zero changes (DBSCAN reused `squared_euclidean_generic` and `validate_data_generic`)
- `distance.rs` — zero changes (DBSCAN used `SquaredEuclidean` directly)

DBSCAN's architecture followed the same three layers: PyO3 boundary (dtype dispatch, GIL release) → algorithm logic (parallel neighbor scan + BFS) → hot kernel (distance comparisons on raw slices). The only novel pattern was the two-phase approach: rayon-parallel neighborhood scan (embarrassingly parallel) followed by sequential BFS cluster expansion (inherently sequential due to data dependencies).

**The lesson**: if your second algorithm requires touching your first algorithm's code, your architecture is wrong. If it doesn't, your architecture is right.

## 11. Cosine distance was a 20-minute feature because of the Distance trait

Once the `Distance<F>` trait existed, adding cosine distance was:

1. A new struct and `impl Distance<F>` (15 lines of kernel code)
2. A `Metric` enum for runtime dispatch at the Python boundary
3. Entry points that match on the enum and call the appropriate monomorphized generic

Both K-means and DBSCAN got cosine support simultaneously. Zero changes to the algorithm logic — they're generic over `D: Distance<F>`.

The one wrinkle: Hamerly's bound updates assume Euclidean metric (they use `.sqrt()` on squared distances). Cosine K-means forces Lloyd automatically. This is documented in the trait, not hacked around.

The eps² optimization for DBSCAN also required metric awareness: Euclidean compares `squared_distance ≤ eps²` (no sqrt), cosine compares `cosine_distance ≤ eps` directly. We moved the eps-squaring into the dispatch layer so the generic algorithm always compares `distance ≤ threshold`.

## 12. Clustering metrics needed the same patterns

Silhouette score, Calinski-Harabasz, Davies-Bouldin — all followed the established patterns:

- Generic over `F: Scalar` (f32/f64)
- Operate on `ArrayView2<F>` and label vectors
- Handle noise labels (-1) from DBSCAN
- Release GIL during computation
- Accumulate in f64 for precision

No new architectural patterns needed. The metric functions are standalone (no pyclass state), exposed as module-level Python functions. They reuse `squared_euclidean_generic` from utils.rs.

## 13. Don't use `panic = "abort"` in a Python extension

This one came from research, not from a production incident, and we're glad.

With `panic = "abort"`, an unexpected panic in Rust kills the entire Python process. No traceback, no error handling, no recovery. With the default panic behavior, PyO3 catches panics and translates them into Python exceptions. Your users get an error message. Their process survives.

For a library that users call from Jupyter notebooks, killing the kernel on an unexpected panic would be unacceptable. We caught this in the architecture design phase and put it in our build profile constraints.

## 14. The benchmark results told us what we are and what we aren't

Our final benchmark (single-threaded, varying seeds, Lloyd's for both):

| Regime | Speedup vs sklearn |
|--------|-------------------|
| d=2, k=8, n=1K | 3.0x |
| d=8, k=8, n=1K | 2.9x |
| d=8, k=32, n=100K | 3.2x |
| d=32, k=32, n=100K | 1.4x |

The pattern is clear: we win big on low-dimensional data and the advantage narrows as dimensions increase. At d=32, we're still faster but not dramatically.

This tells us exactly what we are: a library optimized for the low-to-medium dimensional clustering that most people actually do. Customer segmentation, geospatial clustering, low-dimensional embeddings. We're not a universal sklearn replacement and we don't pretend to be.

linfa (the main Rust ML library) achieved ~1.3x over sklearn with a generic approach. We hit 1.4-3.2x. The difference isn't because our Rust is better — it's because we specialized harder: strict contiguous fast path, raw-slice kernels, full kmeans++ in Rust, zero intermediate allocations, single boundary crossing.

**"Rust is fast" gave linfa 1.3x. Specialization gave us 2-3x.**

## 15. Most performance work is research work

We spent more time reading papers, studying sklearn source code, and understanding cache line sizes than we spent writing the Rust implementation. The implementation itself — all source files, the Python wrapper, 167 tests — came together across a handful of focused sessions. Compilation errors were trivial (missing imports, turbofish annotations).

That doesn't happen because we're particularly good at Rust. It happens because by the time we started writing code, we knew exactly what we were building and why. Every decision — raw slices in the kernel, strict C-contiguous at the boundary, Hamerly not Elkan, three layers not two — was made before the first `cargo init`.

The code is just the last step. The research is the work.

---

## Project timeline

What we shipped, in order:

1. **K-means MVP** — Lloyd's algorithm, kmeans++ init, rayon parallelism, sklearn-compatible API. 20 Rust tests, 31 Python tests.
2. **Hamerly's algorithm** — exact accelerator with O(n) bounds, auto-dispatch heuristic. Benchmark methodology fix.
3. **Criterion benchmarks + Distance trait** — kernel profiling, generic Distance<F> with static dispatch.
4. **Native f32** — Scalar trait, generic algorithms, PyO3 dtype dispatch.
5. **DBSCAN** — parallel neighbor scan + BFS, shared ClusterError.
6. **Cosine distance** — CosineDistance impl, Metric enum, runtime dispatch for both algorithms.
7. **Clustering metrics** — silhouette, Calinski-Harabasz, Davies-Bouldin.
8. **CI/CD + PyPI release** — GitHub Actions for 3 OS × 2 Python versions, multi-platform wheels.

Final count: 8 Rust source files, ~3500 lines of Rust, ~250 lines of Python wrapper, 167 tests, 10 commits.

Each step built on the previous one. No step required rearchitecting what came before. That's the three-layer kernel and the "extract traits from concrete implementations" philosophy working as designed.
