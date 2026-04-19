# We Built a Rust-Backed Python Clustering Library. Here's What We Actually Learned.

There's a certain breed of side project that starts with "I want to learn Rust" and ends with you reading LLVM auto-vectorization documentation at 2am, wondering whether your distance kernel fits in a cache line.

This is the story of building **rustcluster** — a Rust-backed Python K-means clustering library that beats scikit-learn by 1.5-6.4x on the workloads most people actually run. But the interesting parts aren't the benchmark numbers. The interesting parts are all the decisions we made before writing a single line of code, and why most of our initial instincts were wrong.

---

## The premise: not "Rust is fast," but "where is sklearn slow?"

If you start a project like this by saying "Rust is fast, therefore our library will be fast," you're going to have a bad time. scikit-learn's K-means isn't some naive Python loop. It's Cython with OpenMP parallelism, and its inner distance kernel uses a BLAS matrix-multiply trick that turns the assignment step into a call to MKL or OpenBLAS. These are libraries with decades of hand-tuned assembly behind them.

So the first thing we did — before writing any Rust — was study what sklearn actually does. We read the source. Not the docs, the source. And what we found changed our entire approach.

### What sklearn's K-means actually does

When you call `KMeans.fit(X)`, here's what happens:

1. Validate input, force it to C-contiguous float64 (which can trigger a copy)
2. Mean-center the dense input for numerical stability
3. Precompute squared row norms
4. Run kmeans++ initialization (partially in Python, partially in Cython)
5. For each iteration of Lloyd's:
   - Compute distances using the BLAS trick: `||x - c||² = ||x||² + ||c||² - 2·x·cᵀ`
   - This turns the distance computation into a GEMM call
   - Take the argmin per row
   - Recompute centroids
6. Repeat n_init times, keep best inertia

That BLAS trick is clever. Instead of computing pairwise distances point-by-point, you reformulate it as a matrix multiply plus some pre-computed norms. On high-dimensional data, this is devastatingly effective because GEMM is one of the most optimized operations in all of computing.

But here's the thing: **GEMM has overhead**. It allocates intermediate buffers. It has setup cost. And for low-dimensional data — say, d < 20 — that overhead dominates the actual computation.

Most real-world clustering is low-dimensional. Customer segmentation, geospatial clustering, low-dimensional embeddings. The BLAS trick is optimized for a regime that isn't the common case.

That's our opening.

---

## Research before code: the part everyone skips

We spent more time researching than coding. This felt wrong in the moment. It felt like procrastination. It was the best decision we made.

### The algorithm question

Our first instinct was: implement Lloyd's algorithm. That's what everyone teaches. That's what sklearn defaults to. Ship it.

But the research uncovered something we didn't expect: the choice between Lloyd's, Elkan's, Hamerly's, and Yinyang isn't about "which is fastest." It's about which overhead profile matches your data regime.

**Elkan's algorithm** stores k lower bounds per point. That's O(n·k) extra memory. At n=100K and k=100, that's 80MB just for the bounds matrix. On the data sizes we're targeting, this causes cache thrashing — the exact opposite of what you want from an "optimization."

**Hamerly's algorithm** stores one lower bound per point. O(n) memory. The Hamerly paper reports it's fastest for dimensions up to about 50. That's our target regime.

sklearn's own maintainers have noted that their GEMM-boosted Lloyd now beats Elkan in many cases, which is part of why they changed the default. The lesson: even the sklearn team found that the "clever" algorithm isn't always the right one.

Our decision: ship tuned Lloyd first (simplest, correct baseline), then add Hamerly as the first accelerator. Skip Elkan entirely for our regime.

### The kernel architecture question

We found two Rust clustering implementations with fundamentally different designs:

**linfa-clustering** uses `ndarray`'s `Zip` API for its assignment loop — `Zip::from(observations.axis_iter(Axis(0))).and(...).par_for_each(...)`. Elegant. Safe. Integrates with rayon through ndarray's parallel producers. But the compiler sees a high-level row-iteration abstraction, not a tight loop over contiguous memory.

**The `kmeans` crate** takes the opposite approach: raw `Vec<f64>`, hand-written SIMD, explicit `unsafe`, portable_simd. Fast, but the crate's own docs acknowledge "quite a bit of unsafe code."

Neither of these felt right. We wanted the performance of raw slices with the safety of bounds-checked code. The insight was that LLVM's auto-vectorizer works best on simple counted loops over contiguous memory — you don't need `unsafe` or manual SIMD to get vectorized code. You just need to get out of the compiler's way.

So we designed a three-layer kernel architecture:

```
Layer 1: PyO3 boundary    — validates input, releases GIL, returns NumPy arrays
Layer 2: Algorithm logic   — ndarray types, iteration management, convergence
Layer 3: Hot kernel         — raw &[f64] slices, #[inline(always)], auto-vectorized
```

The key move: Layer 2 uses ndarray for ergonomics and safety, but before entering the inner loop, it lowers the data into raw slices and passes them to Layer 3. The hot kernel never sees an ndarray type. It sees `&[f64]` and a loop count.

```rust
#[inline(always)]
fn squared_euclidean(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}
```

This compiles to SIMD on both x86 (AVX2) and ARM (NEON) without a single line of unsafe code. The `debug_assert` gives you safety in development, disappears in release builds, and the simple counted loop gives LLVM everything it needs to emit wide vector loads.

---

## The PyO3 boundary: where most performance is won or lost

Here's something nobody tells you about Rust-backed Python libraries: the biggest performance wins often come from the *boundary*, not the algorithm.

### Zero-copy input

`PyReadonlyArray2<f64>` gives you an `ArrayView2<f64>` via `.as_array()`. This is a zero-copy borrow — the Rust code reads directly from NumPy's memory buffer. No allocation, no copy.

But there's a subtlety. `as_array()` works on any strided layout, including Fortran-order and non-contiguous views. It'll give you a valid view, but iteration over that view will be slow because it's chasing pointers across non-contiguous memory.

We made a deliberate design choice: **the Rust boundary rejects non-C-contiguous arrays**. The Python wrapper handles normalization:

```python
def _prepare(self, X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    return np.ascontiguousarray(X, dtype=np.float64)
```

This means the Rust side can use `as_slice().unwrap()` and know it's getting a flat, contiguous buffer. The `unwrap()` will never panic because we've already checked contiguity. But we comment the invariant, because someone will read this code later and wonder.

### One boundary crossing, not many

A PyO3 function call costs about 68ns. That's nothing for a `fit()` call that takes milliseconds. But it would be catastrophic if you crossed the boundary for every point or every iteration.

sklearn's architecture requires some Python-level orchestration between iterations. Our design doesn't. We pass the data in once, run the entire k-means algorithm in Rust (all iterations, all n_init runs), and return results once. The Python process sees exactly two boundary crossings per fit: one call, one return.

### Release the GIL

This is table stakes but worth emphasizing: `py.allow_threads()` releases the GIL for the entire computation. If someone runs `fit()` in one thread, other Python threads can still make progress. Polars, tokenizers, and pydantic-core all do this. It's a one-line change that makes your library a good citizen in async and threaded Python applications.

---

## What we got wrong (and fixed)

### Rayon import scope

Our first build failed because we used `(0..n).into_par_iter()` in `lib.rs` without importing `rayon::prelude::*`. The trait `IntoParallelIterator` was in scope in `kmeans.rs` (where we had the import) but not in `lib.rs` (where we used it for `predict`).

A trivial fix, but a reminder: Rust's trait system means methods don't exist until you import the trait. Coming from Python, where methods are always available on the object, this is a recurring papercut.

### The `panic = "abort"` trap

Our research flagged this early, and it saved us: **never use `panic = "abort"` in a Python extension module**. With `abort`, an unexpected panic kills the entire Python process. Without it, PyO3 catches the panic and translates it into a Python exception. Your users get an error message. Your process stays alive.

This is one of those decisions that doesn't matter until it matters catastrophically. We put it in our architecture decisions doc before writing any code.

### Empty cluster handling

This is a subtle algorithmic correctness issue. During Lloyd's iteration, a cluster can end up with zero assigned points — especially with poor initialization or when k is close to n. If you naively recompute the centroid as "mean of assigned points," you get a division by zero.

The fix: detect empty clusters and re-seed them by placing the centroid at the point with the largest distance to its currently assigned centroid. This is what sklearn does. It's not glamorous, but without it, your library will produce garbage results on edge cases that real users will hit.

---

## The benchmark results

Single-threaded, n_init=1, max_iter=100, Lloyd's algorithm for both:

| n | d | k | Speedup |
|---|---|---|---------|
| 1,000 | 2 | 8 | 2.5x |
| 1,000 | 8 | 8 | **6.4x** |
| 1,000 | 16 | 8 | 3.8x |
| 1,000 | 32 | 8 | 2.4x |
| 10,000 | 2 | 8 | 2.3x |
| 10,000 | 8 | 8 | 2.4x |
| 10,000 | 16 | 32 | 1.8x |
| 10,000 | 32 | 32 | 1.5x |
| 100,000 | 2 | 8 | 2.5x |
| 100,000 | 8 | 32 | **2.8x** |
| 100,000 | 16 | 32 | 1.8x |
| 100,000 | 32 | 32 | 1.6x |

The pattern is exactly what the research predicted. The sweet spot is low-dimensional data (d ≤ 16) where sklearn's BLAS overhead dominates. The 6.4x win at d=8 is notable: at 8 dimensions of float64, each point is exactly 64 bytes — one cache line. Our direct row-wise kernel reads one cache line per point per centroid comparison. sklearn's GEMM trick can't compete at this granularity.

As dimensions increase, sklearn's BLAS approach becomes more competitive, and our advantage narrows to 1.5x at d=32. At d=128+, we'd expect sklearn to catch up or win — and that's fine. Adding a BLAS path for high-d is a well-defined future optimization, not a gap in the architecture.

### What these numbers don't show

linfa (the main Rust ML library) achieved about 1.3x over sklearn on 1M points in a published benchmark. We're seeing 1.5-6.4x. The difference isn't because our Rust is better. It's because we specialized harder:

1. **Strict contiguous fast path** — no Fortran-order tolerance, no strided views
2. **Raw-slice kernel** — not ndarray Zip, not generic iteration
3. **Full kmeans++ in Rust** — sklearn's init is partially Python
4. **Zero intermediate allocations** — no distance matrix, no norm precomputation
5. **Single boundary crossing** — no Python-level orchestration between iterations

"Rust is fast" gave linfa 1.3x. **Specialization** gave us 2-6x. The language is necessary but not sufficient.

---

## Architectural decisions we're most proud of

### The three-layer kernel

This is the decision that holds the whole project together. By separating the PyO3 boundary, the algorithm logic, and the hot kernel into distinct layers with different type signatures, we got:

- Safety at the boundary (input validation, GIL management)
- Ergonomics in the algorithm (ndarray types, readable iteration logic)
- Performance in the kernel (raw slices, auto-vectorization)

Any one of these layers can be replaced without touching the others. When we add Hamerly's algorithm, we swap Layer 2. When we add explicit SIMD, we swap Layer 3. When we add f32 support, we add a second specialization at Layer 1.

### Rejecting Elkan for our regime

It would have been easy to implement Elkan's algorithm because it's the textbook "faster K-means." But the research showed it's the wrong tradeoff for low-dimensional, medium-n data — its O(n·k) memory overhead causes the exact cache problems we're trying to avoid.

This is a case where "doing less" was the right engineering decision. Hamerly (one bound per point, O(n) memory) is the correct accelerator for our target regime, and we know exactly when to add it.

### The Python wrapper as a normalization layer

The Rust boundary is strict: C-contiguous float64 or reject. The Python wrapper is permissive: it'll accept float32, integers, Fortran-order arrays, and quietly normalize them.

This gives us the best of both worlds. Users get a forgiving API. The Rust kernel gets exactly the memory layout it needs. And the conversion cost (when it happens) is explicit and measurable — it's a single `np.ascontiguousarray()` call, not hidden strided iteration throughout the algorithm.

---

## What's next

This MVP proves the architecture works. The roadmap:

**v1.1: Hamerly's algorithm.** One upper bound plus one lower bound per point. O(n) extra memory. The Hamerly paper says it's the fastest exact method for d < 50, which is our entire target regime.

**v2: Explicit SIMD with runtime dispatch.** The auto-vectorization results are good, but our research showed AVX-512 can roughly double throughput on supported hardware through better tail handling and wider vector loads. The key constraint: explicit SIMD must live behind runtime ISA detection, not compile-time flags, because distributed wheels need to run on any CPU.

**v3: BLAS integration for high-d.** When d > 100, the matrix-multiply distance trick genuinely wins. We'd integrate `faer` or `blas-sys` to match sklearn's approach for that regime. This isn't about ego — it's about having the right tool for each data shape.

**Eventually: more algorithms, community contributions.** DBSCAN, HDBSCAN, agglomerative clustering. The key architectural insight for extensibility: don't extract shared traits until you have two concrete implementations to generalize from. Generalizing from one example gives you the wrong abstractions.

---

## The real lesson

The real lesson isn't about Rust or Python or K-means. It's about the value of research before code.

We spent more time reading sklearn source code, studying algorithm papers, and understanding cache line sizes than we spent writing the implementation. The implementation itself — all four Rust files, the Python wrapper, 31 tests — came together in a single session. It compiled on the second try (missing import). Every test passed first run.

That doesn't happen because we're good at Rust. It happens because by the time we started writing code, we knew exactly what we were building and why. Every decision — raw slices in the kernel, strict C-contiguous at the boundary, Lloyd's not Elkan's, three layers not two — was made before the first `cargo init`.

Most performance work is research work. The code is just the last step.

---

*rustcluster is open source and available at [github.com/mfbaig35r/rustcluster](https://github.com/mfbaig35r/rustcluster). It's an early MVP focused on K-means. Contributions welcome.*
