# Deep Research for rustcluster

## Bottom line

For a Rust-backed Python K-means library aimed at **low-dimensional** data (`d < 20`) and **medium** sample counts (`1K–100K`), the strongest evidence points toward a deliberately **non-generic** design: a **strict zero-copy C-order `float64` fast path**, a **cache-tuned direct distance kernel**, and at least one **triangle-inequality exact accelerator** that is lighter than Elkan. In practice, that means a very strong MVP is **tuned Lloyd for the smallest `d` / smallest `k` cases** plus **Hamerly for the main exact-acceleration path**; then add **Yinyang** only if you later need better behavior at larger `k`. The public evidence does **not** support leading with a BLAS-first design for your target regime. scikit-learn already uses a GEMM-based dense Lloyd kernel, yet its own maintainers note that this GEMM speedup made Lloyd competitive enough that Elkan is no longer always faster, and they also have open performance discussions around BLAS / OpenMP / thread-pool interactions. citeturn9view2turn10view0turn11view0turn12search21turn45search12turn45search8turn45search20turn48view3

The two biggest opportunities to beat scikit-learn on your stated MVP are therefore not “use Rust” in the abstract, but rather: **avoid avoidable Python-side copies**, and **specialize harder for low `d` than scikit-learn’s general-purpose dense path does**. scikit-learn’s `KMeans.fit()` explicitly converts dense input to **C order**, which can itself trigger a copy for non-C-contiguous arrays; it mean-centers dense input; it precomputes row norms; and then it dispatches into Lloyd or Elkan. If `rustcluster` can keep input borrowing cheap and make the low-`d` assignment loop smaller and more branch-efficient than sklearn’s current path, that is the most plausible route to clear wins in your target band. citeturn47view1turn48view0turn48view3

The evidence is strongest for algorithmic tradeoffs, scikit-learn’s source behavior, PyO3 / rust-numpy semantics, and wheel-building practice. The evidence is **weaker** on the exact dimension-by-dimension crossover you asked for between **auto-vectorized Rust**, **explicit AVX2 / AVX-512**, and **BLAS GEMM** for `d = 2, 8, 16, 32, 64, 128, 256, 512`: I could find trustworthy public numbers for high-dimensional kernels and source-level clues about where crossovers should be, but not a clean public benchmark sweep matching your exact matrix. I flag those parts explicitly below. citeturn27view0turn48view3turn45search12

## Exact K-means algorithm choices

The most practically useful way to think about **Lloyd**, **Elkan**, **Hamerly**, and **Yinyang** is not “which one is fastest?”, but “which overhead profile matches the current `(n, d, k)` regime?” Charles Elkan’s original algorithm stores an upper bound plus **`k` lower bounds per point**, and the paper reports that it becomes more effective as `k` grows; the later Yinyang paper makes the memory tradeoff explicit as **`O(nk)`** lower-bound storage for Elkan. Hamerly’s approach keeps only a **single lower bound per point** to the second-closest center plus an upper bound to the assigned center, which is why its extra memory is much smaller. The Hamerly paper reports that it is “significantly faster than any competing algorithm” in dimensions **up to about 50**, while Elkan remains the strong baseline for higher-dimensional data. Yinyang reorganizes lower bounds into groups, giving **consistent speedups** over standard K-means, Elkan, and Drake/Hamerly-style multi-bound methods, with the paper reporting **order-of-magnitude** gains over classic Lloyd and **more than 3× on average** over the fastest previous exact methods in its experiments. citeturn8view2turn9view0turn9view2turn10view0turn11view0turn11view1turn11view3

For your target regime, the most important practical finding is this: **low `d` strongly favors low-overhead bound structures**. The Hamerly paper explicitly says its method is fastest in dimensions up to about 50 and attributes that to a strong lower bound that often eliminates the inner loop plus the fact that the algorithm is simple and easy to optimize. A later poster by Newling and Fleuret summarizes the state of exact K-means this way: **low dimension `< 10` favors Annular / Exponion-style global bounds, moderate dimension `< 100` favors Yinyang, and high dimension `>= 100` favors Elkan**. That is not a direct Hamerly-vs-Yinyang verdict, but it reinforces the same pattern: **as dimension falls, the best exact algorithm usually becomes the one with the smallest bookkeeping overhead per point**, not the one with the most elaborate pruning structure. citeturn9view2turn10view0turn8view3

On memory, the gap is large enough to matter operationally. The Yinyang paper’s cost table gives Elkan **`O(nk)`** space and Yinyang **`O(nt)`**, with `t` the number of maintained lower bounds per point; Hamerly’s original algorithm tracks only one lower bound per point plus an upper bound and assigned-center metadata. A concrete `f64` example makes the tradeoff clear: with `n = 100,000` and `k = 100`, Elkan’s extra lower-bound matrix alone is about **80 MB**, while a Hamerly-style `upper + lower + label` layout is on the order of **a few megabytes**, and Yinyang with `t = k / 10` comes out around **8 MB** just for lower bounds. Halve those numbers with `f32`. That is why Elkan is often unpleasant for the exact regimes where medium `n` and moderate `k` meet a tight Python extension memory budget. citeturn9view0turn9view2turn9view3turn11view3

The best concrete recommendation for `rustcluster` is therefore:

- **Ship tuned Lloyd first** for very small `k`, tiny `d`, and as the simplest correctness baseline.
- **Add Hamerly next** as the exact accelerator most aligned with `d < 20`, medium `n`, and a Python-package memory budget.
- **Treat Yinyang as the next step** once you care about larger `k` or want stronger wins on moderate dimensions.
- **Do not make Elkan your flagship speed path** unless you also accept its `O(nk)` memory cost and the fact that even scikit-learn now sees regimes where its GEMM-boosted Lloyd beats Elkan. citeturn9view2turn10view0turn11view0turn45search12

A minimal decision rule that is simple enough to implement and benchmark honestly would look like this:

```rust
enum ExactEngine {
    Lloyd,
    Hamerly,
    Yinyang,
}

fn choose_engine(n: usize, d: usize, k: usize) -> ExactEngine {
    if d <= 16 && k <= 32 {
        ExactEngine::Lloyd
    } else if d <= 32 && k <= 256 {
        ExactEngine::Hamerly
    } else {
        ExactEngine::Yinyang
    }
}
```

That rule is **not** a published optimum; it is a practical starting heuristic derived from the papers above plus your target product regime. The important part is not the exact thresholds, but that you benchmark the **dispatch rule itself** and treat it as a model-selection problem. citeturn9view2turn10view0turn11view0

## Rust kernel design and memory layout

The two Rust implementations I could inspect point in very different directions, and that contrast is useful. `linfa-clustering` is written around `ndarray`; its assignment step uses `Zip::from(observations.axis_iter(Axis(0))).and(...).par_for_each(...)`, and centroid recomputation also uses `Zip` over rows and counts. This is elegant, safe, and integrates with Rayon through ndarray’s parallel producers, but it does **not** use explicit SIMD in the hot path; it is essentially asking the compiler to optimize relatively high-level row-iteration code. By contrast, the `kmeans` crate states outright that it targets throughput, takes samples as a **raw vector instead of ndarray / nalgebra**, uses **hand-written SIMD**, enables `portable_simd`, and uses “quite a bit of unsafe code” because it materially improved speed. Its public API and docs also show row-major centroid storage and explicit SIMD lane counts such as `KMeans<_, 8>` for wide `f64` vectors. citeturn16view1turn17view2turn18view0turn17view3turn17view4turn19view2

That difference tells you what to steal. For an MVP whose only job is **fast dense K-means**, do **not** write the hottest kernel against a fully generic `ArrayView2` abstraction and hope for the best. Keep the public API ergonomic, but internally lower the data into something closer to the `kmeans` crate’s representation: **row-major contiguous samples**, **row-major contiguous centroids**, and a kernel signature that can take `&[f64]` plus explicit `n`, `d`, and `k`. The strongest support for this is indirect but consistent: scikit-learn forces dense arrays to C-order before fit; rust-numpy makes contiguous-vs-strided status available explicitly; and the `kmeans` crate’s design rationale is exactly to avoid abstraction overhead in the hot path. citeturn47view1turn37view0turn18view0turn19view2

On auto-vectorization, the current best general guidance comes from the entity["organization","LLVM","compiler infrastructure"] vectorizer documentation and the Rust Performance Book: vectorization works best when the compiler sees **simple counted loops**, **proven in-bounds indexing**, and **contiguous memory**. LLVM has both a **Loop Vectorizer** and an **SLP Vectorizer**; the Rust Performance Book specifically calls out restructuring code so that bounds checks are eliminated in hot loops. The relevant practical pattern for distance kernels is to take equal-length slices up front, then iterate over `0..d` or chunked iterators over contiguous spans; that makes it much easier for LLVM to widen loads and multiplies. `linfa-clustering`’s use of row iteration and `Zip` likely gives the compiler enough structure for some vectorization on contiguous inputs, but it is inherently more opaque than a dedicated kernel over plain slices. citeturn23search6turn23search4turn23search8turn16view1

The strongest public benchmark I found for **auto-vectorized vs explicit SIMD** does **not** match your requested low-dimensional sweep, but it is still informative. On recent entity["company","Intel","semiconductor company"] Xeon hardware, SimSIMD reports for **1536-dimensional `f32` squared L2** roughly **3.0 M ops/s** for auto-vectorized code, **3.3 M ops/s** for AVX2, and **6.0 M ops/s** for AVX-512. In other words, explicit AVX2 only modestly beats good auto-vectorization there, while AVX-512 approximately doubles throughput by removing tail handling and widening lanes. That tells you something important for `rustcluster`: **explicit SIMD is most compelling when it gives you better tail handling, better FMA structure, or runtime dispatch to wider ISAs**, not merely because “hand-written intrinsics are always faster.” citeturn27view0

What I could **not** find is a trustworthy public benchmark sweeping **`d = 2, 8, 16, 32, 64, 128, 256, 512`** and comparing **Rust auto-vectorization**, **explicit AVX2 / AVX-512**, and **BLAS GEMM** apples-to-apples. That gap matters. The safe reading of the available evidence is:

- for **very low `d`**, the fixed setup cost of GEMM and the temporary `n × k` buffer are likely to dominate;
- for **moderate to high `d`**, GEMM becomes more plausible, which is exactly why sklearn’s dense Lloyd path uses it;
- explicit SIMD is most likely to help in your target regime when paired with **runtime ISA dispatch** and a **strict contiguous fast path**. citeturn48view3turn27view0turn45search12

For memory layout, the best policy is also the simplest:

```rust
pub struct DenseSamples<'a> {
    pub data: &'a [f64], // row-major, n * d
    pub n: usize,
    pub d: usize,
}

#[inline(always)]
fn l2sq_row_to_centroid(x: &[f64], c: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), c.len());
    let mut acc = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - c[i];
        acc += diff * diff;
    }
    acc
}
```

For **low-dimensional Euclidean K-means**, row-major sample storage is the right default because each point is consumed contiguously. On mainstream 64-byte-line CPUs, an `f64` point with `d = 8` fits in **one cache line**, `d = 16` in **two**, and `d = 20` in **three**. That is exactly your target regime, and it is one reason direct row-wise kernels can be so effective. A column-major or struct-of-arrays design is more attractive when you are reorganizing the computation around matrix multiplies or processing the same dimension across many points at once. The public sources I inspected all tilt the same way: scikit-learn standardizes on C-order input for fit, `linfa-clustering` iterates rows, and the `kmeans` crate stores raw row-major buffers. citeturn47view1turn17view2turn18view0turn19view2

My practical recommendation is to implement **three kernel tiers**:

1. a **strict contiguous scalar / auto-vectorized kernel** for correctness and baseline measurement;
2. an **explicit SIMD kernel** for `d <= 32` or `d <= 64`, using runtime dispatch;
3. an optional **GEMM path** only for larger `d` or large batched assignment work, and only after benchmarking proves a crossover on your target CPUs.

That is more work than a single “generic” kernel, but it matches both the public evidence and your product goal. citeturn27view0turn48view3turn45search12

## PyO3 and NumPy boundary behavior

`PyReadonlyArray2::as_array()` is as close to zero-copy as rust-numpy can reasonably make it: the docs say it “provides an immutable array view of the interior of the NumPy array,” i.e. it gives you an `ndarray::ArrayView` over the existing NumPy allocation. It does **not** require C-contiguity. The separate `as_slice()` API is the one that only succeeds when the array is compatible with a plain Rust slice, and the untyped array methods expose whether the array is **contiguous**, **C-contiguous**, **Fortran-contiguous**, its **shape**, and its **strides**. In other words: `as_array()` is the right zero-copy borrow API; `as_slice()` is the right “I want a flat contiguous fast path” API. citeturn32view1turn37view0

For **non-contiguous**, **Fortran-order**, or otherwise **strided** arrays, the important subtlety is that zero-copy and fast-path are not the same thing. rust-numpy can still borrow such inputs, and its nalgebra interop docs explicitly warn that NumPy defaults to **C / row-major**, nalgebra defaults to **Fortran / column-major**, and sliced views often have **non-standard strides**. The docs recommend **dynamic strides** when you do not fully control layout, and they also note that some conversions can panic on **negative strides**. The consequence for `rustcluster` is straightforward: either **reject layout-mismatched arrays up front** on the hot path, or expose a slow path that materializes a C-contiguous copy deliberately. Do not silently feed arbitrary strides into your hottest assignment loop and expect to be competitive. citeturn32view1turn37view0

On dtype conversion, rust-numpy gives you a clean policy switch via `PyArrayLike<..., TypeMustMatch>` versus `PyArrayLike<..., AllowTypeChange>`. The docs say that with `TypeMustMatch`, the element type must match exactly; with `AllowTypeChange`, it will be cast by NumPy’s `asarray`. That means if a user passes `float32` into an API typed as `f64` and you permit type change, the conversion is **not free**: it is a real cast mediated by NumPy, which implies allocation and a full data pass. For your MVP, the performance-first default should be **reject mismatched dtypes** on the main entry point, and if you want convenience, provide a separate wrapper that either copies explicitly or dispatches into a native `f32` implementation. citeturn36view0turn36view1turn36view2turn35search10

A good API split looks like this:

```rust
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
fn fit_f64_c<'py>(x: PyReadonlyArray2<'py, f64>) -> PyResult<()> {
    if !x.is_c_contiguous() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "rustcluster fast path requires a C-contiguous float64 array",
        ));
    }
    let view = x.as_array(); // zero-copy borrow
    // call row-major kernel here
    let _ = view;
    Ok(())
}
```

This is the pattern I would use for the speed-critical entry point: **strict and unsurprising**. Then add a convenience wrapper in Python that normalizes with `np.asarray(x, dtype=np.float64, order="C")` when the caller prefers ergonomics over raw speed.

The best public microbenchmark I found for **Python-to-Rust call overhead** is in a PyO3 performance issue. In that measurement, a no-argument PyO3 `#[pyfunction]` took about **67.8 ns** per call, compared with about **43 ns** for a pure Python no-op on that setup. The same issue then shows that a body that only returns `None` can be as low as **24.8 ns**, and attributes most of the overhead to PyO3’s panic / GIL-pool handling rather than to argument parsing itself. The takeaway is not that PyO3 is “slow”; it is that you should **batch work aggressively** and never benchmark `fit()`-style APIs by calling into Rust for every single point or every single distance. The boundary is cheap in absolute terms, but still large relative to a tiny low-`d` arithmetic kernel. citeturn39view0

Compared with Cython, the evidence is older and weaker, but directionally consistent. A user benchmark cited in the PyO3 discussion reported an empty Cython `cpdef` call around **40 ns** versus roughly **70 ns** for a comparable Python baseline on that machine, and another benchmark summary found PyO3 and Cython call overheads “comparable with a slight advantage for Cython.” I would treat that as enough evidence to say: **Cython still has the edge for microscopic leaf calls**, but once the Rust side does any nontrivial batch computation, the difference becomes secondary. citeturn40search2turn40search6turn39view0

The memory-ownership patterns worth stealing are clearer from projects than from framework docs. `polars` says its core is written in Rust and that it can consume and produce Apache Arrow data “often with zero-copy operations”; its Python `from_arrow` docs say the operation is **zero copy for the most part**, and the user guide explicitly demonstrates **zero-copy** movement between Arrow and Polars. That is the right mental model for structured columnar data: if you can hand Rust a memory model it already understands, do that instead of converting into Python-native objects. citeturn31view4turn41search2turn41search5turn41search11

`tokenizers` takes the opposite approach for text inputs, and that is instructive too. Its Python bindings are “bindings over the Rust implementation,” but at the boundary it frequently converts Python sequences into **owned Rust strings and owned `EncodeInput<'static>` objects**; its source explicitly says “Convert Python inputs into fully-owned `EncodeInput<'static>`,” and the serialization path uses `PyBytes` for state transfer. That is the pattern to copy when lifetime-independent ownership on the Rust side is more important than avoiding copies. For `rustcluster`, numeric arrays should look more like Polars’ borrowing strategy; user-supplied init arrays, labels, or metadata can look more like tokenizers’ explicit ownership transfer. `orjson` is clearly speed-focused and Rust-backed, but from the public sources I reviewed I do **not** have enough high-confidence detail on its ownership strategy to recommend copying specific internals. citeturn41search0turn42view1turn42view4turn41search1

## What scikit-learn does today

The current `KMeans.fit()` implementation in scikit-learn is more optimized than many people realize. At the top of `fit()`, it validates the input with `accept_sparse="csr"`, accepts `float64` and `float32`, and forces dense input to **C order**, which can trigger a copy if the user passes a non-C-contiguous array. It checks parameters, computes sample weights, records the effective OpenMP thread count, validates any user-supplied init centers, **mean-centers dense input for numerical accuracy**, precomputes **squared row norms**, and then chooses either `_kmeans_single_elkan` or `_kmeans_single_lloyd`. It repeats initialization and the single-run clustering kernel `n_init` times, keeping the best inertia subject to a clustering-equivalence check, then restores the mean for dense input and writes out fitted attributes. citeturn47view1turn48view0

That sequence matters because it tells you exactly where a Rust library can win. The Python estimator wrapper is **not** where most time goes once `n` is moderate. The expensive pieces are the **initialization**, the **assignment / update kernels**, and any **copies triggered before the kernels run**. Therefore, if you benchmark your Python wrapper only against sklearn’s end-to-end `fit()` and do not separately measure the cost of **array normalization to C-order `float64`**, you will misattribute much of the runtime. scikit-learn’s own source is telling you what to split out. citeturn47view1

The dense Lloyd kernel uses the classic **BLAS distance trick**. In `_k_means_lloyd.pyx`, sklearn initializes a `pairwise_distances` buffer with `||C||²`, then calls a GEMM for the `-2 X C^T` term, and finally takes the argmin along each sample row. The source comments say explicitly that instead of computing the full squared distance matrix, it only needs to store `- 2 X·C^T + ||C||²` because `||X||²` is constant with respect to the per-sample argmin. This is exactly the same algebra documented in `sklearn.metrics.pairwise.euclidean_distances`, which writes Euclidean distance as `sqrt(dot(x,x) - 2 dot(x,y) + dot(y,y))`. citeturn48view3turn25search5

There are two practical consequences. First, the GEMM path **still requires a temporary distance-like buffer**, just chunked and partially simplified; it is not “free BLAS magic.” Second, the crossover where GEMM wins is **not universal**. scikit-learn’s own issue tracker shows that the maintainers are aware of performance sensitivity here: one issue explicitly notes that Lloyd got a bigger boost “due to the use of gemm indeed,” making Elkan no longer always faster; another proposes using **SYRK instead of GEMM** in pairwise distances; and others document regressions from BLAS / OpenMP interaction and multithreading oversubscription. So if your product focus is `d < 20`, the public evidence says you should assume **nothing** and benchmark the crossover yourself rather than inheriting a BLAS-centric design by default. citeturn45search12turn24search3turn45search8turn45search20turn45search16

On initialization, sklearn’s public source says its default is **“greedy k-means++”**, not vanilla Arthur–Vassilvitskii k-means++. The docs say that at each sampling step it makes several trials and chooses the best centroid among them, and the source shows a Python-level `kmeans_plusplus(...)` wrapper that checks arrays and row norms before calling a private `_kmeans_plusplus(...)` implementation. In practice, that means sklearn is already spending more effort than a naive one-shot initializer to improve convergence quality. For `rustcluster`, that is a useful design cue: if you want clean apples-to-apples benchmarking versus sklearn, compare either **fixed initial centers**, or **the same greedy k-means++ policy**, or you will end up benchmarking initialization quality differences instead of the clustering kernel. citeturn48view0turn48view1turn47view1

The most relevant known sklearn performance issues for your project are these:

- dense Lloyd has improved enough that **Elkan is not always faster anymore**, particularly because the dense path uses GEMM; citeturn45search12
- `pairwise_distances` and related dense kernels can regress when **BLAS** and **OpenMP** thread pools interact badly; citeturn45search8turn45search20
- some models show **multithreading slower than single-threading** in real settings; citeturn45search16
- maintainers still discuss whether **higher-level BLAS** calls are the right performance strategy, meaning the implementation frontier is not “finished.” citeturn45search4turn24search3

For `rustcluster`, the implication is favorable: sklearn is strong, but it is still a compromise between portability, maintainability, and many workloads. A specialized low-`d`, dense, strictly borrowed Rust path can still reasonably beat it. citeturn47view1turn48view3turn45search12

## Benchmark design that will survive scrutiny

The benchmark that will tell you the truth is **not one benchmark**. You need at least three layers. First, a **kernel-only Rust microbenchmark** that measures assignment-loop throughput with fixed centroids and no Python boundary. Second, an **FFI-only microbenchmark** that measures the cost of calling a trivial PyO3 function and of borrowing / validating arrays with the exact signatures you plan to ship. Third, an **end-to-end Python benchmark** that includes `fit()` with identical initialization policy, identical thread settings, and the same dtype / layout assumptions. The PyO3 overhead data shows why this decomposition matters: a no-op boundary can still cost tens of nanoseconds, which is trivial for a large fit and meaningful for tiny per-call kernels. citeturn39view0

There is at least one public Rust-vs-sklearn K-means benchmark, but its value is mostly directional. Luca Palmieri reported, using `pytest-benchmark`, that a Python wrapper over `linfa` trained a K-means model on **1 million points** in about **467.2 ms**, compared with about **604.7 ms** for scikit-learn, i.e. roughly **1.3× faster**. That is a respectable win, but not a huge one, and it reinforces a useful point: Rust alone does not buy you 10×. The big wins come from **specialization**, **copy avoidance**, **kernel design**, and **threading discipline**. citeturn12search21

For `rustcluster`, I would make the benchmark matrix explicit:

- **single-thread baseline first** with BLAS and OpenMP pinned to one thread;
- then **1, 2, 4, 8 threads** as a separate study;
- sweep **`d = 2, 8, 16, 32, 64, 128, 256, 512`** with fixed `n` and `k`;
- separately sweep **`n = 1K, 10K, 100K`** at fixed `d` and `k`;
- separately sweep **`k = 8, 32, 128, 512`** at fixed `n` and `d`;
- include input-layout cases: **C-order `float64`**, **Fortran-order `float64`**, **non-contiguous view**, **C-order `float32`**. citeturn37view0turn47view1turn39view0

The reason to start single-threaded is not aesthetic. It is because sklearn’s issue tracker contains concrete examples of multithreading regressions and BLAS / OpenMP contention. If you start by comparing “whatever thread count each library decides to use,” you are as much benchmarking threadpool policy as clustering. Single-threaded numbers tell you whether your **kernel** is actually better. After that, thread sweeps tell you whether your **system design** is better. citeturn45search8turn45search16turn45search20

My tooling recommendation is:

- use **Criterion** for Rust kernel microbenchmarks;
- use **pytest-benchmark** for Python API and end-to-end `fit()` comparisons;
- use **hyperfine** only for command-line smoke tests and packaging checks, not for in-process microkernels.

That recommendation is a workflow judgment, not a published standard. The reason it fits your project is simply that the three tools measure three different things, and you need all three.

A sensible Python benchmark harness looks like this:

```python
import numpy as np
import rustcluster
from sklearn.cluster import KMeans

def make_x(n, d, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n, d), dtype=np.float64))

def bench_pair(n=100_000, d=16, k=32, init_centers=None):
    X = make_x(n, d)
    if init_centers is None:
        init_centers = X[:k].copy()

    rustcluster.fit(X, init=init_centers, n_init=1, max_iter=100, algorithm="lloyd")
    KMeans(
        n_clusters=k,
        init=init_centers,
        n_init=1,
        max_iter=100,
        algorithm="lloyd",
        random_state=0,
    ).fit(X)
```

The key methodological choices are the ones that remove confounders:

- **fixed `init`** when comparing kernels;
- **`n_init=1`** unless you are intentionally benchmarking initialization;
- **data generated once, outside the measured region**;
- **imports outside the measured region**;
- **thread counts fixed by environment**;
- **layout and dtype measured separately, not mixed into the main kernel benchmark**.

If you do that, your benchmark results will actually tell you what to build next.

## Library, build, and distribution choices

For the math layer, `ndarray`, `nalgebra`, and `faer` are not interchangeable. `ndarray` is the natural fit for NumPy-like multidimensional data and row-wise iteration; that is why `linfa-clustering` uses it, and why it is a reasonable outer representation for your Python-facing input. `nalgebra` is more naturally matrix-and-geometry flavored and, critically, rust-numpy’s own docs warn that nalgebra defaults to **Fortran / column-major standard strides** while NumPy arrays default to **C / row-major** and sliced views often have non-standard strides. `faer` describes itself as focused on **high performance for algebraic operations on medium / large matrices**. So the clean design split is: **use ndarray or plain slices for direct low-`d` kernels**, **avoid nalgebra in the hot path unless you have a compelling column-major reason**, and **reach for faer only if you deliberately add a BLAS-like / GEMM-heavy distance engine later**. citeturn45search7turn32view1turn23search3turn45search3

That leads to a strong architectural recommendation: keep the public and medium-level code ergonomic, but keep the innermost kernels representation-specific. In practice:

- public input: rust-numpy borrow;
- medium layer: validate dtype / layout / shape, maybe normalize once;
- hot layer: plain `&[f64]` row-major buffers and specialized kernels.

That is the same separation you see, in different ways, in both `linfa-clustering` and the more aggressively tuned `kmeans` crate. citeturn16view1turn17view2turn18view0turn19view2

On threading, the safe rule with PyO3 is simple even if the exact code differs by version: **release the GIL for the compute region, then run Rayon there; never move Python objects into Rayon worker closures**. Instead, convert or borrow what you need **before** the GIL release, then operate only on Rust-owned or Rust-borrowed numeric memory. That keeps ownership clear and avoids mixing Python reference-counted objects with parallel Rust work. The motivation follows directly from PyO3’s performance guidance and rust-numpy’s borrowing design, even though the official performance page focuses more on extraction and conversion costs than on Rayon specifically. citeturn49view0turn31view1

For build configuration, the most important recommendation is what **not** to do. Do **not** compile wheels with `-C target-cpu=native`; that is perfect for local benchmarking and wrong for distributable wheels. Instead, use a conservative wheel baseline and, if you write explicit SIMD, select AVX2 / AVX-512 / NEON paths via **runtime dispatch**. The SimSIMD results are a good reminder that ISA-specific kernels can pay off materially, especially on wide-vector machines and for tail handling, but that optimization belongs behind dispatch, not inside the wheel target itself. On entity["company","Apple","technology company"] Silicon, you should assume **no AVX**, so either let the compiler auto-vectorize to NEON or add a dedicated Arm path later. citeturn27view0

For release profiles, my practical recommendation is:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = "debuginfo"
debug = false
```

I am **not** recommending `panic = "abort"` for a Python extension module, because that gives up PyO3’s panic translation and turns unexpected panics into process termination. For local benchmarking, by all means test with native tuning too:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

but keep that as a benchmark-only configuration.

On packaging, the current maturin guidance is quite good. The maturin user guide says it has decent cross-compilation support for PyO3 bindings, recommends **manylinux-cross Docker images** for manylinux support, and also supports **Zig** for some cross builds. For Windows cross-compilation, it documents support through PyO3’s `generate-import-lib` feature and integration with `cargo-xwin`. The same guide now supports `maturin generate-ci github`, which can generate a GitHub Actions workflow and supports platforms including **manylinux**, **musllinux**, **Windows**, and **macOS**, plus optional **trusted publishing** to PyPI via OIDC. citeturn49view1

The supported workflow pattern is echoed by the `maturin-action` project. Its README shows the minimal action form, points back to `maturin generate-ci github`, and explicitly lists example projects, including **abi3** and **non-abi3** layouts and a **pydantic-core** example with **PyPy support**. That is valuable because it shows you which wheel strategies real large packages are using today. The action itself is maintained actively and is explicitly marketed as having built-in support for cross compilation. citeturn49view2

The packaging ecosystem here is largely anchored in the entity["organization","Python Packaging Authority","python packaging org"] and entity["organization","GitHub","code hosting platform"] Actions world. If you want the least painful CI/CD for `rustcluster`, the default should be:

- `maturin` backend;
- generated GitHub Actions workflow as the starting point;
- trusted publishing to PyPI;
- split wheel jobs per platform / architecture;
- benchmark-only jobs that use `target-cpu=native`, separate from release jobs.

That is the pattern most aligned with current tooling. citeturn49view1turn49view2

On Apple Silicon vs x86_64 vs Linux, the practical gotchas are mostly architectural. On Apple Silicon, the real issue is not maturin itself; it is that your x86_64 SIMD assumptions disappear, so you need either a NEON-aware explicit path or confidence in your scalar / auto-vectorized fallback. On Linux, manylinux compatibility and external BLAS / OpenMP packaging can become the real headache. On Windows, the main friction is always the import-library / CRT / SDK story, which maturin now softens through `generate-import-lib` and `cargo-xwin`. Those cross-platform facts line up with the maturin documentation and with the fact that mature packages such as pydantic-core ship Rust-backed Python artifacts in this ecosystem already. citeturn49view1turn49view2turn49view3

The one part of your packaging question where my evidence is weaker is the **exact current `pyproject.toml` structure** used by Polars and pydantic-core. I verified that Polars is Rust-core with Arrow-oriented zero-copy interop and that pydantic-core is a prominent Rust-backed Python package now living under the main Pydantic repository, and maturin-action directly cites pydantic-core as a real packaging example. I did **not** retrieve the exact current `pyproject.toml` files in this pass, so I would not claim line-by-line certainty there. The high-confidence part is the CI / wheel strategy and the maturin-based pattern around it. citeturn31view4turn41search2turn49view2turn49view3