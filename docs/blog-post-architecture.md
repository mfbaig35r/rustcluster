# The Three Layers of a Fast Clustering Library

*How we built rustcluster: a Rust-backed Python clustering library that's 1.4-3.2x faster than scikit-learn, ships six algorithms, and taught us more about compiler auto-vectorization than we expected.*

---

I've spent the last few months building [rustcluster](https://github.com/mfbaig35r/rustcluster), a Python clustering library with a Rust core. Six algorithms, three distance metrics, KD-tree acceleration, an embedding-specific pipeline, and 421 tests. It installs with `pip install rustcluster` and drops into any sklearn workflow.

But this post isn't about features. It's about the architecture that makes those features possible without the codebase becoming a maintenance disaster. Because the interesting part of building a Rust-backed Python library isn't the Rust or the Python — it's the boundary between them.

## The problem with "just rewrite it in Rust"

The naive approach to speeding up Python with Rust is: take the hot function, rewrite it in Rust, call it from Python via PyO3. This works for exactly one function. Then you add a second algorithm and realize you've duplicated all your input validation, dtype handling, and GIL management. By the third algorithm, your `lib.rs` is 5,000 lines of copy-pasted PyO3 boilerplate with algorithmic logic buried somewhere in the middle.

We hit this wall early. K-means was fast, but adding DBSCAN meant re-solving every PyO3 boundary problem from scratch. Same contiguity checks, same f32/f64 dispatch, same GIL release pattern, same error types, same pickle serialization. The algorithm was 50 lines. The binding code was 200.

The fix was architectural, not incremental.

## Three layers, one principle

Every piece of compute in rustcluster lives in exactly one of three layers:

```
Layer 1: PyO3 boundary     (lib.rs)      — input validation, GIL release, dtype dispatch
Layer 2: Algorithm logic    (kmeans.rs)   — iteration, convergence, ndarray types
Layer 3: Hot kernel         (utils.rs)    — raw &[F] slices, simple counted loops
```

The principle: **each layer talks only to the layer below it, and each layer has exactly one job.**

This sounds obvious. It isn't. The temptation is always to reach from Layer 1 directly into Layer 3, or to put validation logic inside the algorithm, or to let PyO3 types leak into the math. Every time we broke this rule, we paid for it later.

### Layer 3: The hot kernel

Let's start at the bottom, because this is where the actual speedup lives.

```rust
#[inline(always)]
pub fn squared_euclidean_generic<F: Scalar>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = F::zero();
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc = acc + diff * diff;
    }
    acc
}
```

That's it. No ndarray, no generics beyond the scalar type, no error handling, no bounds checks beyond debug assertions. Raw `&[F]` slices and a counted loop.

Why? Because LLVM's auto-vectorizer needs simple patterns. Give it a counted loop over contiguous memory with no branches and it will emit SIMD instructions — SSE, AVX, NEON depending on the target. Give it anything more complex and it bails out to scalar code.

We tested this extensively. Hand-rolled SIMD (using `std::arch`) gave us maybe 10% over what LLVM generated from the simple loop. Not worth the maintenance cost, the `unsafe` blocks, or the platform-specific code paths. The compiler is good enough if you give it clean inputs.

The `Scalar` trait is the key abstraction at this layer:

```rust
pub trait Scalar: Float + Send + Sync + Sum + Debug + 'static {
    fn to_f64_lossy(self) -> f64;
    fn from_f64_lossy(v: f64) -> Self;
}
```

Implemented for `f32` and `f64`. That's it. No runtime polymorphism, no trait objects, no vtable dispatch. The compiler monomorphizes every call site, so `squared_euclidean_generic::<f32>` and `squared_euclidean_generic::<f64>` are two completely separate functions with type-specialized SIMD.

The `Distance` trait sits on top of `Scalar`:

```rust
pub trait Distance<F: Scalar>: Send + Sync + Clone + Copy + 'static {
    fn distance(a: &[F], b: &[F]) -> F;

    fn to_metric(raw: f64) -> f64 { raw }
}
```

Three implementations: `SquaredEuclidean`, `CosineDistance`, `ManhattanDistance`. Adding a new distance metric is implementing two functions. The `to_metric` default handles the fact that `SquaredEuclidean` needs a `sqrt()` to become actual Euclidean distance (used by HDBSCAN for mutual reachability), while cosine and Manhattan are already in metric space.

Static dispatch means adding cosine distance was a 20-minute feature. No runtime cost, no branches in the hot path, no changes to any existing algorithm. Just a new struct with two functions.

### Layer 2: Algorithm logic

This is where the math lives. Each algorithm gets its own module — `kmeans.rs`, `dbscan.rs`, `hdbscan.rs`, `agglomerative.rs`, `minibatch_kmeans.rs`. They operate on `ndarray::ArrayView2<F>` for input and return typed result structs:

```rust
pub struct KMeansState<F: Scalar> {
    pub centroids: Array2<F>,
    pub centroids_flat: Arc<Vec<F>>,
    pub labels: Vec<usize>,
    pub inertia: f64,
    pub n_iter: usize,
}
```

The algorithms are generic over both the scalar type and the distance metric:

```rust
pub fn run_kmeans_n_init_generic<F: Scalar, D: Distance<F>>(
    data: &ArrayView2<F>,
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
    n_init: usize,
    algo: Algorithm,
) -> Result<KMeansState<F>, ClusterError>
```

This function doesn't know or care whether it's being called from Python or from a Rust benchmark. It doesn't know about PyO3, numpy arrays, or the GIL. It takes an ndarray view, does math, returns a result or an error.

The hot inner loops call down to Layer 3:

```rust
let assignments: Vec<(usize, F)> = (0..n)
    .into_par_iter()
    .map(|i| {
        let point = &data_slice[i * d..(i + 1) * d];
        assign_nearest_with::<F, D>(point, centroids_slice, k, d)
    })
    .collect();
```

Notice the `into_par_iter()`. Rayon handles work-stealing parallelism across cores. The Layer 3 kernel (`assign_nearest_with`) operates on raw slices — it doesn't know it's being called in parallel, and it doesn't need to. `F: Scalar` requires `Send + Sync`, so the compiler enforces thread safety at compile time. No runtime locks, no atomic operations in the hot path.

One design choice worth calling out: **centroid accumulation always happens in f64, even when the input is f32.** When you're summing thousands of floating-point values to compute a cluster mean, f32 accumulation introduces enough error to change convergence behavior. So `recompute_centroids` accumulates in f64 and converts back:

```rust
let mut sums = vec![0.0f64; k * d];
// ... accumulate in f64 ...
centroid_slice[start + j] = F::from_f64_lossy(sums[start + j] / count);
```

This is the kind of thing you only discover by running the f32 test suite and watching two clusters that should converge instead oscillate forever.

### Layer 1: The PyO3 boundary

This is where the complexity budget goes. `lib.rs` handles everything that's Python-specific:

- Extracting numpy arrays as `PyReadonlyArray2<f64>` or `PyReadonlyArray2<f32>`
- Checking C-contiguity (numpy arrays can be Fortran-ordered or strided)
- Releasing the GIL before compute via `py.allow_threads()`
- Converting Rust errors to Python exceptions
- Pickle serialization (`__getstate__`/`__setstate__`)
- The `Metric` enum that maps Python strings to Rust distance types

The f32/f64 dispatch is the most repeated pattern. Every `fit()` method needs to try f64 extraction, then f32, then error. We eventually wrote a macro for it:

```rust
macro_rules! dispatch_fit {
    ($self:expr, $py:expr, $x:expr, $Enum:ident, $run_f64:expr, $run_f32:expr) => {{
        if let Ok(arr) = $x.extract::<PyReadonlyArray2<'_, f64>>() {
            if !arr.is_c_contiguous() {
                return Err(ClusterError::NotContiguous.into());
            }
            let view = arr.as_array();
            let state = $py.allow_threads(move || $run_f64(view))?;
            $self.fitted = Some($Enum::F64(state));
            return Ok(());
        }
        // ... same for f32 ...
    }};
}
```

Now adding a new algorithm's `fit()` is:

```rust
dispatch_fit!(
    self, py, x, FittedState,
    |view| run_kmeans_with_metric(&view, k, max_iter, tol, seed, n_init, algo, metric),
    |view| run_kmeans_with_metric_f32(&view, k, max_iter, tol, seed, n_init, algo, metric)
)
```

The GIL release pattern deserves attention. In Python, the Global Interpreter Lock means only one thread can execute Python bytecode at a time. If your Rust extension holds the GIL during a 10-second compute, you've frozen every other Python thread. `py.allow_threads()` releases the GIL for the duration of the closure, letting other Python threads run while Rust does math. This is critical for async applications and multi-threaded servers.

But there's a subtlety: **you can't touch any Python objects inside `allow_threads`.** The numpy array view is a Rust `ArrayView2` by the time you enter the closure — it's a pointer and shape metadata, no Python objects involved. The borrow checker enforces this at compile time. If you accidentally capture a `Bound<'_, PyAny>` in the closure, the code won't compile. This is one of those cases where Rust's type system is doing real safety work, not just making you type more.

### The error story

Errors cross all three layers, so they need a unified type:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ClusterError {
    #[error("Input array must be C-contiguous float32 or float64")]
    NotContiguous,

    #[error("n_clusters ({k}) must be > 0 and <= n_samples ({n})")]
    InvalidClusters { k: usize, n: usize },

    #[error("Model has not been fitted yet — call fit() first")]
    NotFitted,
    // ... 15 more variants
}
```

At the Python boundary, `ClusterError` converts to the appropriate Python exception:

```rust
impl From<ClusterError> for pyo3::PyErr {
    fn from(err: ClusterError) -> pyo3::PyErr {
        match &err {
            ClusterError::NotFitted => PyRuntimeError::new_err(err.to_string()),
            _ => PyValueError::new_err(err.to_string()),
        }
    }
}
```

Every error message is written for the person who will read it. Not `"invalid parameter"` but `"n_clusters (5) must be > 0 and <= n_samples (3)"`. This matters when agents are calling your API — they need machine-parseable, specific error messages to recover gracefully.

## The embedding pipeline: when generic isn't enough

The five standard algorithms (KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, Agglomerative) all follow the three-layer pattern cleanly. But when we built `EmbeddingCluster` — a purpose-built pipeline for clustering dense embedding vectors — the generic architecture needed extension.

The problem: embedding vectors from OpenAI, Cohere, and Voyage are high-dimensional (768-3072d), live on the unit hypersphere, and benefit from a specific sequence of operations that no general-purpose clustering API provides. The pipeline is:

1. **L2-normalize** every row (embeddings should be unit-norm but often aren't after concatenation or averaging)
2. **Dimensionality reduction** via randomized PCA (Halko-Martinsson-Tropp algorithm) — 1536d → 128d
3. **Spherical K-means** — K-means with cosine similarity on the unit hypersphere
4. **Evaluation** — representative points, intra-cluster similarity, resultant lengths
5. **Optional vMF refinement** — von Mises-Fisher mixture model for soft cluster probabilities

Each stage runs inside its own `py.allow_threads()` with a `py.check_signals()` between stages, so Ctrl+C works even during a 60-second PCA computation.

### The PCA bottleneck and faer

Randomized PCA dominates the pipeline runtime. The core operation is a matrix multiply: `Y = (X - mean) * Omega` where X is (323K x 1536) and Omega is (1536 x 138). That's 69 billion multiply-adds.

Our first implementation used hand-rolled row-parallel loops with rayon. It worked, but was slow — each thread was computing dot products one row at a time, missing out on cache-blocked GEMM optimizations that BLAS libraries have spent decades perfecting.

The fix was [faer](https://github.com/sarah-ek/faer-rs), a pure-Rust linear algebra library with SIMD-accelerated, cache-blocked matrix multiplication. Swapping in faer's GEMM gave us an 11x speedup on the PCA stage:

```rust
let x_centered = Mat::from_fn(n, d, |i, j| data[i * d + j] - mean[j]);
let omega_mat = Mat::from_fn(d, k, |i, j| omega[i * k + j]);
let y_mat = &x_centered * &omega_mat;  // faer GEMM — the 11x speedup
```

We also use faer for the thin SVD of the small (k x d) matrix B, and for the second big matmul `B = Q^T * X_centered`. The QR decomposition uses modified Gram-Schmidt — good enough for k <= ~140, and simpler than faer's full QR.

### Typed state for valid pipelines

`EmbeddingCluster` has a multi-stage pipeline, which means its fitted state is more complex than "centroids + labels." The vMF refinement stage is optional and depends on the clustering stage being complete. The PCA projection needs to persist for `refine_vmf()` to reproject new data.

We initially modeled this with 16 `Option<...>` fields on the struct. It worked, but it meant every getter had to check its own `Option`, and there was no compile-time guarantee that fitting produced all the fields together. You could have `labels = Some(...)` but `centroids = None` — a state that should be impossible but was representable.

The fix was two typed structs:

```rust
struct EmbeddingClusterFitted {
    labels: Vec<usize>,
    centroids: Vec<f64>,
    fitted_d: usize,
    objective: f64,
    n_iter: usize,
    representatives: Vec<usize>,
    intra_similarity: Vec<f64>,
    resultant_lengths: Vec<f64>,
    pca_projection: Option<PcaProjection>,
    // ephemeral cache
    reduced_data: Option<Vec<f64>>,
    reduced_n: usize,
    reduced_d: usize,
}

struct VmfState {
    probabilities: Vec<f64>,
    concentrations: Vec<f64>,
    bic: f64,
}
```

Now the `EmbeddingCluster` struct has `fitted: Option<EmbeddingClusterFitted>` and `vmf: Option<VmfState>`. Three valid states: unfitted (`None`, `None`), fitted (`Some`, `None`), fitted with vMF (`Some`, `Some`). Every getter is `self.fitted.as_ref().ok_or(NotFitted)?.labels` — one line, no possibility of partial state.

## The KD-tree decision

DBSCAN and HDBSCAN need neighbor queries: "find all points within distance eps of point i." The naive approach is O(n^2) — compare every point to every other point. For n=100K, that's 10 billion distance computations.

KD-trees reduce this to O(n log n) average-case by partitioning space along axis-aligned planes. But they have a catch: **KD-trees degrade to O(n) per query in high dimensions.** The "curse of dimensionality" means that above ~16 dimensions, the tree's pruning heuristic almost never fires, and you're doing brute-force search with extra overhead.

Our implementation handles this with a build-time decision:

```rust
pub fn build_v2<F: Scalar>(data: &[F], n: usize, d: usize) -> Option<KdTree> {
    if d > MAX_TREE_DIM || n < MIN_TREE_SIZE {
        return None;  // brute force will be faster
    }
    // ... build tree ...
}
```

`MAX_TREE_DIM = 16`. If the tree can't be built (high-d or Cosine metric, which isn't compatible with axis-aligned partitioning), callers fall back to brute force. The `Option` return type makes this explicit — you can't accidentally use a tree that shouldn't exist.

The tree uses a flat arena layout for cache locality: nodes are stored in a contiguous `Vec<KdNode>` with index-based child pointers, not heap-allocated `Box<KdNode>` with pointer chasing. Leaf nodes store up to 32 points (empirically tuned). Bounding-box pruning uses a `BBoxDistance` trait that mirrors the `Distance` trait but operates on axis-aligned ranges.

## What the benchmarks actually show

We benchmark against scikit-learn's K-means, single-threaded, `n_init=1`, median of 5 runs with varying random seeds:

| n | d | k | Speedup |
|---|---|---|---|
| 1,000 | 8 | 8 | 2.9x |
| 10,000 | 8 | 8 | 2.4x |
| 100,000 | 8 | 32 | 3.2x |
| 100,000 | 32 | 32 | 1.4x |

The speedup decreases with dimensionality. Why? Because as d grows, the distance computation dominates — and sklearn's BLAS-backed distance computation catches up to our auto-vectorized loops. At d=32, both implementations are memory-bandwidth-bound, and the algorithmic overhead that we eliminate (Python interpreter, numpy temporaries) matters less.

This is an honest assessment. We're not 10x faster across the board. We're 1.4-3.2x faster in the regime that matters for our use case (low-to-moderate dimensions, moderate cluster counts), and we're *much* faster for the embedding pipeline where the full-stack integration (PCA + spherical K-means + evaluation in one `fit()` call) eliminates the Python overhead entirely.

The embedding pipeline numbers tell a different story:

| Workflow | Time |
|----------|------|
| Full pipeline (323K x 1536d → 128d, K=98) | 58s |
| Clustering only (cached reduced data) | 7.5s |
| 5 different K values on cached data | 74s |
| Matryoshka (no PCA needed) | ~5s |

That "7.5s for subsequent clustering" is the payoff of `EmbeddingReducer` — fit the PCA once, save the 1.5KB projection matrix, and iterate on clustering for free. In practice, finding the right K takes 5-10 runs. Without the reducer, that's 10 minutes. With it, it's 75 seconds.

## Lessons from the build

**The three-layer pattern is the single best decision we made.** When we added DBSCAN, we touched 4 files and reused 4 others. The neighbor scan went in `dbscan.rs` (Layer 2), the distance computation reused `utils.rs` (Layer 3), the PyO3 bindings went in `lib.rs` (Layer 1), and the error types went in `error.rs`. HDBSCAN, Agglomerative, and MiniBatchKMeans followed the same pattern.

**Don't abstract until you have two concrete implementations.** We wrote KMeans and DBSCAN before extracting the `Distance` trait. If we'd designed the trait first, we would have gotten it wrong — we wouldn't have known about `to_metric()` until HDBSCAN needed it for mutual reachability distances.

**f32 support is a feature, not a dtype.** Most clustering libraries silently upcast f32 to f64. This doubles memory consumption and halves cache efficiency. For embedding workloads (323K vectors x 1536 dimensions), the difference between f32 and f64 is 1.9 GB vs 3.8 GB of working memory. We keep the input dtype through the entire pipeline, with f64 accumulation only where numerical stability requires it.

**The PyO3 boundary is where you spend your complexity budget.** The algorithms are straightforward — K-means is well-understood, DBSCAN is a BFS, HDBSCAN is Prim's MST plus Union-Find. The hard part is making them work correctly and ergonomically across the Python-Rust boundary: dtype dispatch, pickle serialization, GIL management, error mapping, contiguity checks. This is where bugs hide and where maintenance cost accumulates.

**Auto-vectorization is enough.** We never wrote a single line of `unsafe` SIMD intrinsics. Simple counted loops over contiguous slices, compiled with `opt-level=3` and `lto="fat"`, generate SIMD code that's within 10% of hand-tuned intrinsics. The `#[inline(always)]` on hot kernels lets the compiler inline through function boundaries, enabling vectorization across the call stack.

## The release profile

One more Rust detail worth mentioning. Our `Cargo.toml` release profile:

```toml
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
strip = "debuginfo"
```

`lto = "fat"` enables link-time optimization across all crate boundaries — including into ndarray, rayon, and faer. This is critical because our hot kernels are generic functions that get monomorphized at call sites in other modules. Without LTO, the compiler can't inline across crate boundaries, and auto-vectorization stops at function calls.

`codegen-units = 1` forces the compiler to process the entire crate as a single compilation unit, enabling more aggressive inlining and optimization. The tradeoff is slower compile times (about 2x), which doesn't matter for release builds.

Together, these settings account for roughly 15-20% of our total performance. Free speed for two lines of config.

## Where we are now

rustcluster 0.4.0 ships six algorithms, three distance metrics, KD-tree acceleration, a purpose-built embedding pipeline with faer-accelerated PCA, von Mises-Fisher soft clustering, pickle serialization for every model, and 421 tests across Rust and Python. It installs with `pip install rustcluster` on Linux, macOS, and Windows across Python 3.9-3.13.

The architecture has held up. Adding an algorithm touches 4 files. Adding a distance metric touches 2. The three-layer pattern scales because each layer's contract is simple: Layer 3 takes slices and returns numbers, Layer 2 takes arrays and returns structs, Layer 1 takes Python objects and returns Python objects.

If you're building a Rust-backed Python library, the temptation is to optimize the kernel first. Don't. Design the boundary first. The kernel will be fast enough if the boundary is clean enough.

```bash
pip install rustcluster
```

*rustcluster is open source under the MIT license. [GitHub](https://github.com/mfbaig35r/rustcluster) | [PyPI](https://pypi.org/project/rustcluster/)*
