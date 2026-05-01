# The Bar Review: rustcluster

## 1. The Conversation

**Samuel** *(swirls an Old Fashioned)*: Alright, who wants to start? Because I have opinions about `lib.rs`.

**Michael**: I was about to say — this is a genuinely well-built library. Six clustering algorithms, Rust-backed, GIL-released, f32/f64 support, proper error types. Someone put real work into this.

**Charlie**: Agreed on the build quality. But `lib.rs` is 2,089 lines of PyO3 bindings and it's screaming for extraction. Every algorithm follows the exact same pattern — `FittedState` enum, `fit`/`predict`/`fit_predict`, `__getstate__`/`__setstate__`, `__repr__`, `__getnewargs__`. I count five copies of the `f64/f32` dtype dispatch pattern in `fit()` alone.

**Samuel**: Exactly my complaint. The `EmbeddingCluster` struct at line 1455 has *sixteen fields*, all `Option<...>`. That's not a typed state machine, that's a bag of maybes. Compare it to `KMeans`, which at least wraps its fitted state in a proper `FittedState` enum. `EmbeddingCluster` just has `labels: Option<Vec<usize>>`, `centroids: Option<Vec<f64>>`, `vmf_probabilities: Option<Vec<f64>>`... You get the wrong `None` and you have no idea which stage failed.

**Sebastián**: And the Python side mirrors this — `experimental.py:65`, `self._X = None`. The `EmbeddingCluster` Python wrapper *caches the entire training array* on `self._X` just so `refine_vmf()` can use it later. That's a silent memory doubling on every fit call. For a library marketed at embedding workloads, those are big arrays.

**Michael**: That would absolutely confuse someone reading the code for the first time. "Why is the training data stored on the object?" It's not documented, there's no `del self._X` path, and if someone pickles this object — oh wait, there's no `__getstate__` on `EmbeddingCluster` at all.

**Samuel**: *(leans forward)* That's a blocker. `EmbeddingCluster` and `EmbeddingReducer` in `experimental.py` have no pickle support. Every other model does. An agent workflow that tries to serialize a fitted `EmbeddingCluster` will silently get a default pickle that includes `self._X` — the entire training dataset — in the pickle stream. Or it'll just fail.

**Charlie**: While we're on `experimental.py` — there's a duplicated `_prepare()` function at line 14 that does the exact same thing as `_prepare_array()` in `__init__.py`. Just import it.

**Sebastián**: Naming, while I'm looking: the Rust class is `Dbscan` but the Python wrapper is `DBSCAN`. Same for `Hdbscan`/`HDBSCAN`. That's actually fine — the Python names match sklearn convention. But the import at `__init__.py:11` exposes this: `from rustcluster._rustcluster import Dbscan as _RustDbscan`. Good aliasing. No complaint there.

**Samuel**: Back to types. `Metric::from_str` and `Algorithm::from_str` in `distance.rs:119` and `kmeans.rs:31` shadow the standard library's `FromStr` trait. They should implement `FromStr` properly. Same for `ClusterSelectionMethod::from_str` and `Linkage::from_str`. This is a Rust API — other Rust code consuming this library can't use `.parse::<Metric>()`.

**Charlie**: The `predict` methods in `lib.rs` clone the entire centroids array every time — line 157, `let centroids = state.centroids.clone()`. That's `k * d * 8` bytes allocated and copied on every single predict call, just to move it into the `allow_threads` closure. You could use `as_slice()` with a lifetime trick, or restructure to avoid the clone.

**Sebastián**: The metrics functions at `__init__.py:572` convert labels to a Python list via `.tolist()` before passing them to Rust. That's `labels = np.asarray(labels, dtype=np.int64).tolist()`. You're converting a numpy array to a Python list of Python ints, then PyO3 converts it back to `Vec<i64>`. Just accept `PyReadonlyArray1<i64>` on the Rust side.

**Samuel**: The `HDBSCAN.__setstate__` in `lib.rs:844` always deserializes into `HdbscanFitted::F64` regardless of what dtype was used during `fit()`. Comment on line 843 says "HDBSCAN state is dtype-independent (all f64)." That's true for labels/probabilities, but the `PhantomData<F>` still carries the type. If someone fits with f32 and unpickles, the internal enum variant will be F64. Is that a problem?

**Michael**: Not functionally — HDBSCAN has no `predict`, so the fitted state is only used for getters. But it's a code smell that would trip up anyone adding `predict` later. And the same pattern appears for `AgglomerativeClustering` at line 1066.

**Charlie**: `hdbscan.rs:284` — the core distance computation does a full sort of all n distances per point. That's O(n log n) per point when you only need the k-th smallest. A partial sort (`select_nth_unstable`) would be O(n) per point.

**Samuel**: Good catch. And `hdbscan.rs:411` uses a `HashMap<usize, usize>` for `root_to_node` inside the condense tree builder. The keys are always in `0..n+n_merges`. A `Vec<usize>` would be faster and clearer.

**Charlie**: The `_node_size` parameter in `assign_subtree_points` at `hdbscan.rs:614` is prefixed with underscore because it's unused. Just remove it from the signature.

**Sebastián**: One more on the API surface — `EmbeddingReducer.load()` in `experimental.py:267` does `obj._target_dim = rust_model.target_dim if hasattr(rust_model, 'target_dim') else None` and `obj._method = None`. So after loading, `_method` is always `None` and `_target_dim` might be `None`. The `__repr__` will print `method='None'`. Not great for debugging.

**Michael**: Overall though — the test coverage is solid. 416 tests across Rust and Python. The error types are well-designed and map correctly to Python exceptions. The three-layer architecture is clean. The README exists. This is better than most production clustering libraries I've seen.

**Samuel**: It ships. It just needs the EmbeddingCluster state machine cleaned up and the serialization gap closed.

---

## 2. Findings, Grouped by Theme

### Serialization & State Management

- **`[blocker]`** `EmbeddingCluster` (Python wrapper) has no `__getstate__`/`__setstate__`. Pickling will either fail or silently include the cached training array (`self._X`), ballooning the serialized size. Every other model in the library supports pickle.

- **`[serious]`** `EmbeddingCluster` caches `self._X` (the full training array) with no cleanup path. For embedding workloads (1536-d, 100K+ rows), this silently doubles memory. The cache exists solely for `refine_vmf()`.

- **`[nit]`** `HDBSCAN.__setstate__` and `AgglomerativeClustering.__setstate__` always deserialize into the F64 variant regardless of original dtype. Functionally harmless today but breaks the type contract if `predict()` is ever added.

### Code Duplication

- **`[serious]`** `lib.rs` has ~800 lines of near-identical PyO3 boilerplate across 7 classes (dtype dispatch in `fit`, `predict`, getstate/setstate). A macro or generic wrapper could eliminate most of it.

- **`[nit]`** `_prepare()` in `experimental.py:14` is an exact duplicate of `_prepare_array()` in `__init__.py:32`. Import from one location.

### Performance

- **`[serious]`** `hdbscan.rs:284` — core distance computation sorts all n distances per point (O(n log n)) when only the k-th smallest is needed. `select_nth_unstable` gives O(n) per point.

- **`[serious]`** `predict()` in `lib.rs` clones the centroids array on every call to satisfy the `allow_threads` borrow. For hot-path predict calls (agent loops), this is avoidable.

- **`[nit]`** `__init__.py:572` — metrics convert labels via `.tolist()` (numpy → Python list → Rust Vec). Accept `PyReadonlyArray1<i64>` directly.

### Rust Idioms

- **`[nit]`** `Metric::from_str`, `Algorithm::from_str`, `ClusterSelectionMethod::from_str`, `Linkage::from_str` shadow the standard `FromStr` trait. Implementing `FromStr` would let downstream Rust code use `.parse()`.

- **`[nit]`** `hdbscan.rs` uses `HashMap` where index-addressed `Vec` would suffice (`root_to_node`, `cluster_stability`, `cluster_lambda_birth`, etc.). Minor perf impact but would simplify the code.

- **`[nit]`** `assign_subtree_points` takes an unused `_node_size` parameter.

### API & Usability

- **`[nit]`** `EmbeddingReducer.load()` sets `_method = None`, causing `__repr__` to print `method='None'`.

- **`[nit]`** `EmbeddingCluster` (Rust) stores all fitted state as individual `Option` fields (16 of them) rather than a typed `EmbeddingClusterState` struct. Makes it impossible to reason about which combinations of fields are valid.

---

## 3. Concrete Fixes

1. **Add pickle support to `EmbeddingCluster`** — Implement `__getstate__`/`__setstate__` on the Rust `EmbeddingCluster` class in `lib.rs:1454-1850`, mirroring the pattern used by `KMeans`. Do *not* serialize the training data. **File:** `src/lib.rs`, `python/rustcluster/experimental.py`. **Why:** Agents and pipelines that serialize fitted models will break silently. **Effort:** Moderate.

2. **Remove `self._X` caching from `EmbeddingCluster`** — The Python wrapper already re-extracts data in `refine_vmf()` on the Rust side (line 1681). Make `refine_vmf` accept `X` as a parameter on the Python side (it already does on Rust side). Delete `self._X`. **File:** `python/rustcluster/experimental.py:65,79-80,94-96`. **Why:** Silent memory doubling. **Effort:** Trivial. *(Note: it already works this way — `self._X` is redundant since the Rust `refine_vmf` takes `x` directly.)*

3. **Use `select_nth_unstable` for core distances** — Replace the full sort at `hdbscan.rs:284` with `dists.select_nth_unstable_by(k, ...)` to get the k-th element in O(n). **File:** `src/hdbscan.rs:284`. **Why:** Turns O(n² log n) into O(n²) for the core distance stage. **Effort:** Trivial.

4. **Deduplicate `_prepare` functions** — Delete `_prepare()` from `experimental.py` and import `_prepare_array` from `rustcluster`. **File:** `python/rustcluster/experimental.py:14-19`. **Why:** Maintenance hazard. **Effort:** Trivial.

5. **Fix `EmbeddingReducer.load()` metadata** — Read `method` and `target_dim` from the loaded Rust state instead of hardcoding `None`. **File:** `python/rustcluster/experimental.py:264-268`. **Why:** `__repr__` prints nonsense after load. **Effort:** Trivial.

6. **Consolidate `EmbeddingCluster` fitted state into a struct** — Replace the 16 `Option` fields with an `Option<EmbeddingClusterState>` that holds all fitted data. **File:** `src/lib.rs:1454-1530`. **Why:** Eliminates impossible state combinations, makes the code readable. **Effort:** Moderate.

7. **Avoid centroid clone in `predict`** — Use `unsafe` to extend the borrow lifetime through `allow_threads`, or restructure to pass a slice reference. Alternatively, store centroids as a flat `Vec<F>` alongside the `Array2` to avoid the clone. **File:** `src/lib.rs:157,198,1198,1244`. **Why:** Allocation on every predict call in a hot loop. **Effort:** Moderate.

8. **Accept `PyReadonlyArray1<i64>` for metric labels** — Change the Rust metric functions to accept array references instead of `Vec<i64>`. **File:** `src/lib.rs:2003-2069`. **Why:** Eliminates a Python list round-trip. **Effort:** Trivial.

---

## 4. Verdict

**Ship it** — with fix #1 (pickle support) before any agent workflow touches `EmbeddingCluster`.

**Top 3 changes that would earn their respect:**
1. Add `EmbeddingCluster` pickle support and drop `self._X` caching (fixes the only blocker and the biggest memory footprint issue)
2. `select_nth_unstable` in HDBSCAN core distances (trivial change, meaningful perf win on the O(n²) path)
3. Consolidate the 16 `Option` fields into a typed `EmbeddingClusterState` struct (structural correctness that unlocks safe future development)

**One thing the code gets right:** The three-layer kernel design — PyO3 boundary, algorithm logic, hot kernels on raw slices — is textbook. GIL release at every compute boundary, f64 accumulation for precision with F-typed output, and the generic `Distance<F>` trait enabling static dispatch without runtime overhead. This is how you write a Rust extension for Python.
