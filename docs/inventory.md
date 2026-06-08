# rustcluster v0.7.0: capability inventory

*Snapshot as of 2026-06-08.*

Single Rust-backed Python wheel (`pip install rustcluster`). PyO3 + maturin. f32/f64 dtype-aware. GIL released across all compute.

## Top-level public surface

```python
from rustcluster import (
    KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering,
    ClusterSnapshot,
    IndexFlatL2, IndexFlatIP,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    index,      # submodule
    utils,      # submodule
)
from rustcluster.experimental import EmbeddingCluster, EmbeddingReducer, HierarchicalSnapshot
from rustcluster.utils import extract_embeddings_from_spark
```

## 1. Clustering algorithms

All accept `metric=` (euclidean / cosine / manhattan unless noted). All support pickle.

| Algorithm | Constructor signature | Fitted attributes |
|---|---|---|
| **`KMeans`** | `(n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10, algorithm="auto", metric="euclidean")` | `labels_`, `cluster_centers_`, `inertia_`, `n_iter_` |
| **`MiniBatchKMeans`** | `(n_clusters, batch_size=1024, max_iter=100, tol=0.0, random_state=0, max_no_improvement=10, metric="euclidean")` | `labels_`, `cluster_centers_`, `inertia_`, `n_iter_` |
| **`DBSCAN`** | `(eps=0.5, min_samples=5, metric="euclidean")` | `labels_` (-1 = noise), `core_sample_indices_` |
| **`HDBSCAN`** | `(min_cluster_size=5, min_samples=None, metric="euclidean", cluster_selection_method="eom")` | `labels_` (-1 = noise), `probabilities_`, `cluster_persistence_` |
| **`AgglomerativeClustering`** | `(n_clusters=None, linkage="ward", metric="euclidean", distance_threshold=None)` | `labels_`, `n_clusters_`, `children_`, `distances_` |
| **`EmbeddingCluster`** (experimental) | `(n_clusters=50, reduction_dim=128, max_iter=100, tol=1e-6, random_state=0, n_init=5, reduction="pca")` | `labels_`, `cluster_centers_`, `objective_`, `n_iter_`, `representatives_`, `intra_similarity_`, `resultant_lengths_`, `reduced_data_`, optional vMF: `probabilities_`, `concentrations_`, `bic_` |

**Methods on every fitted clustering object:** `.fit(X)`, `.snapshot()` (where applicable), `.__getstate__/__setstate__` (pickle).

**KMeans algorithms:** Lloyd (general), Hamerly (Euclidean only, auto-selected). Non-euclidean metrics force Lloyd.

**Agglomerative phases:** condensed dtype-aware distance matrix; NN-chain merge engine (Müllner 2011, arXiv:1109.2378); sklearn-compatible `distance_threshold` (strict-`<` cut). Memory at n=107K: ~322 GB to ~23 GB vs naive.

**HDBSCAN dtype quirk** (documented in code): state is dtype-independent; `PhantomData<F>` is cosmetic.

## 2. Vector index module (`rustcluster.index`)

FAISS-flavored API, **f32-only**.

| Type | Methods |
|---|---|
| `IndexFlatL2(dim)` | `.add(vectors)`, `.add_with_ids(vectors, ids: u64)`, `.search(queries, k, exclude_self=False)`, `.range_search(queries, threshold, exclude_self=False)`, `.similarity_graph(threshold, unique_pairs=False)`, `.save(path)`, `IndexFlatL2.load(path)`, `.dim`, `.ntotal`, `.metric` |
| `IndexFlatIP(dim)` | same shape, max-dot semantics. For cosine, normalize on `add`. |

**Returns:**
- `search()`: `(distances: ndarray (nq, k), labels: ndarray (nq, k))`. `-1` labels + sentinel distance pad short results.
- `range_search()`: FAISS-shape `(lims, distances, labels)` CSR.
- `similarity_graph()`: `(src, dst, scores)` parallel `u64`/`u64`/`f32` arrays. Cache-blocked tile iteration over the upper triangle.

**Persistence**: directory with `vectors.safetensors` + optional `ids.safetensors` + `metadata.json`.

## 3. Cluster snapshots (incremental assignment)

```python
snapshot = model.snapshot()                  # from KMeans, MiniBatchKMeans, or EmbeddingCluster
labels = snapshot.assign(X_new)              # 100x faster than refitting
snapshot.calibrate(X_train)                  # enables v2 features below
result = snapshot.assign_with_scores(X_new, distance_threshold=..., confidence_threshold=...,
                                     adaptive_threshold=True, adaptive_percentile="p10",
                                     boundary_mode="voronoi"|"mahalanobis")
report = snapshot.drift_report(X_recent)
snapshot.save(dir) / ClusterSnapshot.load(dir)
```

`AssignmentResult` fields: `labels_` (-1 if rejected), `confidences_` in [0,1), `distances_`, `rejected_` (bool mask).

`DriftReport` fields: `n_samples_`, `global_mean_distance_`, `relative_drift_` (per-cluster), `new_cluster_sizes_`, `new_mean_distances_`, `rejection_rate_` (**NaN until `calibrate()`**), `kappa_drift_` and `direction_drift_` (spherical + calibrated only).

**Read-only properties:** `.k`, `.d`, `.input_dim`, `.algorithm`, `.metric`, `.spherical`, `.is_calibrated`.

**`boundary_mode="mahalanobis"`** uses per-cluster diagonal variances from `calibrate()` for elongated clusters.

**Persistence**: safetensors (centroids) + JSON (metadata). v2 adds optional `fit_distance_mean/std`, `fit_kappa`, `cluster_variances`, `confidence_stats`. v1 snapshots load cleanly.

## 4. Hierarchical snapshots (`rustcluster.experimental.HierarchicalSnapshot`)

Cascading two-level slotting (e.g. commodity, then sub-commodity).

```python
hier = HierarchicalSnapshot.build(X_train, root_model, n_sub_clusters=5, **fit_kwargs)
result = hier.assign(X_new) / .assign_with_scores(X_new, ...)
hier.save(dir) / HierarchicalSnapshot.load(dir)
```

Returns `HierarchicalAssignmentResult` with `labels_` (tuples), `confidences_`, `rejected_`. Properties: `.k_root`, `.n_children`.

## 5. Embedding pipeline (`rustcluster.experimental`)

### `EmbeddingCluster`

All-in-one: L2-normalize, PCA, spherical K-means, optional vMF refinement.

- `.fit(X)`, `.refine_vmf(X)` (soft probabilities + BIC), `.snapshot()`
- `reduction="pca"|"matryoshka"|"none"`, `reduction_dim=128` (or `None`)
- Returns f32 or f64 output matching input dtype (since v0.7.0)

### `EmbeddingReducer`

Standalone PCA reducer; fit once, transform forever.

```python
EmbeddingReducer(target_dim=128, method="pca", random_state=0, fit_sample_size=None)
.fit(X)
.transform(X, chunk_size=None)
.fit_transform(X, chunk_size=None)
.save(path)  # RCPC binary, ~1.5 KB
EmbeddingReducer.load(path)
```

**v0.7.0 memory features:**
- `chunk_size=N`: process input in N-row blocks
- `fit_sample_size=N`: random subsample for PCA fit (uses `random_state`)
- f32 fast path: f32 input gives f32 hot path (centered matrix stays f32) gives f32 output

**Dtype contract:** input dtype determines output dtype. Storage on disk is always f64 (canonical, ~750 KB at d=1536/target=128).

## 6. Utilities (`rustcluster.utils`)

```python
extract_embeddings_from_spark(
    df,
    embedding_col,
    metadata_cols=None,
    dtype=np.float32,
    sample_n=None,
    seed=0,
) -> (ndarray (n, d), pandas.DataFrame | None)
```

Streams a Spark DataFrame via `toLocalIterator()`. No Python list-of-lists overhead. pyspark + pandas are lazy-imported (clear `ImportError` if missing).

## 7. Evaluation metrics (module level)

| Function | Range | Direction |
|---|---|---|
| `silhouette_score(X, labels)` | [-1, 1] | higher better |
| `calinski_harabasz_score(X, labels)` | (0, ∞) | higher better |
| `davies_bouldin_score(X, labels)` | (0, ∞) | lower better |

## 8. Distance metrics

| Metric | Aliases | KD-tree | Notes |
|---|---|---|---|
| `"euclidean"` | `"l2"` | yes | default |
| `"cosine"` | | no | KMeans forces Lloyd (Hamerly assumes Euclidean) |
| `"manhattan"` | `"cityblock"`, `"l1"` | yes | |

Ward linkage requires euclidean.

## 9. Persistence formats

| Object | Format | Notes |
|---|---|---|
| Any fitted model | Python pickle | `__getstate__/__setstate__` on every class |
| `EmbeddingReducer` | RCPC binary (custom) | ~1.5 KB; f64 storage regardless of fit dtype |
| `ClusterSnapshot` | directory: safetensors + `metadata.json` | ~50 KB for 50-cluster, 128d. v1 forward-compatible. |
| `IndexFlatL2/IP` | directory: safetensors (vectors + optional ids) + `metadata.json` | `format_version=1` |
| `HierarchicalSnapshot` | directory tree of snapshots | one safetensors per child |

## 10. Source layout

```
src/
├── lib.rs               PyO3 boundary (~3,400 lines)
├── distance.rs          Distance traits + Scalar
├── kmeans.rs            Lloyd + KMeans++ init
├── hamerly.rs           Hamerly acceleration (euclidean only)
├── minibatch_kmeans.rs
├── dbscan.rs
├── hdbscan.rs
├── agglomerative.rs     NN-chain, condensed matrix, distance_threshold
├── kdtree.rs            DBSCAN/HDBSCAN neighbor pruning (d <= 16)
├── metrics.rs           silhouette / CH / DB scores
├── utils.rs             Hot-loop helpers
├── error.rs             ClusterError enum
├── snapshot/            mod.rs + assign + drift + calibrate + mahalanobis
├── snapshot_io.rs       Snapshot safetensors I/O
├── index/               mod.rs + flat + ids + kernel + topk + persistence + similarity_graph
└── embedding/           mod.rs + spherical_kmeans + reduction + reducer + normalize + vmf + evaluation + fusion
```

## 11. Notable contracts / gotchas worth knowing

- **`rejection_rate_` is NaN until `snapshot.calibrate(X_train)`** is called. Per-cluster bounds come from calibration, not fit-time.
- **`EmbeddingReducer.transform` preserves input dtype** as of v0.7.0. Pre-v0.7.0 it always upcast to f64. Add `.astype(np.float64)` if you needed the upcast.
- **Native f32**: no silent upcast in clustering hot paths.
- **GIL released** during all `.fit()`, `.transform()`, `.assign()` calls.
- **PCA save format** stays f64 even when fit ran in f32 (storage is ~750 KB; not worth migrating).
- **KMeans cosine** forces Lloyd; Hamerly is Euclidean-only.
- **Ward linkage** requires Euclidean.
- **HDBSCAN has no `predict()`** by design; the snapshot path is `KMeans`/`MiniBatchKMeans`/`EmbeddingCluster` only.
- **Windows wheel cross-compile** needs `pyo3/generate-import-lib` (already enabled in the `python` feature).

## 12. Test surface (as of v0.7.0)

- 237 Rust unit tests (`cargo test --no-default-features --lib`)
- 394 Python tests + 6 perf opt-in (`pytest -m perf`)

## 13. Related docs in this repo

- `README.md`: install, quickstart, headline performance numbers
- `docs/embedding-clustering-guide.md`: end-to-end user guide for embedding workflows
- `docs/architecture-decisions.md`: rationale for the three-layer kernel pattern, faer choice, etc.
- `docs/lessons-building-rustcluster.md`: build narrative
- `docs/embedding-cluster-whitepaper.md`: algorithm details for EmbeddingCluster
- `docs/memory-optimization-requirements.md`: P2/P3 future work (in-place transform, streaming PCA fit, Spark UDF)
- `docs/blog-v1.2-retrospective-draft.md`: the eleven-week SIMD experiment that didn't beat faer
- `CONTRIBUTING.md`: how to add an algorithm, metric, or test
