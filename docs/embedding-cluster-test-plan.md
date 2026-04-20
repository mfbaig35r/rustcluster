# EmbeddingCluster Experimental Validation Plan

Test plan for populating Section 5 of the white paper. Each experiment is designed to answer a specific question, with exact methodology, datasets, metrics, and baselines defined upfront.

---

## Experiment 1: Cluster Recovery on Synthetic vMF Mixtures

**Question:** How accurately does EmbeddingCluster recover known cluster structure across varying dimensionality, scale, and cluster count?

**Data Generation:**
- Generate K clusters from vMF distributions with known μ_k, κ_k
- Concentrations: κ ∈ {10, 50, 100, 200} (loose → tight)
- Mix with uniform noise: 5% of points drawn uniformly on S^{d-1}

**Configurations:**

| Parameter | Values |
|-----------|--------|
| d (dimensions) | 64, 128, 256, 512, 1536 |
| n (samples) | 10K, 50K, 100K, 500K |
| K (clusters) | 5, 20, 50, 100 |
| κ (concentration) | 10, 50, 100, 200 |

**Pipeline configurations tested:**
1. EmbeddingCluster (no PCA)
2. EmbeddingCluster (PCA → 128)
3. EmbeddingCluster (PCA → 32)

**Metrics:**
- Adjusted Rand Index (ARI) vs ground-truth labels
- Normalized Mutual Information (NMI) vs ground-truth
- Cluster purity (fraction of dominant true label per predicted cluster)
- Number of recovered clusters (predicted K vs true K)

**Success criteria:**
- ARI > 0.95 for κ ≥ 50 at all d and n
- ARI > 0.80 for κ = 10 (loose clusters) at d ≤ 256
- Graceful degradation as d increases (purity should decrease monotonically with d at fixed κ)
- PCA → 128 should match no-PCA quality within 2% ARI for d ≤ 1536

**Implementation:**

```python
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from rustcluster.experimental import EmbeddingCluster

def generate_vmf_mixture(n, d, k, kappa, noise_frac=0.05, seed=42):
    """Generate synthetic vMF mixture with known labels."""
    rng = np.random.default_rng(seed)
    n_per = int(n * (1 - noise_frac)) // k
    n_noise = n - n_per * k

    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    data, labels = [], []
    for i in range(k):
        # Sample from vMF(centers[i], kappa)
        # Approximate: center + noise scaled by 1/sqrt(kappa), then normalize
        noise = rng.standard_normal((n_per, d)) / np.sqrt(kappa)
        points = centers[i] + noise
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        data.append(points)
        labels.extend([i] * n_per)

    # Uniform noise on sphere
    if n_noise > 0:
        noise_points = rng.standard_normal((n_noise, d))
        noise_points /= np.linalg.norm(noise_points, axis=1, keepdims=True)
        data.append(noise_points)
        labels.extend([-1] * n_noise)

    return np.vstack(data).astype(np.float32), np.array(labels)

def run_experiment_1():
    results = []
    for d in [64, 128, 256, 512, 1536]:
        for n in [10_000, 50_000, 100_000]:
            for k in [5, 20, 50]:
                for kappa in [10, 50, 100, 200]:
                    X, true_labels = generate_vmf_mixture(n, d, k, kappa)
                    valid_mask = true_labels >= 0

                    for reduction_dim, label in [(None, "none"), (128, "pca128"), (32, "pca32")]:
                        if reduction_dim and reduction_dim >= d:
                            continue
                        model = EmbeddingCluster(
                            n_clusters=k,
                            reduction_dim=reduction_dim,
                            random_state=42,
                            n_init=3,
                        )
                        model.fit(X)
                        pred = np.array(model.labels_)

                        ari = adjusted_rand_score(true_labels[valid_mask], pred[valid_mask])
                        nmi = normalized_mutual_info_score(true_labels[valid_mask], pred[valid_mask])

                        results.append({
                            "d": d, "n": n, "k": k, "kappa": kappa,
                            "reduction": label, "ari": ari, "nmi": nmi,
                        })
    return results
```

---

## Experiment 2: Comparison Against Baselines

**Question:** How does EmbeddingCluster compare to existing systems on real text embeddings?

**Datasets:**
1. **20 Newsgroups** — 18,846 documents, 20 classes. Embed with text-embedding-3-small (1536d).
2. **AG News** — 120,000 documents, 4 classes. Embed with text-embedding-3-small.
3. **Banking77** — 13,083 customer queries, 77 intents. Embed with text-embedding-3-small.

**Baselines:**
1. **sklearn KMeans** — `KMeans(n_clusters=K, n_init=10)` on raw embeddings
2. **sklearn KMeans + cosine** — `KMeans` on L2-normalized embeddings (pseudo-cosine)
3. **BERTopic** — default pipeline (`UMAP(n_components=5) → HDBSCAN → c-TF-IDF`)
4. **FAISS KMeans** — `faiss.Kmeans(d, K, spherical=True)` on GPU if available
5. **EmbeddingCluster (no PCA)** — `EmbeddingCluster(n_clusters=K, reduction_dim=None)`
6. **EmbeddingCluster (PCA→128)** — `EmbeddingCluster(n_clusters=K, reduction_dim=128)`
7. **EmbeddingCluster (PCA→128 + vMF)** — with `refine_vmf()`

**Methodology:**
- Set K = ground-truth number of classes for all centroidal methods
- For BERTopic: let HDBSCAN determine K, then report discovered K
- Run 3 times with different seeds, report median
- Measure on the same machine, single-threaded where possible

**Metrics:**
- ARI vs ground-truth labels
- NMI vs ground-truth labels
- Cluster purity
- Wall-clock time (total pipeline: embed → reduce → cluster)
- Peak memory (RSS at max)
- Number of clusters discovered (for BERTopic)

**Embedding generation:**
- Pre-compute embeddings using OpenAI API (`text-embedding-3-small`)
- Cache as `.npy` files (f32)
- Embedding generation time excluded from benchmark (same for all methods)

**Success criteria:**
- ARI within 5% of best baseline on each dataset
- Wall-clock time 5-50× faster than BERTopic
- Peak memory < 500 MB for 120K embeddings (AG News)
- sklearn should OOM or be >10× slower at 120K × 1536

**Implementation:**

```python
import time
import tracemalloc
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans
from rustcluster.experimental import EmbeddingCluster

def benchmark_system(name, fit_fn, X, true_labels, runs=3):
    """Benchmark a clustering system."""
    results = []
    for seed in range(runs):
        tracemalloc.start()
        t0 = time.perf_counter()
        labels = fit_fn(X, seed)
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ari = adjusted_rand_score(true_labels, labels)
        nmi = normalized_mutual_info_score(true_labels, labels)
        results.append({"time": elapsed, "ari": ari, "nmi": nmi, "peak_mb": peak_mem / 1e6})

    median_idx = len(results) // 2
    results.sort(key=lambda r: r["time"])
    return {"name": name, **results[median_idx]}

def run_experiment_2(dataset_name, X, true_labels, k):
    benchmarks = []

    # sklearn KMeans
    benchmarks.append(benchmark_system(
        "sklearn_kmeans",
        lambda X, s: SklearnKMeans(n_clusters=k, n_init=3, random_state=s).fit(X).labels_,
        X.astype(np.float64), true_labels
    ))

    # EmbeddingCluster (no PCA)
    benchmarks.append(benchmark_system(
        "embedding_cluster_raw",
        lambda X, s: np.array(EmbeddingCluster(n_clusters=k, reduction_dim=None, random_state=s, n_init=3).fit(X).labels_),
        X, true_labels
    ))

    # EmbeddingCluster (PCA→128)
    benchmarks.append(benchmark_system(
        "embedding_cluster_pca128",
        lambda X, s: np.array(EmbeddingCluster(n_clusters=k, reduction_dim=128, random_state=s, n_init=3).fit(X).labels_),
        X, true_labels
    ))

    return benchmarks
```

---

## Experiment 3: Wall-Clock Scaling

**Question:** How does EmbeddingCluster runtime scale with n, and how does it compare to BERTopic at scale?

**Setup:**
- Fixed: d=1536, K=50, PCA→128
- Vary n: 10K, 25K, 50K, 100K, 250K, 500K
- Data: synthetic vMF mixture (κ=100) at each scale
- Single-threaded (RAYON_NUM_THREADS=1, OMP_NUM_THREADS=1)
- Measure total pipeline time (normalize + PCA + cluster)

**Systems:**
1. EmbeddingCluster (PCA→128, n_init=3)
2. EmbeddingCluster (PCA→32, n_init=3)
3. BERTopic (default UMAP→HDBSCAN, if installable)
4. sklearn KMeans (n_init=3) — expect OOM at 250K+

**Expected scaling:**
- EmbeddingCluster: O(n·d·k_pca) for PCA + O(n·k_pca·K·iters) for K-means
  - At d=1536, k_pca=128, K=50, 20 iters, n=500K: PCA ~5s + K-means ~30s ≈ 35s
- BERTopic: O(n²) for UMAP graph construction — should be orders of magnitude slower at 500K

**Output:**

| n | EmbeddingCluster (PCA→128) | EmbeddingCluster (PCA→32) | BERTopic | sklearn |
|---|---|---|---|---|
| 10K | | | | |
| 25K | | | | |
| 50K | | | | |
| 100K | | | | |
| 250K | | | | OOM |
| 500K | | | | OOM |

**Target:** EmbeddingCluster (PCA→128) under 60 seconds at 500K embeddings.

**Implementation:**

```python
import time
import numpy as np
from rustcluster.experimental import EmbeddingCluster

def run_experiment_3():
    results = []
    d = 1536
    k = 50
    kappa = 100

    for n in [10_000, 25_000, 50_000, 100_000, 250_000, 500_000]:
        X, _ = generate_vmf_mixture(n, d, k, kappa)

        for reduction_dim, label in [(128, "pca128"), (32, "pca32")]:
            times = []
            for run in range(3):
                t0 = time.perf_counter()
                model = EmbeddingCluster(
                    n_clusters=k, reduction_dim=reduction_dim,
                    random_state=run, n_init=1,
                )
                model.fit(X)
                times.append(time.perf_counter() - t0)
            median_time = sorted(times)[1]
            results.append({"n": n, "reduction": label, "time_s": median_time})
            print(f"n={n:>7,} {label:>8}: {median_time:.2f}s")

    return results
```

---

## Experiment 4: PCA Ablation

**Question:** What is the optimal PCA target dimension for embedding clustering quality?

**Setup:**
- Dataset: 20 Newsgroups embeddings (18,846 × 1536)
- K = 20 (ground truth)
- Vary reduction_dim: None, 256, 128, 64, 32, 16, 8
- Fixed: n_init=5, random_state=42

**Metrics:**
- ARI vs ground-truth
- NMI vs ground-truth
- Wall-clock time
- Cosine silhouette score (computed in the reduced space)

**Expected results:**
- Quality plateau between 32 and 128 (modern embeddings are overparameterized)
- Sharp quality drop below 16 dimensions
- Speed monotonically improves with lower dimensions
- Sweet spot: 64-128 dimensions

**Output table:**

| reduction_dim | ARI | NMI | Silhouette | Time (s) |
|---|---|---|---|---|
| None (1536) | | | | |
| 256 | | | | |
| 128 | | | | |
| 64 | | | | |
| 32 | | | | |
| 16 | | | | |
| 8 | | | | |

---

## Experiment 5: vMF Refinement Quality

**Question:** Does vMF soft refinement improve cluster assignments on multi-topic documents?

**Setup:**
- Dataset: 20 Newsgroups (contains documents that span topics, e.g., sci.med + sci.electronics)
- K = 20
- Compare hard spherical K-means labels vs vMF posterior argmax labels
- For multi-topic documents (manually identified or via keyword overlap), compare:
  - Hard label assignment quality
  - vMF posterior entropy (high entropy = multi-topic)
  - Top-2 posterior probabilities (does the second-best cluster make semantic sense?)

**Metrics:**
- ARI improvement from vMF refinement (hard argmax of posteriors vs original labels)
- Mean posterior entropy per true class
- Fraction of documents where top-2 posteriors both correspond to semantically related classes
- Concentration κ per cluster (do some topics have naturally looser clusters?)

**Additional analysis:**
- Identify documents with posterior entropy > 1.0 (high uncertainty)
- Manually inspect 20 high-entropy documents: are they genuinely multi-topic?
- Plot κ vs cluster size: does concentration correlate with topic specificity?

---

## Experiment 6: BIC Model Selection

**Question:** Does BIC-optimal K match the ground-truth topic count?

**Setup:**
- Datasets: 20 Newsgroups (K_true=20), AG News (K_true=4), Banking77 (K_true=77)
- Sweep K from 2 to 2×K_true
- For each K: fit EmbeddingCluster → refine_vmf → record BIC
- Also compute cosine silhouette for comparison

**Output:**
- BIC vs K curve for each dataset (expect minimum near K_true)
- Silhouette vs K curve (expect maximum near K_true)
- Predicted K_opt vs actual K_true

**Success criteria:**
- BIC minimum within ±20% of K_true on 2/3 datasets
- Silhouette maximum within ±30% of K_true

**Implementation:**

```python
def run_experiment_6(X, true_k, dataset_name):
    k_range = range(2, min(true_k * 3, 200), max(1, true_k // 5))
    results = []

    for k in k_range:
        model = EmbeddingCluster(n_clusters=k, reduction_dim=128, random_state=42, n_init=3)
        model.fit(X)
        model.refine_vmf()

        results.append({
            "k": k,
            "bic": model.bic_,
            "objective": model.objective_,
            "mean_resultant": np.mean(model.resultant_lengths_),
        })

    bic_best_k = min(results, key=lambda r: r["bic"])["k"]
    print(f"{dataset_name}: BIC-optimal K={bic_best_k}, true K={true_k}")
    return results
```

---

## Experiment 7: Comparison of Initialization Strategies

**Question:** Is spherical k-means++ better than uniform random initialization for embedding data?

**Motivation:** The literature suggests k-means++ is not universally better for spherical clustering — Schubert [9] found 7-8% worse performance on 20 Newsgroups.

**Setup:**
- Dataset: 20 Newsgroups embeddings
- K = 20
- Compare: spherical k-means++ vs uniform random init
- n_init = 1 (isolate initialization effect)
- 50 runs per strategy, report mean ± std of ARI and objective

**Expected outcome:**
- k-means++ should have lower variance
- Mean quality may be similar or k-means++ slightly better on clean data
- On data with outliers, k-means++ may be worse (outlier attraction)

---

## Execution Priority

| Experiment | Priority | Difficulty | Dependencies |
|---|---|---|---|
| 1. Synthetic recovery | P0 | Easy | None — synthetic data only |
| 3. Wall-clock scaling | P0 | Easy | None — synthetic data only |
| 4. PCA ablation | P1 | Medium | Requires 20 Newsgroups embeddings |
| 2. Baseline comparison | P1 | Medium | Requires embeddings + sklearn + BERTopic |
| 6. BIC model selection | P1 | Medium | Requires embeddings |
| 5. vMF refinement | P2 | Hard | Requires embeddings + manual inspection |
| 7. Init strategies | P2 | Easy | Requires embeddings |

**P0 experiments can run immediately** with synthetic data. P1 requires pre-computing OpenAI embeddings for the three datasets (~$5-10 in API costs). P2 requires manual qualitative analysis.

---

## Reproducibility Requirements

- All random seeds fixed and documented
- Exact software versions: rustcluster, sklearn, BERTopic, numpy, Python
- Hardware spec: CPU model, core count, RAM, OS
- Thread count explicitly set: `RAYON_NUM_THREADS=1`, `OMP_NUM_THREADS=1`
- Embedding model: `text-embedding-3-small` with version date
- All pre-computed embeddings published as .npy files (or generation script)
- All experiment scripts committed to `experiments/` directory
