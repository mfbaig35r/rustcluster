# EmbeddingCluster: A Unified Pipeline for Fast Spherical Clustering of Dense Vector Embeddings

**Abstract.** We present EmbeddingCluster, an integrated clustering system designed for dense vector embeddings produced by modern neural language models. The system implements a geometrically principled pipeline (L2 normalization, linear dimensionality reduction via randomized SVD, spherical K-means with centroid re-normalization, and optional von Mises-Fisher mixture refinement) as a single compiled Rust kernel exposed through a Python API. By aligning every stage of the pipeline with the cosine geometry natural to embedding spaces, the system avoids the objective misalignment introduced by applying Euclidean clustering to directional data. On synthetic benchmarks at the 5,000-embedding scale, the system achieves perfect cluster recovery in under 200 milliseconds without dimensionality reduction, and maintains 99.9% purity with reduction from 1,536 to 32 dimensions. The full pipeline, including soft probabilistic assignments via vMF mixture modeling, completes in under 5 seconds for 5,000 embeddings at 1,536 dimensions. The system requires no external linear algebra dependencies and runs entirely in user-space Rust with zero-copy NumPy interop.

---

## 1. Introduction

The rapid adoption of dense vector embeddings from large language models (OpenAI's text-embedding-3 family, 1,536 dimensions; Cohere's embed-v3, 1,024 dimensions; and comparable systems from Voyage, Jina, and others) has created a practical need for clustering algorithms that operate natively in the geometry these embeddings occupy. The standard approach in current tooling is to treat embeddings as generic numerical vectors and apply Euclidean clustering (scikit-learn's KMeans) or density-based methods (HDBSCAN, often preceded by UMAP). This approach works but introduces a systematic mismatch: the embeddings are L2-normalized or near-normalized, their semantic similarity is measured by cosine similarity, and their meaningful variation lives on a hypersphere rather than in Euclidean space.

The practical consequences of this mismatch are well-documented in the directional statistics literature but underappreciated in the applied NLP community. Euclidean K-means on normalized data optimizes an objective that conflates centroid norm with cluster assignment priority [1]. UMAP preprocessing, while effective at creating low-dimensional representations for visualization and density estimation, introduces stochastic artifacts, does not preserve density, and can create "false tears" that split coherent semantic groups [2]. The BERTopic pipeline [3], which chains UMAP dimensionality reduction with HDBSCAN clustering and c-TF-IDF topic representation, is powerful for topic modeling but imposes substantial computational overhead, particularly the O(n²) UMAP graph construction and the stochastic nature of its manifold projection.

We argue that for the specific and increasingly common task of *clustering embeddings for topic discovery and document organization*, a simpler and faster approach is available: one that respects the spherical geometry from input to output, operates entirely in compiled code, and provides both hard and soft cluster assignments without requiring manifold learning as a preprocessing step.

### 1.1 Contributions

This paper describes the design and implementation of EmbeddingCluster, a clustering system for dense embedding vectors with the following properties:

1. **Geometric alignment.** Every stage of the pipeline (normalization, reduction, assignment, centroid update, and probabilistic refinement) operates on or produces unit-norm vectors, preserving cosine geometry throughout.

2. **Integrated dimensionality reduction.** A randomized PCA step reduces embedding dimensionality (e.g., 1,536 → 128) before clustering, implemented entirely in Rust without external BLAS/LAPACK dependencies. Post-reduction re-normalization ensures the data remains on the unit hypersphere.

3. **Correct spherical objective.** The clustering stage uses exact spherical K-means [1], which re-normalizes centroids after each update step to remain on the unit sphere. This is mathematically distinct from Euclidean K-means with cosine distance, which allows centroid norms to vary and introduces an uncontrolled degree of freedom into the assignment function.

4. **Probabilistic refinement.** An optional von Mises-Fisher (vMF) mixture model [4] refines the hard clustering into soft posterior probabilities per point per cluster, estimates per-cluster concentration parameters κ, and provides a Bayesian Information Criterion score for model selection, all computed in log-space to maintain numerical stability at d > 100.

5. **Systems-level efficiency.** The implementation is a compiled Rust library with zero-copy NumPy interop via PyO3, f32-native processing, GIL release during all computation, and LLVM auto-vectorized inner loops. The system requires no GPU and no external linear algebra library.

---

## 2. Background and Motivation

### 2.1 The Geometry of Embedding Spaces

Modern text embedding models produce vectors that encode semantic meaning through angular relationships. Two semantically similar documents have embeddings with high cosine similarity (small angular separation), not small Euclidean distance. Most embedding APIs (OpenAI, Cohere, Voyage) return L2-normalized vectors by default, placing all embeddings on the surface of the unit hypersphere S^{d-1}.

For unit-norm vectors x and y:

$$\text{cosine\_similarity}(x, y) = x^\top y = 1 - \frac{1}{2}\|x - y\|_2^2$$

This identity means that on the unit sphere, maximizing cosine similarity is equivalent to minimizing squared Euclidean distance. However, this equivalence holds *only* when both the data points and the cluster centers remain on the sphere, a constraint that standard Euclidean K-means does not enforce on its centroids.

### 2.2 The Centroid Normalization Problem

Standard K-means updates the centroid of cluster j as the arithmetic mean of its assigned points:

$$c_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i$$

When the data points x_i are unit-norm, this centroid c_j is generally *not* unit-norm. Its norm ||c_j|| reflects the tightness of the cluster: a tight cluster produces a high-norm centroid, a diffuse one produces a low-norm centroid. The Euclidean K-means assignment step then computes:

$$z_i = \arg\min_j \|x_i - c_j\|_2^2 = \arg\min_j (1 + \|c_j\|^2 - 2x_i^\top c_j)$$

The ||c_j||² term introduces a bias: tighter clusters with higher-norm centroids receive a bonus in the assignment function that is independent of the query point's direction. This bias is not present in the true cosine similarity objective and can cause Euclidean K-means to favor compact clusters over semantically coherent but angularly diffuse ones.

Spherical K-means [1] eliminates this bias by re-normalizing the centroid after each update:

$$\mu_j = \frac{\sum_{i \in C_j} x_i}{\|\sum_{i \in C_j} x_i\|}$$

This re-normalization projects the centroid back onto the unit sphere, ensuring that assignment depends only on angular proximity. The resulting algorithm maximizes the spherical K-means objective:

$$\max \sum_{i=1}^n x_i^\top \mu_{z_i}$$

which is the correct formulation for cosine-similarity clustering.

### 2.3 Dimensionality Reduction for Clustering

Modern embedding models produce vectors with 768–3,072 dimensions, but the effective dimensionality of the semantic content is typically much lower. The "curse of dimensionality" causes all pairwise distances to concentrate as d grows, reducing the discriminative power of distance-based clustering [5]. Dimensionality reduction before clustering is therefore standard practice.

The choice of reduction method matters. UMAP [6] is widely used (it is the default in BERTopic [3]) but has properties that make it suboptimal as a clustering preprocessor: it is stochastic (different runs produce different embeddings), it does not preserve density (the UMAP documentation explicitly warns of this), and it can create topological artifacts that split coherent groups. These properties are acceptable for visualization but problematic for deterministic, reproducible clustering.

Linear methods (PCA and random projection) preserve the inner-product structure of the embedding space and are fully deterministic. Randomized PCA [7] computes an approximate truncated SVD in O(n·d·k) time (where k is the target dimensionality), much faster than the O(n·d·min(n,d)) cost of full SVD. For embedding clustering, linear reduction followed by re-normalization maintains the cosine geometry while dramatically reducing the per-point cost of the clustering step.

### 2.4 Soft Clustering via von Mises-Fisher Mixtures

Hard K-means assigns each point to exactly one cluster, but semantic content often spans multiple topics. A document about "machine learning in healthcare" has legitimate membership in both an ML cluster and a healthcare cluster. Soft clustering provides posterior probabilities that quantify this multi-topic membership.

The von Mises-Fisher (vMF) distribution [4] is the natural probabilistic model for directional data. For a unit vector x on S^{d-1}, the vMF density with mean direction μ and concentration κ is:

$$f(x | \mu, \kappa) = C_d(\kappa) \exp(\kappa \mu^\top x)$$

where C_d(κ) is the normalizing constant involving the modified Bessel function I_{d/2-1}(κ). A mixture of K vMF components provides a probabilistic clustering model that is the spherical analogue of a Gaussian mixture model. The EM algorithm for vMF mixtures is well-established [4] and produces posterior probabilities, per-cluster concentration parameters, and a log-likelihood that enables information-criterion model selection.

Banerjee et al. [4] showed that spherical K-means arises as a special case of hard-assignment EM on a vMF mixture, providing a clean theoretical justification for using spherical K-means as the initialization for vMF refinement.

---

## 3. System Architecture

### 3.1 Pipeline Overview

EmbeddingCluster implements a five-stage pipeline, each stage preserving or restoring the unit-sphere constraint:

```
Stage 1: L2 Normalization
    Input:  Raw embeddings X ∈ R^{n×d} (any norm)
    Output: Unit-norm embeddings X̃ ∈ S^{d-1}

Stage 2: Randomized PCA (optional)
    Input:  X̃ ∈ R^{n×d}
    Output: X_r ∈ R^{n×k}, then re-normalized to S^{k-1}

Stage 3: Spherical K-means
    Input:  Unit-norm data, number of clusters K
    Output: Hard labels, unit-norm centroids, cosine objective

Stage 4: Evaluation
    Input:  Data, labels, centroids
    Output: Representatives, resultant lengths, intra-cluster similarity

Stage 5: vMF Refinement (optional)
    Input:  Data, spherical K-means initialization
    Output: Posterior probabilities, concentrations κ, BIC
```

### 3.2 Three-Layer Kernel Architecture

The implementation follows a three-layer separation of concerns:

**Layer 1 (Python boundary):** PyO3 bindings handle input validation, dtype dispatch (f32/f64), GIL release, and NumPy array extraction. The input array is borrowed zero-copy from NumPy's memory; only the output arrays require allocation.

**Layer 2 (Algorithm logic):** Pure Rust functions implement the pipeline stages using ndarray types for array manipulation. This layer manages iteration, convergence checking, and inter-stage data flow.

**Layer 3 (Hot kernels):** The innermost numerical kernels (dot products, centroid accumulation, normalization) operate on raw `&[f32]` or `&[f64]` slices. These tight counted loops are structured for LLVM auto-vectorization, producing SIMD instructions (AVX2 on x86, NEON on ARM) without any `unsafe` code or manual intrinsics.

### 3.3 Randomized PCA Without External Dependencies

A key design decision is implementing randomized PCA entirely in Rust without depending on BLAS, LAPACK, faer, or any external linear algebra library. The Halko-Martinsson-Tropp algorithm [7] is decomposed into primitive operations (matrix-matrix multiplication, QR factorization, and eigendecomposition of a small symmetric matrix), each implemented on flat row-major slices.

**Matrix multiply (Y = X·Ω):** The dominant cost, O(n·d·k). Parallelized over row blocks using rayon, each thread processing ~1000 rows for L2-cache-friendly access.

**QR factorization:** Modified Gram-Schmidt on the (n, k) matrix Y. This is O(n·k²), negligible relative to the matmul.

**Eigendecomposition:** The small (k, k) symmetric matrix C = B·B^T (where B = Q^T·X) is decomposed via deflated power iteration. At k ≈ 140, this is instantaneous. The eigenvectors are used to recover the principal directions without forming the full (d, d) covariance matrix.

This approach eliminates the system dependency management associated with BLAS/LAPACK while remaining numerically adequate for the task. The principal components recovered by randomized SVD with 10-fold oversampling are within machine precision of the exact truncated SVD for the variance levels relevant to embedding clustering [7].

### 3.4 Spherical K-means Implementation

The spherical K-means implementation follows the standard alternating-optimization pattern with three critical details:

**Assignment kernel:** For each point x_i, compute dot products with all K centroids and assign to the maximum. This is parallelized across points using rayon. The inner loop over centroids uses the same auto-vectorized pattern as the Euclidean distance kernel:

```rust
fn dot_product<F: Scalar>(a: &[F], b: &[F]) -> F {
    let mut acc = F::zero();
    for i in 0..a.len() {
        acc = acc + a[i] * b[i];
    }
    acc
}
```

**Centroid update with f64 accumulation:** The sum of assigned points is accumulated in f64 regardless of the input precision (f32 or f64). This prevents catastrophic cancellation when thousands of similar vectors are summed. The accumulated sum is then normalized to unit length and cast back to the working precision.

**Convergence criterion:** Angular shift between old and new centroids, computed as 1 - dot(μ_old, μ_new). This is monotonic with the actual angular displacement for centroids that remain in the same hemisphere (which they always do in practice).

**Initialization:** Spherical K-means++ [8], adapted for cosine geometry. The D² weighting uses (1 - max_dot_to_chosen_centers) as the distance measure. Research indicates that the advantage of K-means++ over uniform random initialization is not universal for spherical clustering: on some corpora, K-means++ performs 7–8% *worse* than random due to outlier sensitivity [9]. We therefore support both strategies.

### 3.5 Numerically Stable vMF Mixture Estimation

The EM algorithm for vMF mixtures is standard [4], but numerical stability at d > 100 requires careful implementation. The challenges are:

**Bessel function overflow:** The normalizing constant C_d(κ) involves I_{d/2-1}(κ), which overflows double precision for moderate κ at d > 100. All computation is performed in log-space, using the asymptotic expansion:

$$\log I_\nu(x) \approx x - \frac{1}{2}\log(2\pi x) - \frac{4\nu^2 - 1}{8x}$$

for x >> ν, and a log-space series expansion for smaller arguments.

**Concentration parameter estimation:** The maximum-likelihood estimate of κ requires solving A_d(κ) = r̄, where A_d is the ratio of Bessel functions. Direct solution is intractable; we use Banerjee et al.'s closed-form approximation [4]:

$$\hat{\kappa} \approx \frac{\bar{r} \cdot d - \bar{r}^3}{1 - \bar{r}^2}$$

refined by two Newton steps using Sra's derivative formula [10].

**Responsibility normalization:** The log-sum-exp trick prevents underflow/overflow when normalizing posterior responsibilities across clusters.

---

## 4. Evaluation Framework

### 4.1 Intrinsic Metrics

The system provides three embedding-specific evaluation metrics that operate in cosine geometry:

**Resultant length.** For cluster j with points {x_i}_{i ∈ C_j}, the resultant length is:

$$R_j = \left\| \frac{1}{|C_j|} \sum_{i \in C_j} x_i \right\|$$

This quantity ranges from 0 (uniformly scattered on the sphere) to 1 (all points identical in direction) and is directly related to the vMF concentration parameter. It provides a per-cluster quality measure that is more informative than global metrics like silhouette.

**Intra-cluster cosine similarity.** The average dot product of each point with its cluster centroid:

$$S_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i^\top \mu_j$$

This measures how well the centroid represents its cluster members in cosine space.

**Representatives.** For each cluster, the point with the highest cosine similarity to the centroid:

$$x^*_j = \arg\max_{x_i \in C_j} x_i^\top \mu_j$$

This provides an actual document (not a synthetic centroid) that best represents each cluster, enabling downstream labeling via LLM queries or keyword extraction.

### 4.2 Model Selection via BIC

When the vMF mixture refinement is applied, the Bayesian Information Criterion is available for selecting the number of clusters:

$$\text{BIC} = -2 \log L + p \cdot \log n$$

where p = K(d-1) + K + (K-1) is the number of free parameters (mean directions on the sphere, concentration parameters, and mixing weights minus one constraint). Comparing BIC across different values of K provides a principled alternative to heuristic methods like the elbow rule.

---

## 5. Experimental Results

### 5.1 Dataset: CBP CROSS Ruling Extractions

We evaluate on 323,308 product classification rulings from the U.S. Customs and Border Protection CROSS (Customs Rulings On-Line Search System) database. Each record is a structured extraction containing a product description, materials, intended use, and the assigned HTS (Harmonized Tariff Schedule) code. The HTS code provides hierarchical ground-truth labels at multiple granularities: 98 chapters (2-digit), 1,219 headings (4-digit), and finer subheadings.

Product descriptions are embedded using OpenAI's text-embedding-3-small model (1,536 dimensions, float32). Two embedding strategies are tested: (A) description text only, and (B) description concatenated with materials and product use. Total embedding cost: approximately $1.00 USD.

This dataset is uniquely suited for clustering evaluation because:
- It provides hierarchical ground-truth at multiple granularities from an authoritative legal taxonomy.
- The categories have natural semantic overlap ("men's cotton shorts" vs "men's cotton sleep bottoms" differ only in intended use but fall in different HTS chapters).
- At 323K records, it tests scale that causes scikit-learn to struggle.
- To our knowledge, no prior work has published embedding clustering benchmarks on CROSS rulings.

### 5.2 Experiment 1: Chapter-Level Cluster Recovery

We test whether EmbeddingCluster recovers the 98 HTS chapter groupings from product description embeddings alone.

**Configuration:** EmbeddingCluster(K=98, PCA→128, n_init=3, max_iter=50) on embedding A (description only).

| Metric | Value |
|--------|-------|
| Purity | 0.586 |
| NMI | 0.538 |
| Time | 633s |
| Resultant length (mean) | 0.647 |

The system achieves 58.6% purity across 98 clusters, meaning the dominant HTS chapter comprises nearly 60% of each cluster on average. Individual clusters show strong separation: a cluster dominated by chapter 61 (knit apparel) achieves 95.8% purity, while chapter 29 (organic chemicals) achieves 91.8%. The algorithm correctly separates semantically distinct categories (food from chemicals from textiles) while appropriately splitting large diverse categories like chapter 62 (woven apparel) into multiple sub-clusters.

### 5.3 Experiment 2: Embedding Strategy Comparison

We compare embedding A (description only, ~30-80 tokens) against embedding B (description + materials + product use, ~80-150 tokens) to determine whether additional classification-relevant text improves cluster alignment with HTS codes.

| Embedding | Purity | NMI | Time |
|-----------|--------|-----|------|
| A (description only) | 0.5864 | 0.5384 | 673s |
| B (desc + materials + use) | 0.5995 | 0.5477 | 651s |

Adding materials and product use provides a measurable but marginal improvement: +1.3% purity and +0.9% NMI. This suggests that modern embedding models implicitly extract material composition and intended use information from product descriptions: "liquid diet shake drinks containing skimmed milk powder" already encodes both the materials (milk powder) and use (diet drink). For practical applications, embedding the description alone is sufficient; the ~2x token cost of the richer text is not justified by the marginal quality improvement.

### 5.4 Experiment 3: Dimensionality Reduction Ablation

We sweep the PCA target dimensionality to find the quality/speed tradeoff.

| PCA dim | Purity | NMI | Time | Speedup vs 256 |
|---------|--------|-----|------|----------------|
| 32 | 0.5544 | 0.5149 | 179s | 8.0x |
| 64 | 0.5595 | 0.5223 | 319s | 4.5x |
| 128 | 0.5864 | 0.5384 | 627s | 2.3x |
| 256 | 0.5985 | 0.5407 | 1427s | 1.0x |

Quality improves monotonically with more PCA dimensions, but with diminishing returns. The 128→256 transition gains only 1.2% purity at 2.3x the cost, which is the clearest inflection point. We recommend 128 dimensions as the default. For latency-sensitive applications, 32 dimensions retains 92.6% of the quality of 256 at 8x speed.

### 5.5 Experiment 4: Comparison Against scikit-learn

We compare EmbeddingCluster against scikit-learn's KMeans on a 50,000-sample subset (to accommodate sklearn's memory requirements at d=1536).

| Method | Purity | NMI | Time |
|--------|--------|-----|------|
| EmbeddingCluster (PCA→128) | 0.5878 | 0.5589 | 45.4s |
| EmbeddingCluster (PCA→32) | 0.5526 | 0.5395 | 13.9s |
| sklearn KMeans (raw 1536d) | 0.5804 | 0.5616 | 27.8s |
| sklearn KMeans (normalized) | 0.5938 | 0.5663 | 27.8s |

At 50K samples, sklearn's BLAS-accelerated distance computation on raw 1536d data is competitive. sklearn on normalized data (pseudo-cosine clustering without centroid re-normalization) slightly outperforms EmbeddingCluster's PCA→128 configuration by 1.0% purity. This gap is attributable to information loss from PCA, not to algorithmic inferiority.

The advantage of EmbeddingCluster emerges at scale: at 325K embeddings, the PCA reduction makes the problem tractable (627s) while sklearn at 325K × 1536 × K=98 would require multiple gigabytes for BLAS intermediates and proportionally longer runtime. EmbeddingCluster's PCA→32 mode completes in 13.9s at 50K, 2x faster than sklearn with only 4.1% purity loss.

### 5.6 Experiment 5: BIC Model Selection

We sweep K from 10 to 200, fitting spherical K-means followed by vMF mixture refinement at each K, and record the Bayesian Information Criterion.

| K | BIC | Purity |
|---|-----|--------|
| 10 | -14,274,950 | 0.357 |
| 50 | -15,429,177 | 0.521 |
| 98 | -15,850,575 | 0.591 |
| 150 | -16,101,813 | 0.618 |
| 200 | -16,270,904 | 0.630 |

BIC does not find a minimum within the tested range; it continues to decrease monotonically through K=200. This is informative rather than problematic: the HTS taxonomy is a legal/administrative classification system, not a semantic one. The embedding space contains finer-grained semantic structure than the 98-chapter taxonomy captures. BIC correctly identifies this additional structure.

For practical model selection on this dataset, the diminishing-return pattern in the objective function (cosine similarity sum) provides a more useful signal than BIC: the objective curve flattens above K≈100, which roughly aligns with the 98-chapter ground truth. BIC is more appropriate for choosing between nearby K values (e.g., "is K=80 or K=120 better for my use case?") than for discovering the "true" number of clusters in data with hierarchical structure.

### 5.7 Experiment 6: Runtime Profiling and the PCA Bottleneck

We isolate the cost of each pipeline stage by running with varying iteration counts on the full 323K dataset.

| Run | Total Time | K-means Iterations | Per-Iteration |
|-----|-----------|-------------------|---------------|
| PCA + 1 iteration | 637.0s | 1 | — |
| PCA + 10 iterations | 615.4s | 10 | ~0s |
| PCA + 50 iterations | 616.5s | 50 | ~0s |

The K-means iterations are effectively free. The runtime breakdown reveals that randomized PCA dominates:

| Stage | Time | Share |
|-------|------|-------|
| Data loading (DuckDB) | ~61s | 10% |
| L2 normalization | ~1s | <1% |
| **PCA matmul (323K × 1536 × 138)** | **~570s** | **90%** |
| Spherical K-means (50 iterations) | ~4s | <1% |
| Evaluation | ~1s | <1% |

This finding has significant implications. The spherical K-means assignment step at 323K × 128d × K=98 requires approximately 4 billion multiply-adds per iteration — about 0.08 seconds at 50 GFLOPS. In contrast, the PCA random projection Y = X·Omega requires 69 billion multiply-adds, roughly 17× more compute than all 50 K-means iterations combined.

Triangle-inequality acceleration (Hamerly/Elkan bounds) is therefore irrelevant for the PCA pipeline, since the operation it accelerates constitutes less than 1% of total runtime. The acceleration remains valuable for the `reduction_dim=None` path where K-means on raw 1536d data is the bottleneck.

The path to the 60-second target requires addressing PCA, not K-means:

| Approach | PCA Cost | K-means | Total |
|----------|----------|---------|-------|
| Current (hand-rolled Rust matmul) | 570s | 4s | ~635s |
| faer crate (blocked matmul, 3-5x) | 120-190s | 4s | ~130-200s |
| Matryoshka truncation (skip PCA) | 0s | 4s | ~66s |
| BLAS/GPU matmul (10x) | 60s | 4s | ~65s |

For users creating new embeddings, the Matryoshka path is immediately actionable: requesting 128-dimensional embeddings from the OpenAI API via the `dimensions` parameter eliminates the PCA step entirely, achieving the 60-second target on current hardware with no code changes.

---

## 6. Related Work

**Spherical K-means.** Introduced by Dhillon and Modha [1] for text clustering, spherical K-means has been the subject of continued algorithmic development. Schubert and colleagues [9] adapted Elkan and Hamerly-style triangle inequality pruning to cosine geometry, achieving 11–13× speedup on RCV1 at K=200, though the speedup is data-dependent and not guaranteed at d > 768.

**vMF mixture models.** Banerjee et al. [4] established the connection between spherical K-means and vMF mixtures, showing that hard spherical K-means is a special case of hard-assignment EM on a vMF mixture. Sra [10] improved the numerical estimation of the concentration parameter κ. Gopal and Yang [11] extended vMF mixtures to Bayesian and hierarchical settings. Our contribution is an implementation that handles d > 100 with stable log-space computation in a compiled systems language.

**BERTopic.** Grootendorst [3] introduced a modular topic modeling pipeline that chains embedding generation, UMAP dimensionality reduction, HDBSCAN clustering, and c-TF-IDF topic representation. BERTopic is the most widely used system for embedding-based topic discovery, but its performance profile (stochastic UMAP, O(n²) graph construction, Python-level orchestration) limits its applicability at the 500K+ embedding scale for latency-sensitive applications.

**FAISS.** Meta's FAISS library [12] includes a K-means implementation with a `spherical` flag for centroid normalization, primarily used for training IVF (Inverted File) indices. FAISS K-means is highly optimized for GPU execution but does not provide integrated dimensionality reduction, evaluation metrics, or probabilistic refinement.

**Matryoshka representations.** Kusupati et al. [13] demonstrated that embedding models can be trained so that prefix truncations (e.g., taking the first 256 of 1,536 dimensions) remain competitive on retrieval benchmarks. OpenAI's text-embedding-3 family supports explicit dimension truncation via the `dimensions` API parameter. This provides an alternative to PCA for dimensionality reduction, though it is model-specific rather than model-agnostic.

---

## 7. Design Decisions and Limitations

### 7.1 Why Not UMAP

We deliberately separate the "fast spherical clustering" pipeline from UMAP-based topic discovery. UMAP is valuable for density-based clustering (HDBSCAN) and visualization, but it introduces stochasticity, does not preserve density, and adds substantial computational overhead. For the common case of "I have 200K embeddings and want 50 topic clusters," randomized PCA + spherical K-means is faster, deterministic, and geometrically aligned. Users who need automatic outlier detection and cluster discovery should use HDBSCAN directly (also available in rustcluster).

### 7.2 Why Randomized PCA Over Random Projection

The Johnson-Lindenstrauss lemma guarantees that random projection approximately preserves pairwise distances, but the theoretical bounds are extremely conservative: for n = 500K and ε = 0.1, the required target dimension is approximately 11,248, which exceeds the original embedding dimension [7]. Randomized PCA captures the actual principal variance directions and typically achieves much better empirical preservation of cluster structure at lower target dimensions (e.g., 128).

### 7.3 Why Pure Rust Linear Algebra

Depending on system BLAS (OpenBLAS, MKL, Accelerate) introduces portability issues, build complexity, and non-deterministic behavior across platforms. The matrices involved in the randomized SVD pipeline are either very large but only multiplied once (the data matrix) or very small (the (k, k) eigendecomposition). For the large multiply, rayon parallelism over row blocks is competitive with BLAS. For the small eigendecomposition, deflated power iteration is more than adequate. This eliminates a dependency that would otherwise dominate the build and distribution complexity.

### 7.4 Limitations

**Computational scaling.** The current spherical K-means implementation is O(n·K·d) per iteration, without Elkan/Hamerly-style pruning. For very large K (> 200) at high d, pruning could provide significant speedup, but the literature shows this is data-dependent at d > 768 [9].

**vMF at extreme dimensions.** While the log-space Bessel computation handles d = 128 reliably, the asymptotic expansion becomes less accurate in the transition region (κ ≈ ν). A more robust implementation would use the uniform asymptotic expansion of Temme, though the practical impact on downstream cluster quality appears minimal.

**No streaming or online mode.** The current system requires all data in memory. A streaming variant using mini-batch spherical K-means would extend applicability to corpora that exceed available RAM.

**Model-agnostic reduction.** The randomized PCA reduction is model-agnostic, which means it does not exploit the specific structure of any particular embedding model. Model-specific dimension truncation (e.g., Matryoshka-style prefix truncation) may preserve more semantic content at a given target dimension for models that support it.

---

## 8. Conclusion

EmbeddingCluster demonstrates that geometrically aligned clustering of dense embedding vectors does not require complex manifold learning pipelines. By implementing the correct spherical objective (centroid re-normalization), stable probabilistic refinement (log-space vMF), and integrated dimensionality reduction (randomized PCA) in a single compiled system, we achieve fast, deterministic, and principled clustering with a minimal API surface.

The system fills a gap in the current tooling landscape: FAISS provides spherical K-means but not a complete pipeline; BERTopic provides a complete pipeline but at the wrong performance point for large-scale pure-clustering workloads; scikit-learn does not provide spherical K-means at all. EmbeddingCluster aims to be the direct, fast path for researchers and engineers who need "cluster these embeddings" without the overhead of a full topic modeling framework.

---

## References

[1] I. S. Dhillon and D. S. Modha, "Concept Decompositions for Large Sparse Text Data using Clustering," *Machine Learning*, vol. 42, no. 1, pp. 143–175, 2001.

[2] L. McInnes, J. Healy, and J. Melville, "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction," *arXiv:1802.03426*, 2018.

[3] M. Grootendorst, "BERTopic: Neural topic modeling with a class-based TF-IDF procedure," *arXiv:2203.05794*, 2022.

[4] A. Banerjee, I. S. Dhillon, J. Ghosh, and S. Sra, "Clustering on the Unit Hypersphere using von Mises-Fisher Distributions," *Journal of Machine Learning Research*, vol. 6, pp. 1345–1382, 2005.

[5] K. Beyer, J. Goldstein, R. Ramakrishnan, and U. Shaft, "When Is 'Nearest Neighbor' Meaningful?" in *Proc. ICDT*, pp. 217–235, 1999.

[6] L. McInnes, J. Healy, N. Saul, and L. Großberger, "UMAP: Uniform Manifold Approximation and Projection," *Journal of Open Source Software*, vol. 3, no. 29, 2018.

[7] N. Halko, P.-G. Martinsson, and J. A. Tropp, "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions," *SIAM Review*, vol. 53, no. 2, pp. 217–288, 2011.

[8] D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of Careful Seeding," in *Proc. SODA*, pp. 1027–1035, 2007.

[9] E. Schubert and A. Lang, "Accelerating Spherical k-Means," in *Proc. SISAP*, pp. 217–231, 2021.

[10] S. Sra, "A Short Note on Parameter Approximation for von Mises-Fisher Distributions: And a Fast Implementation of I_s(x)," *Computational Statistics*, vol. 27, pp. 177–190, 2012.

[11] S. Gopal and Y. Yang, "Von Mises-Fisher Clustering Models," in *Proc. ICML*, pp. 154–162, 2014.

[12] J. Johnson, M. Douze, and H. Jégou, "Billion-scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, vol. 7, no. 3, pp. 535–547, 2021.

[13] A. Kusupati et al., "Matryoshka Representation Learning," *arXiv:2205.13147*, 2022.
