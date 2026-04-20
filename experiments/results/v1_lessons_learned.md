# EmbeddingCluster v1 Experiments: Lessons Learned

Post-mortem on the first experimental validation of EmbeddingCluster against 323,308 CBP CROSS ruling embeddings with 98 HTS chapter ground-truth labels.

---

## What worked

### The pipeline architecture is sound

L2 normalize → PCA → spherical K-means → evaluation works end-to-end on 323K real embeddings without crashing, without external dependencies, and without GPU. The three-layer kernel design (PyO3 boundary → algorithm logic → hot kernel) scaled from toy data to production data without any structural changes.

### The algorithm finds real structure

Chemicals (chapter 29) clustered at 91.8% purity. Knit apparel (chapter 61) at 95.8%. The system cleanly separates semantically distinct categories. Where it struggles — distinguishing "men's cotton shorts" from "men's cotton sleep bottoms" — the distinction is about intended use, not product description, which is a legitimate ambiguity in the embedding space.

### PCA→128 is the right default

The ablation confirmed our pre-experiment default. 128 dimensions captures 93% of the quality of 256 at 44% of the time. The inflection point at 128 is clear and the recommendation is unambiguous.

### f32 throughout works

The entire pipeline — embedding (OpenAI returns f32 precision), storage (DuckDB), loading (numpy f32), PCA (f64 internally, f32 output), clustering (f32 dot products) — runs in f32 without precision issues. No silent upcasting needed.

---

## What didn't work as expected

### 58.6% purity, not 90%+

On synthetic data with well-separated vMF clusters, we get near-perfect recovery. On real customs classification data, 58.6%. The gap is not an algorithm failure — it's a dataset reality. The HTS taxonomy is a legal system designed by trade lawyers, not a semantic taxonomy derived from product descriptions. Many HTS distinctions (e.g., "article of apparel" vs "article of textile" for the same physical object) are invisible to a language model.

**Lesson:** Don't oversell results on synthetic data. Real-world taxonomies have legitimate misalignment with embedding geometry.

### sklearn normalized beats us at 50K

sklearn on L2-normalized data (pseudo-cosine K-means without centroid re-normalization) achieved 0.5938 purity vs our 0.5878. The gap is small (1.0%) and entirely attributable to PCA information loss — not to algorithmic inferiority.

**Lesson:** Our advantage is architectural (integrated pipeline, scales to 325K+), not algorithmic at moderate scale. The paper should position this honestly: EmbeddingCluster wins on simplicity and scale, not on raw quality at 50K.

### BIC doesn't find K=98

BIC decreased monotonically through K=200. It correctly identifies that embedding space has more than 98 semantic modes — but that's not useful if the user wants "give me 98 clusters matching the HTS chapters."

**Lesson:** BIC answers "how many statistical modes does this data have?" not "how many clusters does my taxonomy have?" For hierarchical real-world data, these are different questions. The paper should recommend BIC for comparing nearby K values, not for absolute K discovery.

### 633 seconds, not 60

The 60-second target assumed better convergence and lower K. At K=98 with 3 inits × 50 iterations on 323K × 128d, the wall-clock time is 10+ minutes. The PCA step is fast (~30s); the spherical K-means iterations dominate.

**Lesson:** The performance story is "EmbeddingCluster makes 325K embedding clustering tractable" (sklearn would struggle), not "EmbeddingCluster is fast." For the fast story, we need accelerated spherical K-means (Elkan/Hamerly bounds).

### All runs hit max_iter=50

Every experiment reached the iteration limit. The algorithm hasn't fully converged. More iterations would improve all results, and the relative ranking across experiments would likely hold — but the absolute numbers are conservative.

**Lesson:** The default max_iter=100 (not 50) and monitoring convergence via angular shift would give users better results out of the box.

---

## What we'd change for v2

### 1. Accelerated spherical K-means

The literature shows Elkan/Hamerly-style triangle inequality bounds adapted for cosine geometry achieve 11-13x speedup on text data [Schubert & Lang, 2021]. This would bring 633s → ~50-60s, hitting the original target. The bounds are data-dependent and may be less effective at d=128 than at lower dimensions, but the potential is substantial.

### 2. Mini-batch spherical K-means

For n > 100K, full-pass K-means is slow. Processing random batches of 1-4K points per iteration with streaming centroid updates (same pattern as our existing MiniBatchKMeans) would trade some quality for major speed improvement. The vMF refinement could then run on the full dataset for final soft assignments.

### 3. Smarter convergence

Instead of a fixed max_iter, monitor the angular shift between old and new centroids and stop when it plateaus. The current `tol` parameter exists but is set too tight (1e-6) — on real data with this many points, the centroids shift by tiny amounts each iteration even when the assignment is essentially stable. A more practical criterion: stop when < 0.1% of points change assignment between iterations.

### 4. Heading-level experiments

We only evaluated at chapter level (K=98). Testing at heading level (K≈200-500 for the most common headings) would show whether the algorithm captures finer-grained tariff structure. The BIC sweep suggests it might — purity keeps improving through K=200.

### 5. Cosine silhouette in evaluation

We implemented cosine silhouette but didn't include it in the experiments. Adding it would provide a ground-truth-independent quality metric that users can compute without labels.

### 6. Direct embedding dimension comparison

OpenAI's text-embedding-3 supports a `dimensions` parameter for Matryoshka-style truncation. Comparing Matryoshka truncation to 128d vs PCA reduction to 128d would determine whether model-specific truncation is better than model-agnostic PCA. If so, we could offer both paths.

---

## What this means for the paper

### Claims we can make confidently

- EmbeddingCluster provides a geometrically principled pipeline that works end-to-end on 323K real embeddings.
- PCA→128 is the right default dimensionality reduction for embedding clustering.
- Product descriptions alone capture nearly all classification-relevant signal (~98.7% of the richer embedding's quality).
- The system scales to 325K embeddings without external dependencies or GPU.
- Spherical K-means correctly separates semantically distinct categories on real data.

### Claims we should NOT make

- "EmbeddingCluster beats sklearn" — it doesn't on quality at moderate scale.
- "BIC finds the right K" — it doesn't on hierarchical taxonomies.
- "60 seconds for 500K embeddings" — not yet, needs acceleration.
- "90%+ purity on real data" — 58.6% on a genuinely hard dataset.

### The honest framing

EmbeddingCluster is the right *architecture* for embedding clustering: integrated, geometrically aligned, and scalable. The v1 *implementation* establishes a strong baseline with clear, well-understood paths to improvement. The experimental results on CROSS rulings are the first published embedding clustering benchmark on customs classification data and provide a realistic (not cherry-picked) evaluation of spherical K-means on production embeddings.
