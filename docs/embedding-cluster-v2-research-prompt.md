# Deep Research Prompt: EmbeddingCluster v2 Development Strategy

> ## Context
>
> I built EmbeddingCluster, a purpose-built embedding clustering pipeline implemented in Rust with Python bindings. It ships as part of `rustcluster`, a broader clustering library published on PyPI. The v1 implementation is complete and validated on 323K real-world embeddings. I need concrete, evidence-based research to guide v2 development.
>
> ## What v1 Implements (Current State)
>
> **Pipeline:** L2 normalize → Randomized PCA → Spherical K-means → Evaluation → Optional vMF refinement
>
> **Spherical K-means implementation:**
> - Lloyd-style alternating optimization: assign by max dot product, update by accumulate-and-renormalize
> - Spherical k-means++ initialization (weight by 1 - max_dot)
> - f64 accumulation for centroid sums, cast to f32 for assignment
> - Convergence: angular shift (1 - dot(old, new)) < tol
> - n_init restarts, keep best objective (max sum of dot products)
> - Parallelized assignment step via rayon
> - NO Elkan/Hamerly-style bounds — every point checks every centroid every iteration
>
> **Randomized PCA:**
> - Halko-Martinsson-Tropp with 10x oversampling
> - QR via modified Gram-Schmidt (pure Rust, no BLAS)
> - Eigendecomposition of small (k×k) matrix via deflated power iteration
> - Matrix multiply parallelized over row blocks via rayon
> - No external linear algebra dependency
>
> **vMF mixture refinement:**
> - Full soft EM initialized from spherical K-means
> - Banerjee κ approximation + 2 Sra Newton steps
> - Log-space Bessel via asymptotic expansion for large arguments, series for small
> - lgamma via Stirling + recurrence
> - BIC computation for model selection
> - All computation in f64
>
> **Evaluation metrics:**
> - Resultant length per cluster (directional concentration)
> - Intra-cluster cosine similarity (avg dot to centroid)
> - Representatives (nearest-to-centroid point per cluster)
> - Cosine silhouette (with optional sampling for large n)
>
> ## v1 Experimental Results (323K CROSS Ruling Embeddings)
>
> **Dataset:** 323,308 product description embeddings from U.S. Customs rulings (text-embedding-3-small, 1536d, f32). Ground truth: 98 HTS chapters.
>
> **Key results:**
> - Chapter-level recovery: 58.6% purity, 0.538 NMI at K=98 with PCA→128
> - Embedding A vs B: description-only captures 98.7% of the quality of description+materials+use
> - PCA ablation: 128d is sweet spot (93% quality of 256d at 44% time). Quality improves monotonically but with diminishing returns
> - vs sklearn: sklearn normalized beats us by 1.0% purity at 50K (0.5938 vs 0.5878). Gap is from PCA information loss, not algorithmic
> - BIC sweep: no minimum found through K=200 — HTS taxonomy doesn't correspond to semantic structure
> - Time: 633 seconds for 323K × 128d × K=98 × 3 inits × 50 iters. All runs hit max_iter
>
> ## What I Need Researched for v2
>
> ### 1. Accelerated Spherical K-means (Highest Priority)
>
> The 633-second runtime is the #1 problem. The v1 inner loop does n×K dot products per iteration — no pruning.
>
> - **Schubert & Lang (2021)** adapted Elkan and Hamerly bounds to cosine geometry and report 11-13x speedup on RCV1. I need the exact implementation details:
>   - What is the triangle inequality for cosine similarity? Is it `|cos(a,c) - cos(b,c)| ≤ some_bound`? Or do they work in angular distance space?
>   - How are the upper/lower bounds initialized and updated after centroid re-normalization?
>   - The paper shows Elkan variants use O(n×K) memory for lower bounds. At n=323K, K=98, that's 253MB in f64. Is Hamerly (O(n) bounds) the better choice for our regime?
>   - Are the bounds exact or approximate after centroid re-normalization? (In Euclidean K-means, bounds degrade by the centroid shift amount. Does the same hold for cosine?)
>   - What about "simplified Elkan" — the paper mentions it's faster than full Elkan on some data. What's simplified?
>   - **Critical question:** At d=128 (after PCA), how effective is the pruning? The paper benchmarks at d=47K (sparse text) and d=300 (word2vec). Dense 128d embeddings may prune differently.
>
> - Are there other acceleration approaches beyond Elkan/Hamerly?
>   - Yinyang K-means adapted for cosine?
>   - Annular/Exponion bounds?
>   - Ball tree / KD-tree for maximum inner product search (MIPS)?
>   - Product quantization for approximate assignment?
>
> - What is the best practical approach for d=128, n=100K-500K, K=50-200? I want a concrete recommendation, not a survey.
>
> ### 2. Mini-Batch Spherical K-means
>
> Our existing `MiniBatchKMeans` uses streaming centroid updates on Euclidean data. How should this adapt for the spherical case?
>
> - Does the streaming update `c += learning_rate * (x - c)` work on unit-norm centroids? Or does the renormalization after each mini-batch update destroy the streaming property?
> - What is the correct mini-batch update for spherical K-means? Options:
>   - (A) Accumulate sum over batch, renormalize, blend with previous centroid
>   - (B) Standard streaming update, then renormalize
>   - (C) Something else from the literature?
> - What batch sizes work well? Our Euclidean mini-batch uses 1024 default.
> - How does convergence compare to full-pass spherical K-means? On our dataset, full-pass takes 50 iterations to converge. Would mini-batch converge faster in wall-clock time despite more iterations?
> - Is there a published mini-batch spherical K-means algorithm? Or is this novel enough to document?
>
> ### 3. Better Convergence Detection
>
> All v1 experiments hit max_iter=50. The algorithm hasn't fully converged.
>
> - What is the right convergence criterion for spherical K-means?
>   - Angular shift < tol (our current approach, but tol=1e-6 is too tight for 323K points)
>   - Fraction of points that change assignment < threshold (e.g., < 0.1%)
>   - Relative objective change < threshold
>   - Some combination?
> - What are practical tol values for 100K+ point datasets? Our tol=1e-6 never triggers.
> - Is there an adaptive approach — run until improvement per iteration drops below some fraction?
>
> ### 4. PCA vs Matryoshka Truncation
>
> OpenAI's text-embedding-3 supports `dimensions` parameter for Matryoshka-style prefix truncation. How does this compare to our randomized PCA?
>
> - At d_target=128, does Matryoshka truncation preserve more or less semantic structure than PCA?
> - Is there published evaluation comparing Matryoshka truncation vs PCA for downstream clustering tasks (not just retrieval)?
> - If Matryoshka is better, should we offer it as an alternative to PCA? (It's model-specific; PCA is model-agnostic)
> - Are there other model-agnostic reduction methods worth considering? Random projection? Autoencoder? SparseRandomProjection?
>
> ### 5. Improving Cluster Quality Beyond 58.6%
>
> Our 58.6% purity at K=98 is honest but not spectacular. What algorithmic improvements could lift it?
>
> - **Multi-view clustering:** We have two embeddings (A and B). Is there a principled way to combine them rather than choosing one? Concatenate and re-embed? Weighted objective?
> - **Refinement passes:** After initial clustering, re-embed cluster centroids and do a second pass? Or use the vMF posteriors to soft-reassign and recluster?
> - **Hierarchical approach:** Cluster at K=20 (section level) first, then sub-cluster each into ~5 groups? Does this produce better chapter-level alignment than flat K=98?
> - **Post-clustering merging:** Start with K=200, then merge clusters that are too similar (cosine between centroids > threshold)? This is what HDBSCAN's condensed tree does.
> - **Is 58.6% actually the ceiling for this task?** What does sklearn's spectral clustering, or a tuned BERTopic pipeline, achieve on the same data? Is there a theoretical upper bound for clustering 98 administratively-defined categories from product description embeddings?
>
> ### 6. Scaling the vMF Refinement
>
> Our vMF EM at n=50K, K=98 took only ~6 seconds per iteration (it's dominated by the spherical K-means time). But the n×K responsibility matrix (50K × 98 × 8 bytes = 39MB) grows linearly.
>
> - At n=500K, K=200: responsibilities = 800MB. Is there a sparse or chunked approach?
> - Hard-assignment vMF (Banerjee's "moVMF hard") skips the responsibility matrix entirely. How much quality do you lose vs soft EM?
> - Is there a streaming/online vMF EM that processes data in chunks?
> - What about variational inference for vMF mixtures? Is it faster than EM for large n?
>
> ### 7. Practical Engineering Questions
>
> - **BLAS for the PCA matmul:** Our pure-Rust matrix multiply for Y = X·Omega is parallelized via rayon but doesn't use BLAS. At n=323K, d=1536, target=128: would linking against BLAS (via faer or openblas-sys) meaningfully speed up PCA? Is it worth the dependency?
> - **Auto-vectorization verification:** Our dot product kernel auto-vectorizes via LLVM. Is there a way to verify this at d=128? Should we add an explicit SIMD path for this dimension?
> - **Memory layout:** We store points as flat row-major f32. For the assignment step (n×K dot products), would column-major storage or tiling be better for cache behavior?
> - **GIL release granularity:** We release the GIL for the entire fit() call. Should we release/acquire at a finer granularity (per-iteration?) to support Python signal handling (Ctrl+C) during long runs?
>
> ### 8. Competitive Positioning
>
> - What are the latest BERTopic / Top2Vec / FAISS benchmarks on datasets comparable to ours (100K+ embeddings, 50+ clusters)?
> - Is anyone else shipping spherical K-means with integrated PCA and vMF refinement as a Python package?
> - What would make EmbeddingCluster the default choice for embedding clustering? Speed? Quality? Simplicity? All three?
>
> ## What I Need From This Research
>
> For each topic:
> 1. **Concrete recommendation** — what to implement, in what order
> 2. **Mathematical details** — exact formulas, not just references
> 3. **Expected impact** — how much speedup / quality improvement, with evidence
> 4. **Implementation complexity** — how hard is it, roughly how many lines of Rust
> 5. **Risks** — what could go wrong, what are the failure modes
>
> Deliver findings organized by section with specific numbers, formulas, algorithm pseudocode, and references. Flag anything where the evidence is weak. I want to make engineering decisions, not read a survey.
