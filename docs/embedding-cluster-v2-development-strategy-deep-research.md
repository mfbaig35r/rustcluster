# EmbeddingCluster v2 Development Strategy

## Implementation order

The highest-value v2 path is to make the exact solver much cheaper before you add more modeling. The order I would implement is: first, an exact cosine-geometry Hamerly accelerator for spherical k-means; second, practical stopping rules plus per-iteration instrumentation; third, chunked vMF refinement with a hard-vMF option; fourth, weighted multi-view fusion of your two embedding views; fifth, a model-specific Matryoshka fast path and a faster dense linear algebra backend for PCA. That order matches where the literature shows the biggest runtime wins for exact k-means-style clustering, and it avoids spending engineering time on approximate or Bayesian machinery before the dominant \(n\times K\) assignment loop is fixed. citeturn8search0turn18view0turn36view2turn30search0turn22view1turn40search0

For your regime—dense normalized embeddings at \(d=128\), \(n\approx 10^5\) to \(5\times 10^5\), \(K=50\) to \(200\)—the best default v2 target is still an **exact** algorithm, not an ANN-first one. The reason is practical rather than ideological: recent exact k-means work keeps winning by pairing triangle-inequality pruning with hardware-friendly dense kernels, while approximate search introduces objective drift and weakens your current “honest fixed-\(K\) clustering” positioning. Where the evidence is weak, I call that out explicitly below. citeturn8search0turn8search3turn10view2

## Accelerated spherical k-means

The concrete recommendation is to implement **cosine Hamerly first**, not full Elkan, not Yinyang, and not a tree or PQ backend. The 2021 spherical acceleration paper adapts both Elkan and Hamerly to spherical k-means and reports strong speedups; it also explicitly notes that Elkan-style variants need \(O(nK)\) lower-bound storage while the Hamerly family uses only \(O(n)\) point bounds, and that the Hamerly variants had much lower memory overhead in their experiments. By comparison, scikit-learn’s own documentation still warns that Elkan’s acceleration is more memory intensive because it allocates an extra \((n_{\text{samples}},n_{\text{clusters}})\) array. In your setting, that memory asymmetry matters more than the last bit of pruning power. At \(n=323{,}308,\ K=98\), an \(n\times K\) f64 lower-bound matrix is about \(323{,}308\times 98\times 8 \approx 253\) MB before allocator overhead; Hamerly needs only one upper and one lower bound per point, so even in f64 it is only a few megabytes. citeturn8search0turn18view0

The key geometric point is that the **exact** triangle inequality lives in **angular distance space**, not in raw cosine space. For unit vectors,
\[
d_\angle(x,y)=\arccos(x^\top y),
\]
and \(d_\angle\) is a true metric on the sphere, so
\[
d_\angle(x,y)\le d_\angle(x,z)+d_\angle(z,y).
\]
That gives exact cosine bounds by monotonicity of \(\cos\) on \([0,\pi]\):
\[
x^\top y \ge \cos\!\big(\arccos(x^\top z)+\arccos(z^\top y)\big),
\]
and
\[
x^\top y \le \cos\!\big(|\arccos(x^\top z)-\arccos(z^\top y)|\big).
\]
The 2021 SISAP paper derived these cosine-similarity inequalities precisely because cosine itself is not a metric, and the spherical k-means acceleration paper then uses those angular bounds to adapt Elkan and Hamerly exactly. In other words, the right implementation mental model is “**store bounds in angle, compute assignments with dots**.” citeturn7view2turn8search0

For Hamerly, the exact bookkeeping is the spherical analogue of the Euclidean case. Let \(a_i\) be the current assigned centroid for point \(x_i\). Store an upper angular bound
\[
u_i \ge d_\angle(x_i,c_{a_i})
\]
and a lower angular bound
\[
l_i \le \min_{j\neq a_i} d_\angle(x_i,c_j).
\]
Let
\[
s_j=\frac12\min_{k\neq j} d_\angle(c_j,c_k)
\]
be the half-separation of centroid \(j\). Then point \(i\) can be skipped if
\[
u_i \le \max(l_i, s_{a_i}),
\]
because neither the second-best centroid bound nor any centroid-center half-separation permits a reassignment. After centroid updates, if centroid \(j\) moved by
\[
m_j=d_\angle(c_j^{\text{old}},c_j^{\text{new}}),
\]
then the bounds remain exact under
\[
u_i \leftarrow u_i + m_{a_i},
\qquad
l_i \leftarrow \max(l_i - m_{\max}, 0),
\]
where \(m_{\max}=\max_j m_j\). This is exact for the same reason it is exact in Euclidean Hamerly: by the triangle inequality, non-assigned centers can only get closer by at most their movement, and the assigned center can only get farther by at most its movement. The spherical paper’s core contribution is precisely that this Euclidean reasoning survives on the sphere once you use angular distance. citeturn8search0turn17search1

A practical Rust-oriented pseudocode sketch looks like this:

```text
normalize data and centroids
initialize a[i], u[i], l[i]

repeat:
    compute centroid-centroid angular distances
    s[j] = 0.5 * min_{k != j} angle(c[j], c[k])

    for each point i:
        u[i] += movement[a[i]]
        l[i] = max(l[i] - max_movement, 0)

        if u[i] <= max(l[i], s[a[i]]):
            continue

        # refresh exact distance to assigned center
        u[i] = angle(x[i], c[a[i]])
        if u[i] <= max(l[i], s[a[i]]):
            continue

        # full rival scan
        best = a[i]
        best_ang = u[i]
        second = +inf
        for each centroid j != a[i]:
            if best_ang <= 0.5 * cc[best, j]:
                continue
            ang = angle(x[i], c[j])
            if ang < best_ang:
                second = best_ang
                best_ang = ang
                best = j
            else if ang < second:
                second = ang
        a[i] = best
        u[i] = best_ang
        l[i] = second

    recompute centroids
until stop
```

The easiest way to make this robust is to store the bounds and centroid-movement angles in f64 even if assignments still use f32 dot products. Hamerly’s memory is so low that there is no reason to risk “almost exact” pruning because of bound roundoff. This also minimizes the chance that a borderline comparison suppresses a needed rival check. citeturn8search0

On expected impact, the evidence is strong that exact pruning can be dramatic, but **direct evidence for dense 128d embeddings is weak**. The spherical paper benchmarks sparse text and dense 300d word embeddings, not dense 128d transformer embeddings, so nobody should promise a universal 10×. In your regime, a realistic expectation is roughly **2× to 8×** wall-clock speedup on the assignment step once centroids stabilize, with the low end on overlapping clusters and the high end on well-separated ones. Because your current runs are still hitting `max_iter`, the first several iterations will prune less effectively; the later iterations are where Hamerly will make its money. That is why better stopping criteria should be implemented at the same time. citeturn8search0turn18view0

I would **not** make Yinyang your first v2 implementation. The original Yinyang paper shows order-of-magnitude speedups for Euclidean k-means by grouping centers and combining local and group-level bounds, and the spherical acceleration paper explicitly notes that a Yin-Yang variant would be attractive because it can trade off pruning against RAM. But I did not find a mature, directly deployable spherical Yinyang implementation with the same clarity of formulae and engineering guidance as the Hamerly adaptation. That makes it a strong **v2.2** candidate, not a v2 blocker. citeturn10view1turn8search0

I also would not prioritize annular/Exponion-style accelerators here. Those methods are strongest when radial information relative to the origin is informative; on the unit sphere, all points and all normalized centroids sit at essentially the same radius, so a large part of that geometry degenerates. Tree-based exact acceleration is similarly unconvincing for your use case: cover-tree hybrids are promising for Euclidean k-means, but that line of work is not yet a drop-in spherical solution; and for maximum inner product search more broadly, hardware-efficient blocked brute force can beat sophisticated MIPS indices on some realistic workloads. For EmbeddingCluster, that argues for “**prune the exact solver, and make the dense kernel fast**” rather than “bolt on a general MIPS index.” citeturn10view2turn8search3turn9search2

The implementation complexity for cosine Hamerly is moderate: roughly **600 to 1,000 lines of Rust** including radius/separation precompute, tests against the baseline solver, telemetry, and edge-case handling. The biggest risks are invalid bounds after empty-cluster repair, branch-heavy code that loses to the baseline when pruning is poor, and subtle numerical bugs at skip thresholds. Those risks are manageable if the v2 rollout includes a “shadow mode” that runs Hamerly and baseline Lloyd on the same seed for small datasets and asserts identical assignments and objective values. citeturn8search0turn17search1

## Mini-batch updates and convergence

Mini-batch spherical k-means is worth adding, but **not** before exact Hamerly. The literature already contains a clean online spherical update: given a winning cluster \(k(x)\), update
\[
\mu^{\text{new}}_{k(x)}
=
\frac{\mu_{k(x)}+\eta x}{\|\mu_{k(x)}+\eta x\|}.
\]
The same paper also shows that the apparently more Euclidean-looking form
\[
\frac{\mu+\eta(x-\mu)}{\|\mu+\eta(x-\mu)\|}
\]
is equivalent after renormalization. So renormalization does **not** destroy the streaming property; it is part of the correct update on the sphere. citeturn14view0turn15view1

For a true mini-batch implementation, the best design is to maintain an **unnormalized running resultant** \(r_k\) and treat the published centroid as
\[
\mu_k = \frac{r_k}{\|r_k\|}.
\]
For a mini-batch \(B\), assign points using the current \(\mu_k\), then compute per-cluster batch sums
\[
s_k=\sum_{x\in B,\ a(x)=k} x.
\]
Update with
\[
r_k \leftarrow (1-\rho_k)r_k + \rho_k s_k,
\qquad
\mu_k \leftarrow \frac{r_k}{\|r_k\|},
\]
where \(\rho_k\) is either a streaming-average weight like
\[
\rho_k=\frac{|B_k|}{N_k+|B_k|}
\]
using the effective historical count \(N_k\), or a Robbins-Monro schedule such as \(\rho_k=t_k^{-\alpha}\) with \(\alpha\in[0.5,1]\). This is the spherical analogue of incremental k-means, but expressed in the correct sufficient statistic for directional data: the resultant vector. It parallelizes better than per-sample updates and is much easier to batch on CPU. The literature supports the underlying online update; the mini-batch collapse to per-cluster resultants is the natural engineering generalization. citeturn14view0turn15view1

If you want one concrete v2 default, I would use a **batch-resultant update**, not pure per-sample Winner-Take-All. Give it a default batch size of **4,096** for CPU work at \(d=128\), expose a range of 1,024 to 8,192, and let users tune upward if they are throughput-oriented. I am flagging that batch-size guidance as **engineering judgment rather than strongly published evidence**. The closest hard public reference point is that scikit-learn presents mini-batch k-means as the faster large-scale alternative for \(n>10{,}000\), with some quality loss that is often small in practice. citeturn18view1

For convergence, I would retire “maximum centroid angular shift only” as the primary criterion. Three different ecosystems point in the same direction: scikit-learn uses center movement; the R `skmeans` package uses **minimum relative objective improvement per iteration**; and `movMF` checks **relative log-likelihood and/or parameter change**. For spherical k-means, the most operationally useful criterion is the objective
\[
J_t = \sum_i x_i^\top \mu_{a_i}^{(t)},
\]
because this is the quantity the fixed-point solver is actually maximizing. citeturn18view0turn19view0turn36view2

My recommendation is a two-part stop rule:

\[
\frac{J_t - J_{t-1}}{\max(|J_{t-1}|,\varepsilon)} < 10^{-4}
\]
and
\[
\text{churn}_t
=
\frac{1}{n}\sum_i \mathbf{1}[a_i^{(t)}\neq a_i^{(t-1)}]
< 10^{-3}
\]
for **two consecutive iterations**.

Use maximum centroid angular shift only as a guardrail, not the main stop signal. If you still want a centroid-shift fallback, stop on
\[
1-\mu_k^{(t-1)\top}\mu_k^{(t)} < 10^{-5}
\]
or \(10^{-4}\) when combined with the two tests above. Your current threshold \(1-\cos\theta<10^{-6}\) implies \(\theta\approx \sqrt{2\times 10^{-6}}\approx 1.4\times 10^{-3}\) radians, about **0.081 degrees**; for large \(n\), that is a very strict requirement and completely consistent with “never triggers before `max_iter`.” The two-consecutive-iterations rule protects you against shallow objective plateaus where a few borderline points continue to flip. citeturn18view0turn19view0turn36view2

The implementation cost here is modest: **100 to 150 lines** for good stopping logic and telemetry, and **300 to 500 lines** for a mini-batch solver that reuses your assignment kernel. The risk is not correctness but expectations: mini-batch will almost certainly improve wall-clock convergence, but it will not be a drop-in replacement for your exact solver when users care about reproducible fixed-\(K\) purity. citeturn18view1turn14view0

## Dimensionality reduction choices

Keep randomized PCA as the **default** reduction method, and add Matryoshka truncation as a **fast path** for models that natively support it. The current embeddings API explicitly supports shortening `text-embedding-3-*` vectors via the `dimensions` parameter, and the launch material states that a 256-dimensional shortened `text-embedding-3-large` embedding can still outperform a full 1,536-dimensional `text-embedding-ada-002` vector on MTEB. That is strong evidence that prefix truncation is viable as a production feature when the model is trained for it. citeturn22view1turn22view2

The original Matryoshka Representation Learning paper is encouraging, but it is not the exact comparison you need. It shows that MRL preserves quality for classification and retrieval and can produce much smaller representations at similar accuracy, but it does **not** give a direct clustering-vs-PCA answer for embedding clustering. More recent compression work is also mixed: one 2025 paper argues that direct Matryoshka truncation inevitably loses information and introduces adaptive dimension selection because dimension importance is not strongly tied to coordinate position, while another 2025 study finds that classification and clustering embeddings often tolerate very aggressive naive truncation with small degradation because those tasks are highly redundant. In short: the literature says “truncation can work surprisingly well,” not “prefix-128 will beat PCA-128 for your task.” citeturn23view1turn23view0turn23view4turn25view0

That means the product decision is straightforward. If the user is **creating new embeddings** from a model that supports Matryoshka shortening, offer
`reduction="matryoshka"` as an option because it removes PCA fit time, removes PCA projection cost, and lowers storage and memory from the start. If the user already has full-dimensional embeddings, or is mixing providers, or wants a model-agnostic pipeline, keep PCA as the default. In other words: Matryoshka is a **provider-specific ingestion optimization**; PCA is still your **portable clustering preprocessor**. citeturn22view1turn22view2turn25view1

Among model-agnostic alternatives, PCA remains the best v2 answer. PCA is the optimal linear variance-preserving reduction, and randomized PCA is specifically the large-scale approximation used by modern libraries. A 2019 comparison of randomized PCA and random projections found that when accuracy matters, RPCA should usually be preferred over random projection for downstream learning; scikit-learn’s PCA docs likewise present randomized PCA as the efficient large-matrix route. So I would **not** spend v2 time on SparseRandomProjection as a quality-first alternative. It is a speed/footprint escape hatch, not a likely purity improver for this use case. citeturn25view1turn26view0

My concrete API recommendation is to expose three modes:

\[
\texttt{reduction} \in \{\texttt{"none"},\ \texttt{"pca"},\ \texttt{"matryoshka"}\}
\]

with `pca` as default, `matryoshka` only for supported models, and `none` when the user already truncated upstream. If you have to choose one 128-dimensional default for v2 without fresh A/B data, choose **PCA-128**; if you want a low-friction fast path, offer **prefix-128** as an opt-in and benchmark it against PCA-128 on a 10k to 20k holdout. Based on the mixed literature, I would expect prefix truncation to be **competitive**, but I would not assume it is better for fixed-\(K\) clustering. That last point is an informed inference, not a published theorem. citeturn22view1turn25view0turn23view4

## Quality improvements

The cleanest path to a real quality gain is **multi-view fusion**, not a second clustering family. The multi-view clustering literature has made the same basic point for years: when two views are complementary, learning a clustering that respects both views can outperform single-view clustering, and that result appears for partitioning methods as well as spectral ones. The modern `mvlearn` package even includes multiview spherical k-means and multiview spectral clustering as first-class algorithms, which is a good signal that your problem is canonical rather than exotic. citeturn30search0turn27search1turn29view0turn28search18

For EmbeddingCluster, the practical fusion rule I would use is **weighted late fusion in feature space**. Let \(z_A\) and \(z_B\) be separately normalized, optionally separately PCA-reduced views. Form
\[
\tilde z = \big[\sqrt{w_A}\,z_A,\ \sqrt{w_B}\,z_B\big],
\qquad
z=\frac{\tilde z}{\|\tilde z\|}.
\]
Running spherical k-means on \(z\) is equivalent to maximizing a weighted sum of per-view cosine similarities, because
\[
z_i^\top c_k
\propto
w_A\, z_{A,i}^\top c_{A,k}
+
w_B\, z_{B,i}^\top c_{B,k}.
\]
This gives you a principled, easy-to-document objective, keeps the solver unchanged, and avoids the fragility of “re-embed centroids” tricks. Since your own experiments already suggest the richer view only adds a little over description-only, the winning setup may well be \(w_A \approx 0.7,\ w_B \approx 0.3\) rather than \(0.5/0.5\). The literature supports the value of multiple views; the exact weighting policy above is the engineering recommendation I would ship. citeturn30search0turn27search1

I would **not** prioritize local-search refinement for v2 unless you see many small clusters. The classic spherical k-means local-search work showed that first-variation and Kernighan-Lin-style chains can rescue poor local optima, but it also emphasizes that the biggest qualitative failures happen when clusters are small, on the order of a few dozen documents. Your chapter-level regime has far larger effective cluster sizes, so this is more likely to be code complexity than a meaningful purity lift. It is a nice optional refinement later, not where I would bet the next month of engineering. citeturn31search1turn31search0turn19view0

I also would not chase spectral clustering, BERTopic, or Top2Vec as the main fixed-\(K\) chapter baseline. Spectral clustering is attractive when clusters are highly non-convex, but its standard formulation still depends on an affinity matrix or graph construction stage that is much less natural for 323k-item fixed-\(K\) document embedding clustering. BERTopic’s official pipeline is an embedding-driven topic-model stack with dimensionality reduction, clustering, and c-TF-IDF-based topic representation; it also supports online, semi-supervised, supervised, and manual topic modeling. Top2Vec similarly builds topics by embedding documents, reducing dimensionality with UMAP, and finding dense regions with HDBSCAN before constructing topic vectors. Those are useful topic-discovery ecosystems, but they solve a more open-ended problem than “recover 98 administratively defined chapters as faithfully and quickly as possible.” citeturn32search0turn34search1turn34search0turn34search8turn34search10turn33view1

The honest answer to “is 58.6% the ceiling?” is that there is **no clean theoretical upper bound** for unsupervised recovery of an external administrative taxonomy from semantic embeddings alone. What the external evidence does say is that richer directional models such as vMF mixtures can outperform plain spherical k-means on some text corpora, especially with overlap and imbalance, and that multi-view clustering can improve over single-view clustering when views are complementary. That points to **modest but real** headroom. If you want one expectation to plan around, I would budget for a plausible **+1 to +5 purity-point** improvement from view fusion and light model refinement, not a miraculous +20. That estimate is intentionally conservative and should be treated as a planning prior, not a published benchmark. citeturn37view1turn36view1turn30search0

## Scaling vMF and engineering decisions

Your soft vMF refinement does **not** need an \(n\times K\) resident responsibility matrix. The EM updates for a mixture of vMFs only require the sufficient statistics
\[
N_k=\sum_i r_{ik},
\qquad
S_k=\sum_i r_{ik}x_i,
\]
with
\[
r_{ik}=p(k\mid x_i,\Theta)\propto \pi_k f(x_i\mid \mu_k,\kappa_k).
\]
Then
\[
\hat\pi_k = \frac{N_k}{n},
\qquad
\hat\mu_k=\frac{S_k}{\|S_k\|},
\]
and \(\hat\kappa_k\) comes from the resultant-length equation. That means you can compute responsibilities in chunks, accumulate \((N_k,S_k)\), accumulate the log-likelihood, and throw the chunk away. The memory is then \(O(BK + Kd)\), not \(O(nK)\). For example, with \(B=8192\) and \(K=200\), the chunk responsibility buffer in f64 is only about \(8192\times 200\times 8 \approx 13\) MB. citeturn36view2

That makes the v2 recommendation simple: implement two refinement modes. The first is `hard_vmf`, which inserts the hardening step
\[
r_{ik} \leftarrow \mathbf{1}\!\left[k=\arg\max_j r_{ij}\right]
\]
between E and M. The second is `soft_vmf_chunked`, which is your current soft EM rewritten to stream chunk statistics instead of storing all responsibilities. Banerjee et al. explicitly describe both soft and hard moVMF, and the runtime table shows that hard-moVMF becomes **dramatically** faster as \(K\) grows—for example, on News20 at \(K=20\), the reported times are about 29.08 seconds for hard-moVMF versus 3,368 seconds for soft-moVMF. At the same time, their quality results show that soft-moVMF often does best on harder overlapping text sets, while hard-moVMF often yields the best separated clusters. That makes hard-vMF an excellent low-memory default refinement and chunked soft-vMF a premium option. citeturn36view2turn37view1turn38view0turn37view0

I would **not** put variational or hierarchical vMF inference into v2 unless you specifically want hierarchy or temporal sharing. The Bayesian and hierarchical vMF literature is promising and uses fast variational methods, but those models are solving a richer problem than “cheap fixed-\(K\) refinement after spherical k-means.” For your current product shape, chunked EM gives almost all the scalability benefit for much less code. citeturn36view1

On PCA engineering, I do think a high-performance dense linear algebra backend is worth the dependency. The `faer` crate explicitly positions itself as a high-performance Rust linear algebra library for medium and large dense matrices, which is exactly what your \(Y=X\Omega\) and QR steps are. Because your current PCA implementation uses custom Gram-Schmidt and custom matmul, the dense linear algebra path is the part most likely to benefit from a specialized backend. I would start with `faer` before going all the way to platform BLAS, because it preserves the “Rust-first” story while still giving you blocked kernels and better dense-matrix infrastructure. citeturn40search0turn40search3

On SIMD, the right move is “**verify, then specialize**.” The Rust performance book recommends `-C target-cpu=native` because it can improve runtime when the compiler finds vectorization opportunities; the Rust docs also show that `target-feature=+avx2` or `target-cpu=native` are the standard ways to enable those paths; and tools such as `cargo-asm` and `cargo-show-asm` are explicitly meant to inspect the generated assembly or LLVM IR. In practice, that means you should first confirm that your 128-wide dot kernel is already emitting AVX2/FMA code on your deployment CPUs. Only if the emitted kernel is clearly suboptimal would I add an explicit SIMD path for exactly \(d=128\). citeturn41search0turn41search1turn41search2turn41search3turn41search4

On memory layout, I would **keep the point matrix row-major** and make the assignment step more GEMM-like through blocking and centroid packing. Your assignment phase is logically a dense \(X C^\top\) multiply followed by an argmax. In that workload, contiguous point rows are useful because each point is consumed as a unit across many centroids; switching the whole dataset to column-major would make per-point access strided. The better optimization is to pack centroids into aligned tiles each iteration and compute point-block × centroid-block products in cache-sized chunks. `faer`’s own docs are a reminder that column-major storage and padding help dense algebra internally, but I would use that insight inside the kernel rather than change your public storage contract. The last sentence is an engineering inference, not a direct external result. citeturn40search2turn40search0

Finally, on interruptibility, keep the GIL released for almost all of `fit()`, but give Python regular chances to process signals. PyO3’s docs are very explicit here: `Ctrl-C` only sets a flag, and long-running Rust code should call `Python::check_signals()` regularly because that is how `KeyboardInterrupt` is surfaced. The safest design is to reacquire briefly once per outer iteration—or once per a few large point blocks in mini-batch mode—call `check_signals()`, and release again. That preserves throughput while making the library feel native in Python. citeturn39search0turn39search1

## Competitive positioning

The public ecosystem around embedding clustering is surprisingly polarized. On one side are **topic-model stacks** such as BERTopic and Top2Vec, which are designed to discover a variable number of semantically coherent topics, often with human-readable topic representations and hierarchical operations. On the other side are general-purpose k-means implementations, which typically treat cosine clustering, PCA, and directional refinement as separate concerns. BERTopic’s own docs emphasize its embedding-plus-topic-representation pipeline and its online, semi-supervised, supervised, and manual operating modes; Top2Vec’s docs emphasize automatic topic discovery, hierarchical topics, and dense-region detection after UMAP/HDBSCAN. Neither is the same product as “fast, exact fixed-\(K\) embedding clustering with directional diagnostics.” citeturn34search1turn34search0turn34search8turn34search10turn33view1

That is exactly where EmbeddingCluster can win. If v2 ships an exact cosine Hamerly backend, practical stopping rules, a PCA/Matryoshka toggle, weighted multi-view fusion, and hard/soft chunked vMF refinement, you have a much clearer story than “another topic model.” The story becomes: **fixed-\(K\) embedding clustering for users who already know the granularity they want, care about speed, care about representatives and concentration diagnostics, and want a solver that is honest about objective and runtime.** The closest mature prior art I found is split across older or separate packages rather than integrated into one Python-facing stack: the R `skmeans` package for spherical k-means and the R `movMF` package for vMF mixtures are examples of that fragmentation. citeturn19view0turn36view2

If you want one sentence of positioning to guide the roadmap, it is this: **be the default choice for large fixed-\(K\) embedding clustering, not for open-ended topic discovery**. That means speed first, but not speed alone. Your best differentiators are: exact cosine clustering at 100k–500k scale, predictable fixed-\(K\) behavior, built-in quality diagnostics for directional data, and optional probabilistic refinement that does not require users to leave the pipeline. With that framing, the Hamerly implementation and chunked vMF refactor are not just “engineering cleanups”; they are the center of the product. citeturn8search0turn36view2turn34search1turn33view1