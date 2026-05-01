# Purpose-Built Embedding Clustering Algorithm for rustcluster

For dense semantic embeddings at the 100K–1M scale, the most defensible implementation path for `rustcluster` is **not** to start with BERTopic-style UMAP→HDBSCAN as the core engine. The strongest engineering case is a **three-stage pipeline**: **L2 normalize → reduce dimension linearly to about 64–256 dimensions → run spherical clustering**, with **exact spherical K-means** as the primary hard-clustering engine and **vMF mixture refinement** as an optional soft-clustering layer when posterior membership, per-cluster concentration, or information-criterion model selection matters. The literature and current tooling point to that combination as the best balance of speed, objective alignment for cosine geometry, and scalability on dense embeddings; UMAP/HDBSCAN remains valuable as a separate “automatic topic discovery / outlier-aware” mode rather than the default million-point workhorse. citeturn25view0turn22view0turn32search0turn39view0turn42view0turn44search0

My implementation recommendation, in order, is: first, **exact spherical K-means** with good seeding and optional cosine-aware pruning; second, **randomized PCA** in Rust; third, **mini-batch spherical K-means** for very large corpora; fourth, an **optional vMF EM refinement** initialized from spherical K-means; and fifth, a separate **topic-modeling mode** that exposes UMAP/HDBSCAN/c-TF-IDF-style labeling for users who want BERTopic-like behavior. The evidence for this ordering is strong for the first two items, good-but-more-conditional for the third, and mixed for making vMF or UMAP/HDBSCAN the default path. citeturn13view0turn26view0turn32search7turn37search3turn38view0turn42view0

## Spherical objectives and vMF mixtures

The exact spherical K-means problem assumes unit-normalized data \(x_i \in \mathbb{S}^{d-1}\) and unit-norm centers \(\mu_j \in \mathbb{S}^{d-1}\). Its standard objective is

\[
\max_{\{z_i\}, \{\mu_j\}} \sum_{i=1}^n x_i^\top \mu_{z_i},
\]

which is equivalent to minimizing average cosine dissimilarity,

\[
\min \sum_{i=1}^n \left(1 - x_i^\top \mu_{z_i}\right),
\]

and, because both \(x_i\) and \(\mu_j\) have norm \(1\), also equivalent to minimizing squared Euclidean distance on the sphere,

\[
\min \sum_{i=1}^n \|x_i - \mu_{z_i}\|_2^2
= 2 \sum_{i=1}^n \left(1 - x_i^\top \mu_{z_i}\right).
\]

This is **not** the same as taking ordinary Euclidean K-means on normalized points and merely “interpreting” it as cosine clustering, because standard K-means updates an unconstrained centroid \(c_j = \frac{1}{|C_j|}\sum_{i\in C_j} x_i\), whose norm varies by cluster. Under that objective, assignment depends on \(1 + \|c_j\|^2 - 2 x^\top c_j\), so cluster norm affects decisions; spherical K-means removes that degree of freedom by re-normalizing the mean direction after every update. That re-normalization is the mathematically important step, not an implementation detail. citeturn4view0turn25view0turn20view0

For fixed assignments, spherical K-means updates each center by taking the cluster resultant and re-normalizing it:

\[
r_j = \sum_{i:z_i=j} x_i,\qquad
\mu_j = \frac{r_j}{\|r_j\|_2}.
\]

For fixed centers, the optimal assignment is simply

\[
z_i = \arg\max_j x_i^\top \mu_j.
\]

That gives the exact Lloyd-style alternating procedure:

```text
Input: unit-normalized X ∈ R^{n×d}, k
Initialize unit centers μ1,…,μk
repeat
    for each point xi:
        zi ← argmaxj xi · μj
    for each cluster j:
        rj ← Σi:zi=j xi
        if ||rj|| > 0:
            μj ← rj / ||rj||
        else:
            reseed μj
until assignments or objective stop changing
```

Because each assignment step optimizes the objective for fixed centers and each update step optimizes the objective for fixed assignments, the objective improves monotonically and the procedure converges to a local fixed point, just like Lloyd’s algorithm. What I did **not** find in the primary spherical K-means literature is a practically useful sharp convergence-rate theorem beyond this local monotonicity / finite-assignment argument; that absence matters if you want to promise iteration bounds. citeturn4view0turn25view0turn20view0

The role of centroid normalization is substantial. If you skip it, you are no longer solving the spherical problem. In directional data this tends to favor tighter clusters with larger resultant norms and to penalize more diffuse but semantically coherent groups. The literature is much stronger on the mathematical necessity of re-normalization than on clean modern dense-embedding ablations comparing “cosine assignment without center normalization” against true spherical K-means. In other words: the theoretical reason to normalize is strong; the direct evidence on present-day embedding corpora is thinner. citeturn25view0turn20view0

Initialization remains one of the biggest practical levers. The canonical Dhillon–Modha era work predates ordinary K-means++ and therefore does **not** give you a modern ++ answer. More recent work shows that spherical variants of K-means++ exist, and approximation papers show that K-means++-style and K-means||-style seeding can be transferred to \(\alpha\)-spherical formulations that restore useful geometric guarantees. But the empirical evidence is not uniformly one-sided: in the cosine-accelerated spherical K-means study by entity["people","Erich Schubert","data mining researcher"] and coauthors, the final objective differences among uniform random, spherical K-means++, and AFK-MC2-style seeding were usually small, and on 20 Newsgroups K-means++ was actually up to about **7–8% worse** than uniform random, plausibly because of anomalies and outliers. That makes a strong case for exposing several seeders rather than hard-coding a single “best” choice. citeturn15view0turn16view0turn13view0

The main failure modes of spherical K-means on embeddings are predictable. It assumes roughly centroidal clusters and cannot model different concentration levels except through cluster size. It is sensitive to bad seeding, empty clusters, outliers, and topic continua where there is no clear angular basin. It will also split a large semantically broad topic into several directional modes if that improves the cosine objective, and it will happily force every point into some cluster even when an outlier label would be the right answer. This is precisely where vMF mixtures or HDBSCAN-type methods have an advantage. citeturn20view0turn24view0turn39view0

The acceleration story is better than many people realize. The 2021 “Accelerating Spherical K-Means” paper adapted **Elkan** and **Hamerly**-style pruning to cosine similarity using a triangle inequality for cosine geometry. On the RCV1 benchmark, their runtime table shows the standard spherical algorithm at **6,064,203 ms** for \(k=200\), versus **547,110 ms** for Elkan and **474,800 ms** for simplified Elkan, a speedup of roughly **11–13×**. But the same paper also shows that there is **no universal winner**: on some higher-dimensional / less favorable shapes, the pruning overhead weakens or reverses the gains, and the Elkan variants can become memory-heavy because they store many bounds. That is highly relevant to 768d–1536d dense embeddings: pruning is viable, but you should treat it as data-dependent and benchmark-gated, not as a guaranteed drop-in speedup. citeturn13view0turn12view2

vMF mixtures give the probabilistic analogue of spherical K-means. A \(d\)-dimensional vMF component has density

\[
f(x\mid \mu,\kappa)=C_d(\kappa)\exp(\kappa \mu^\top x),
\qquad \|\mu\|=1,\;\kappa\ge 0,
\]

with normalizer

\[
C_d(\kappa)=\frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}.
\]

A \(k\)-component mixture is

\[
p(x)=\sum_{j=1}^k \alpha_j f(x\mid \mu_j,\kappa_j),
\qquad \alpha_j>0,\;\sum_j \alpha_j=1.
\]

This is the hyperspherical analogue of a Gaussian mixture model: \(\mu_j\) is the mean direction, \(\kappa_j\) is the concentration, and \(\alpha_j\) is the mixture weight. Banerjee, Dhillon, Ghosh, and Sra explicitly show that spherical K-means arises as a special case of EM on a vMF mixture under additional restrictions, which is one of the cleanest theoretical justifications for cosine-style clustering. citeturn26view0turn25view0turn20view0

The EM equations are straightforward once written in log-space. For the E-step,

\[
r_{ik}=p(z_i=k\mid x_i,\Theta)
\propto \alpha_k \, C_d(\kappa_k)\exp(\kappa_k \mu_k^\top x_i).
\]

For the M-step,

\[
\alpha_k = \frac{1}{n}\sum_i r_{ik},\qquad
R_k = \sum_i r_{ik}x_i,\qquad
\mu_k = \frac{R_k}{\|R_k\|}.
\]

Then define the mean resultant length

\[
\bar r_k = \frac{\|R_k\|}{\sum_i r_{ik}},
\]

and solve

\[
A_d(\kappa_k)=\frac{I_{d/2}(\kappa_k)}{I_{d/2-1}(\kappa_k)}=\bar r_k.
\]

Banerjee et al. propose the high-dimensional approximation

\[
\hat \kappa_k \approx \frac{\bar r_k d - \bar r_k^3}{1-\bar r_k^2},
\]

and Sra later showed that starting from this approximation and doing a small number of Newton updates gives more accurate \(\kappa\) estimates at similar cost. Soft EM monotonically increases incomplete log-likelihood; the hard-assignment analogue maximizes a lower bound instead and is much cheaper in time and memory. citeturn25view0turn22view0turn26view0turn19view2

A practical vMF EM loop looks like this:

```text
Input: unit-normalized X, k
Initialize αk, μk, κk
repeat
    # E-step
    for each point xi and cluster k:
        log rik ← log αk + log Cd(κk) + κk (μk · xi)
    normalize rik with logsumexp over k

    # M-step
    αk ← (1/n) Σi rik
    Rk ← Σi rik xi
    μk ← Rk / ||Rk||
    rbar_k ← ||Rk|| / Σi rik
    κk ← solve Ad(κk) = rbar_k
until likelihood change < tol
```

For dense embedding workloads, the dominant cost per iteration is still the \(O(nkd)\) dot products, exactly as in spherical K-means, but soft vMF adds an \(O(nk)\) responsibility matrix and \(O(nk)\) normalization work. At \(n=10^6\) and \(k=200\), responsibilities alone are about **800 MB in fp32** or **1.6 GB in fp64** before any scratch buffers. That is the single biggest reason not to make full soft vMF your default for million-vector clustering. citeturn25view0turn22view0turn24view0

What Banerjee et al. found is still relevant for modern embeddings. Hard and soft vMF outperformed simpler baselines on harder directional clustering tasks, and the ability to learn per-cluster \(\kappa\) was especially helpful when clusters overlapped, were skewed in size, or were small relative to dimension. They also observed that soft vMF began with diffuse posteriors and then “hardened” over iterations, behaving like an adaptive annealing process. That is exactly what you would want for dense embeddings where there is often a smooth semantic continuum early in optimization and sharper thematic islands at convergence. But they also measured the computational bill very directly: on News20 with 20 clusters, hard moVMF took **29.08 s** while soft moVMF took **3368 s** in their environment. The relative scale would differ today, but the qualitative gap remains important. citeturn20view0turn24view0

The numerical challenge in vMF is the Bessel term. Hornik and Grün’s `movMF` work, Sra’s note, and SciPy’s special-function documentation all point in the same direction: **do not** evaluate \(I_\nu(\kappa)\) naively for high \(\nu\) or \(\kappa\). Use either the ratio \(A_d(\kappa)\), asymptotic expansions, or **exponentially scaled** Bessel functions such as `ive`, and keep likelihood computations in log-space. This is not optional at \(d=768\) or \(d=1536\). A robust Rust implementation should therefore avoid raw Bessel calls in the inner loop entirely and either use Banerjee/Sra approximations for \(\kappa\) or implement a stable ratio evaluator plus a few Newton steps. citeturn18view0turn26view0turn17search3

Scalable vMF variants do exist, but the evidence is weaker than for plain spherical K-means. There are Bayesian and hierarchical vMF models with **variational inference** and collapsed Gibbs methods, notably the 2014 vMF clustering models paper, and there is general theory for **mini-batch EM on exponential-family mixtures**, which should apply to vMF in principle. What I did **not** find is a widely adopted, canonical, production-grade mini-batch vMF clustering algorithm for dense text embeddings analogous to MiniBatchKMeans. That suggests vMF is a good **optional refinement layer** in `rustcluster`, but not the first algorithm to build. citeturn28view0turn29view0

## Dimensionality reduction and choosing k

For dense embeddings, the strongest default reduction method before clustering is **linear** reduction, not nonlinear manifold learning. Randomized PCA gives you a low-rank approximation that preserves the largest-variance directions, runs much faster than full SVD when you only need a small number of components, and is substantially easier to reason about than UMAP in a clustering pipeline. Halko, Martinsson, and Tropp’s randomized SVD framework is the canonical reference, and current scikit-learn documentation still recommends randomized SVD specifically when you want a good approximate truncated decomposition at low target rank. citeturn32search0turn32search7turn32search11

Random projection gives the clearest theory but the theory is extraordinarily conservative for your use case. The Johnson–Lindenstrauss bound used in scikit-learn is

\[
m \ge \frac{4\log n}{\varepsilon^2/2 - \varepsilon^3/3}.
\]

Plugging in \(n=500{,}000\) yields about **11,248 dimensions** for \(\varepsilon=0.1\), **3,028** for \(\varepsilon=0.2\), and **1,458** for \(\varepsilon=0.3\). For \(n=10^6\), \(\varepsilon=0.1\) gives about **11,841** dimensions. Since your original embedding dimension is 768 or 1536, the worst-case JL guarantee often does **not** certify an aggressive compression at all. That does not mean RP fails in practice; it means JL is too loose to justify “reduce 1536 → 64” as a theorem. For embedding clustering, RP is best viewed as a very fast heuristic baseline, not as the mathematically strongest preprocessing path. citeturn36view0

UMAP occupies a different niche. It is often useful before HDBSCAN because it creates a low-dimensional manifold where density clustering becomes feasible, but the UMAP documentation explicitly warns that UMAP does **not** preserve density and can create “false tears,” i.e., split what was a single structure into finer clusters than really exist. That is exactly the wrong failure mode if your default product promise is “stable semantic grouping.” BERTopic succeeds with UMAP because it is optimizing for topic discovery and outlier handling, not because UMAP is the safest generic preconditioner for clustering. This is why I would separate “fast scalable embedding clustering” from “topic-modeling pipeline” in your API. citeturn32search2turn39view0turn42view0

The literature does **not** give a universal answer to “what is the optimal target dimension for clustering embeddings?” The strongest evidence is model- and corpus-dependent, not universal. Two signals matter here. First, modern embedding systems increasingly support dimension shortening explicitly; the current OpenAI embeddings guide says `text-embedding-3-small` is 1536-dimensional by default and supports a `dimensions` parameter to reduce size while preserving concept-representing behavior. Second, Matryoshka representation learning shows that some models can be trained so that prefix truncations remain competitive at much lower dimensions. Those are strong hints that many modern embedding spaces are overprovisioned for downstream clustering, but they are **not** a general proof that 32 or 64 dimensions are always enough. citeturn31search1turn31search5turn31search0turn31search8

My practical recommendation is therefore empirical and corpus-specific: expose target dimensions like **32, 64, 128, 256**, but make **128** the default for clustering and **64** the aggressive-speed option. For dense semantic embeddings, 128 dimensions is usually enough to preserve a great deal of topic structure while making assignment and centroid updates an order of magnitude cheaper than at 1536 dimensions. I am labeling this as an engineering recommendation rather than a settled literature constant because the available evidence is suggestive rather than universal. citeturn32search0turn31search1turn33view0

For choosing \(k\), I would separate the methods by whether you are fitting a centroidal model or a density model. If you are fitting spherical K-means or vMF, **information criteria on a vMF mixture** are the most principled route because they align with the likelihood you are actually optimizing. There is directional-clustering literature on model selection for vMF mixtures, but I did not find a strong modern consensus that one criterion universally dominates. Gap statistic is classical and general, but it is computationally expensive because it requires many reference re-fits. At 500K+ points, it becomes hard to justify as a default unless you amortize via minibatching or subsampling. citeturn46search2turn46search3

Silhouette remains useful, but only if you compute it in the **same geometry as the clustering objective**. The silhouette formula itself is generic,

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))},
\]

and scikit-learn supports arbitrary metrics, including cosine. So for normalized embedding clustering, **cosine silhouette** is not a hack; it is the right way to align the metric with the geometry. What is less convincing is computing Euclidean CH or DB scores on severely concentrated normalized embeddings and treating them as absolute truth. In high-dimensional directional spaces, those centroid-and-scatter metrics are much easier to misread. citeturn49search2turn49search6

BERTopic’s “automatic \(k\)” is really HDBSCAN’s automatic extraction of stable clusters after UMAP. That has clear benefits: you get automatic outlier labeling, no need to pre-specify \(k\), and a topic count driven by density stability. Its failure modes are also very clear in the official docs: stochastic UMAP makes results vary across runs unless you fix `random_state`; low `n_neighbors` or low topic-size thresholds can create many micro-topics; high thresholds can collapse topics; and `calculate_probabilities=True` becomes both slow and memory-heavy on large corpora. If you offer a BERTopic-like mode, these should be front-and-center in the docs, because they are not edge cases. citeturn39view0turn42view0

There is real research interest in using the spectrum of a similarity graph or Laplacian to estimate a natural cluster count, and a recent short-text embedding paper goes directly in that direction with a cohesion-based spectral estimator. Conceptually this fits embeddings well. Practically, the bottleneck is graph construction: a dense \(n\times n\) similarity matrix is impossible at 500K–1M, so any eigengap approach must be based on a sparse approximate k-NN graph or on sampling. I would treat eigenspectrum-based \(k\) estimation as an experimental feature, not as your first auto-\(k\) implementation. citeturn46search5turn49search3

## System landscape and scalability

BERTopic’s exact default pipeline is now well documented: embeddings, then **UMAP** with `n_neighbors=15`, `n_components=5`, `min_dist=0.0`, `metric='cosine'`, then **HDBSCAN** with `min_cluster_size=15`, `metric='euclidean'`, `cluster_selection_method='eom'`, and `prediction_data=True`, then bag-of-words, then **c-TF-IDF**, with optional representation tuning through models such as KeyBERT-inspired methods or LLM-based labelers. c-TF-IDF works by aggregating all documents in a cluster into one document, L1-normalizing term frequencies, and weighting terms by a class-based IDF derived from term frequencies across clusters. This is a powerful pipeline for interpretable topic labels, but it is doing significantly more than clustering embeddings. citeturn39view0

Why BERTopic is slow is also unusually explicit in its own docs. Embedding generation is expensive unless you have a GPU. UMAP is stochastic and can be memory-hungry; the docs suggest `low_memory=True` to mitigate that. HDBSCAN probability calculation is particularly expensive, and the docs recommend disabling `calculate_probabilities` for large datasets because it can create a huge document-topic matrix and significantly increase runtime. That is strong evidence that BERTopic should be treated as a **topic-modeling system** rather than as the baseline to beat for pure embedding clustering throughput. citeturn42view0

Top2Vec differs from BERTopic mainly in representation strategy, not in its large-scale bottlenecks. Its documented algorithm is: create jointly embedded document and word vectors, reduce document vectors with UMAP, find dense areas with HDBSCAN, then compute each topic vector as the centroid of the original-dimension document vectors inside each dense area. In other words, it still leans on UMAP and HDBSCAN for automatic topic discovery, but unlike BERTopic it defines topics directly in the shared embedding space rather than through c-TF-IDF. That is elegant when you have both document and word embeddings, but it does not solve the million-scale density-clustering bottleneck. citeturn43search0turn43search5

entity["company","Meta","social tech company"]’s FAISS is immediately relevant to your design. Official FAISS docs expose K-means / clustering as a first-class building block, with GPU support, multiple restarts, and a `spherical` flag that L2-normalizes centroids at each iteration. The FAISS library paper also states directly that FAISS uses spherical K-means for some maximum-inner-product / IVF-related settings. FAISS’s IVF coarse quantizer is not a semantic topic model on its own, but it is highly relevant as a **coarse partitioner** or assignment accelerator. If you want a scalable Rust-backed clustering stack, FAISS is the most important external reference for what “production embedding clustering” looks like in practice. citeturn44search0turn44search1turn44search5turn44search20

entity["company","NVIDIA","gpu computing company"]’s cuML is strongest today for GPU-accelerated Euclidean K-means, UMAP, and HDBSCAN. BERTopic’s documentation explicitly recommends cuML’s GPU UMAP and HDBSCAN to speed up the topic-modeling pipeline, and NVIDIA has published a benchmark showing cuML HDBSCAN reducing clustering time on a 5-dimensional UMAP projection from **382 s to 92 s**—about **4.1×**—with larger speedups reported in higher dimensions. What cuML still does **not** appear to expose cleanly is a native cosine/spherical K-means API; current documentation for KMeans does not show a metric parameter, and open feature requests ask for cosine support in both KMeans and HDBSCAN. That means GPU acceleration today is easiest for a BERTopic-style density pipeline or for Euclidean K-means on normalized embeddings, not for a true native spherical implementation. citeturn42view0turn45search6turn45search8turn45search2turn45search3

The current sentence-transformers stack offers examples for **K-means**, **agglomerative clustering**, and a “fast clustering” / local-community approach. The docs state that agglomerative clustering becomes too slow beyond a few thousand sentences, while the fast clustering example is tuned for about **50K sentences in less than 5 seconds**. That is useful, but it is not a full-purpose clustering library for 500K–1M embeddings. The same docs point users toward BERTopic and Top2Vec for topic modeling. In other words, sentence-transformers gives good examples and utilities, not the system architecture you want to ship in `rustcluster`. citeturn48view0

On Rust-native tooling, the ecosystem is currently stronger on **linear algebra**, **PCA**, and **embedding generation** than on dedicated embedding-clustering systems. `linfa-reduction` already ships PCA plus Gaussian and sparse random projection; `faer` and `nalgebra` ship SVD machinery; `efficient_pca` now exposes an explicit randomized PCA `rfit`; and there is even `rsvd-faer` for randomized SVD on `faer` matrices. By contrast, I did **not** find an obvious Rust-native equivalent of BERTopic, Top2Vec, or FAISS’s clustering stack aimed specifically at dense semantic embeddings. That is an opportunity for `rustcluster`, especially if you keep the core design simple, geometry-aligned, and fast. citeturn37search3turn37search0turn37search1turn38view0turn37search14

## Evaluation, labeling, and hierarchies

For intrinsic evaluation of embedding clusters, the most useful metrics are the ones that stay in embedding geometry. **Cosine silhouette** is absolutely appropriate for normalized embeddings and is directly supported by standard tooling because silhouette is defined from a distance function, not from Euclidean geometry specifically. For spherical / vMF-style models, I would also compute **average intra-cluster cosine similarity**, **average centroid similarity**, and the **resultant length** of each cluster,

\[
R_j = \frac{1}{|C_j|}\left\|\sum_{i\in C_j} x_i\right\|_2,
\]

because \(R_j\) is directly tied to directional concentration and to \(\kappa_j\) in a vMF interpretation. A high-quality embedding cluster should generally have high internal cosine agreement and a resultant that is noticeably above the corpus baseline. That last criterion is especially important because semantic corpora are often anisotropic. citeturn49search2turn25view0turn26view0turn33view0

If you have ground-truth labels, the standard external metrics remain **purity**, **NMI**, **AMI**, and **ARI**. Purity is simple but artificially increases as \(k\) grows. NMI scales mutual information to \([0,1]\) but is not chance-corrected. AMI adjusts for chance, and ARI adjusts the Rand index for chance while allowing negative values for worse-than-random agreement. For a clustering library, the right default external-metric trio is probably **NMI + AMI + ARI**, with purity as an optional descriptive diagnostic. citeturn49search1turn49search5turn49search9turn49search13

On topic coherence, the formulas are useful only if you still have the text. Classic **UCI coherence** is PMI-style co-occurrence over a sliding window; **NPMI** normalizes PMI to \([-1,1]\); and **C\_v** is a composite measure built from indirect confirmation and cosine similarity over context vectors. Röder et al.’s coherence framework is still the key reference here, and more recent work continues to treat \(C_v\), NPMI, and UCI as the baseline family. If you have only embedding vectors and no original text or token counts, these metrics are not directly available. That is a strong argument for letting `rustcluster` optionally hold lightweight per-cluster lexical statistics when users cluster text embeddings, because otherwise you cannot compute the most standard topic-model quality metrics afterward. citeturn49search8turn49search0turn49search4

There is no universally accepted “good” absolute value for these metrics on modern embedding corpora. Silhouette is in \([-1,1]\), with values near **1** meaning strong separation and values near **0** meaning overlap, but semantic datasets often have fuzzy topic boundaries, so even modest cosine-silhouette values can correspond to useful clusters. Likewise, NMI/AMI/ARI values that would look mediocre on a clean benchmark may be quite strong on multi-topic, weakly labeled document collections. This is one area where the literature is frustratingly non-portable: scores do not transfer cleanly across corpora, embedding models, or topic granularity. Treat ranges as diagnostics, not as pass/fail thresholds. citeturn49search2turn49search6turn49search1turn49search5

For semantic representatives, the fastest choice is **nearest-to-centroid** on the unit sphere:

\[
x^\star_j = \arg\max_{x_i \in C_j} x_i^\top \mu_j.
\]

If you need a representative that is itself robust to cluster shape and outliers, use the **cosine medoid** instead, i.e., the item with the highest average cosine to the rest of the cluster. Centroid representatives are cheaper and align naturally with spherical K-means and vMF; medoids are safer if clusters are elongated or if you want an actual document exemplar guaranteed not to sit in a sparse direction. For topic browsers and document organization, I would expose both. citeturn25view0turn39view0

For labeling, BERTopic’s c-TF-IDF remains one of the most practical post-clustering strategies because it combines cluster assignments with cluster-level term weighting, and the BERTopic docs show that large-model labelers are often feasible if you run them only on representative documents per cluster. That is the right design pattern for `rustcluster` too: first compute representative documents or medoids; then either run c-TF-IDF / BM25-style keyword extraction or feed a small set of representatives to an optional LLM labeler. A centroid embedding by itself is not enough to recover words reliably. If the original text is unavailable, there is no dependable embedding-only inverse map that will magically produce “keywords.” citeturn39view0

For hierarchies, the main design decision is whether to build the tree **top-down** or **bottom-up**. Bottom-up agglomerative clustering over all documents has an intuitive hierarchy, but the document-level version is fundamentally the wrong tool at 500K–1M because the time and memory costs approach \(O(n^2)\). Sentence-transformers’ own docs already warn that agglomerative clustering is practical only for a few thousand sentences. For embedding-scale systems, top-down recursive partitioning is the more realistic architecture: bisect or \(k\)-split a cluster only when it is large enough and only when the split improves a directional quality criterion. citeturn48view0

The stopping rules that make most sense for recursive embedding splits are: minimum cluster size, maximum depth, minimum improvement in cosine silhouette or directional likelihood, and a concentration-based rule such as “do not split if the parent cluster already has high resultant length and the split barely increases it.” If you later add vMF, a likelihood-ratio, BIC, or held-out criterion for “split vs. do not split” is even cleaner. I am presenting these as engineering recommendations rather than a universally settled literature recipe, because the exact thresholds are use-case dependent. Still, they map naturally onto the geometry of embeddings and are much easier to justify than arbitrary fixed-depth recursion. citeturn22view0turn46search3

## Rust implementation blueprint

For a Rust implementation, the numeric hot path for normalized embeddings should be **dot products, not generic cosine similarity**. Once each embedding row is L2-normalized, cosine similarity reduces to \(x^\top y\), and your inner loop becomes a straight fused multiply-add accumulation over `&[f32]`. For 1536-dimensional vectors, an `f32` accumulator is usually fast enough for assignment, but centroid accumulation should use **`f64` running sums** or compensated summation, because those sums add thousands of points and are much more sensitive to drift than a single point-center dot product. This is one of the highest-leverage low-level choices you can make. citeturn13view0turn18view0

A robust centroid normalization routine should compute the accumulated squared norm in `f64`, check for `norm <= eps`, and then either reseed or recycle a center. Zero-norm centers can arise from empty clusters, from exact cancellation, or from numerical nonsense if a cluster is tiny and nearly antipodal. The safe pattern is: accumulate in `f64`, normalize once, then cast the finished center back to `f32` for the assignment loop. Do **not** repeatedly renormalize partial sums within the same update. That loses information about cluster mass and can destabilize minibatch updates. citeturn25view0turn26view0

For vMF specifically, the stable rule is: never compute \(I_{d/2-1}(\kappa)\) directly if you can avoid it. Estimate \(\kappa\) from \(\bar r\) using Banerjee’s approximation, optionally refine with one or two Newton steps using Sra’s derivative formula, and compute the log normalizer with a scaled-Bessel or asymptotic routine if you need exact-ish log-likelihoods. In practice that means your Rust code should have a dedicated small module for **Bessel-ratio / log-normalizer** evaluation rather than leaning on a generic special-functions crate and hoping for the best at \(d=1536\). citeturn22view0turn26view0turn17search3

On layout, the best default for your use case is **row-major, one embedding per row**, because the main operation is iterating over points and computing dot products against centers. That layout also lines up naturally with Python / NumPy contiguous `float32` arrays passed through `PyO3` or `numpy` bindings. If you later add a GEMM-based assignment kernel, you can keep points row-major and either pretranspose or tile the centroid matrix to suit the backend. The right abstraction is therefore “row-major public storage, backend-tuned internal packing if needed.” I am calling this an implementation inference rather than a published benchmark result, but it is the most practical choice for a Rust-backed Python library focused on dense embeddings. citeturn37search8turn37search0turn37search1

The Rust reduction stack is already good enough to make linear preprocessing realistic inside `rustcluster`. `linfa-reduction` ships PCA and random projection. `faer` and `nalgebra` expose SVD decompositions. `efficient_pca` already offers an explicit randomized `rfit`, including a pure-Rust `faer` backend. So if you want “PCA in Rust” next quarter, that is straightforward; if you want “fast randomized PCA” next quarter, that is also realistic. A clean implementation plan is to ship **exact PCA for smallish data** and **randomized PCA for large / high-dimensional data**, with identical transform APIs. citeturn37search3turn37search0turn37search1turn38view0

For the end-to-end scalability target, the realistic answer is conditional. A pipeline of **normalize → randomized PCA to 128 → spherical K-means with good seeding → cosine-silhouette-on-sample** can plausibly run under **60 seconds for 500K embeddings** on commodity multi-core hardware if \(k\) is moderate and the implementation is cache- and SIMD-friendly. A pipeline of **raw 1536d → UMAP → HDBSCAN → probability computation** will usually not. Likewise, **full soft vMF** at million scale is feasible as an offline refinement step, but it is not the default “fast path.” That is the clearest product boundary the research supports. citeturn32search0turn42view0turn24view0turn13view0

The concrete architecture I would ship in `rustcluster` is this. The default estimator should be something like **`SphericalKMeans`** with exact cosine objective, center renormalization, K-means++ / K-means||-style seeding, and an optional accelerated mode using cosine-aware Elkan/Hamerly bounds. The large-scale estimator should be **`MiniBatchSphericalKMeans`** with epoch-level objective checks and optional final full-dataset refinement. The probabilistic estimator should be **`VMFMixture`**, initialized from spherical K-means and explicitly documented as slower but better when users want soft memberships, concentration estimates, or BIC-based model comparison. Around those, add a preprocessing module with **`RandomizedPCA`**, **`PCA`**, **Gaussian / sparse random projection**, and a separate **topic-labeling layer** that works only when original texts are available. That design is tightly aligned with the literature, with current production tools, and with the performance envelope you are targeting. citeturn25view0turn13view0turn32search0turn38view0turn39view0turn44search0