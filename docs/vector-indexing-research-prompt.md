# Deep Research Prompt: rustcluster Vector Indexing Engine

> ## Context
>
> I maintain `rustcluster`, a Rust-backed Python clustering library on PyPI (v0.6.0). It ships pure-CPU wheels for Linux/macOS/Windows via maturin + PyO3. Algorithms today: KMeans (Lloyd + Hamerly), MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, and `EmbeddingCluster` (L2-normalize → randomized PCA → spherical K-means, with optional vMF EM refinement).
>
> Beyond clustering itself, the library has accumulated infrastructure that is directly relevant to vector indexing:
>
> - **`ClusterSnapshot` (v2)** — frozen centroids + per-cluster calibration (variances, percentile thresholds, vMF concentration) saved as safetensors + JSON. Supports `assign`, `assign_with_scores` (confidence, rejection), Mahalanobis boundaries, drift reports, and `HierarchicalSnapshot` (cascading two-level assignment).
> - **`EmbeddingReducer`** — standalone PCA transformer (faer-backed GEMM, randomized SVD, optional Matryoshka truncation).
> - **faer 0.24** — SIMD-optimized GEMM, used in PCA. Available throughout the crate.
> - **Three-layer architecture** — PyO3 boundary (validation, GIL release, dtype dispatch) → algorithm logic (ndarray) → hot kernels (raw `&[f32]`/`&[f64]` slices for autovectorization).
> - **Persistence** — safetensors for tensors, JSON for metadata. Working well; reused across snapshots and reducer.
> - **Concurrency** — rayon throughout. GIL released during all heavy compute.
> - **f32/f64 dispatch** — every algorithm supports both, no silent upcasts.
>
> Validated on 323K CROSS ruling embeddings (text-embedding-3-small, 1536d→128d) and 312K supplier embeddings.
>
> ## What I Want to Build
>
> A new module — working name `rustcluster.index` — adding **CPU-first approximate nearest neighbor search** to the library. This is not a FAISS replacement. The goal is a focused, readable, embedding-workload-optimized vector search layer that integrates cleanly with the existing clustering pipeline.
>
> Proposed scope:
> - `IndexFlatL2`, `IndexFlatIP` (exact baseline)
> - `IndexIVFFlat` (train with rustcluster KMeans, inverted lists, configurable nprobe)
> - `ProductQuantizer` (standalone), then `IndexIVFPQ`
> - `IndexHNSW` (M, ef_construction, ef_search)
> - Persistence via safetensors + JSON (consistent with existing snapshot format)
> - sklearn/FAISS-flavored Python API
>
> Out of scope for v1: GPU, distributed, FAISS format compat, binary vectors, mmap optimization, full filter language.
>
> I need concrete, evidence-based research before committing to a design. Where you state a number, cite the source. Where you recommend a path, commit to it — don't just lay out tradeoffs.
>
> ## What I Need Researched
>
> ### 1. Rust Vector Search Ecosystem (Wrap-vs-Build Survey)
>
> For each viable Rust crate, give: maintenance status (last release, commit cadence), license, API style, what it implements, what it lacks, and current production users.
>
> Specifically:
> - **`hnsw_rs`** (Cerebras) — HNSW. How idiomatic? Is it usable from Rust as a library, or is it tied to its own CLI?
> - **`instant-distance`** (Cloudflare) — HNSW. API quality? Production track record?
> - **`usearch`** — has Rust bindings. Maturity? License (Apache)? Performance vs FAISS?
> - **`qdrant`'s internal HNSW** — extracted/extractable as a library? License?
> - **`granne`** (Spotify, archived) — useful as a reference?
> - **`hora`** — multi-algorithm (HNSW, IVF, PQ). Maturity?
> - **`milli`** / **Meilisearch internals** — anything reusable?
> - **`candle`** (HuggingFace) — does it expose ANN primitives or just tensor ops?
>
> For each: would I be better off wrapping it or reimplementing? Be opinionated. The ranking criterion is: time-to-shipping × code-quality-of-result × license-compatibility-with-MIT.
>
> ### 2. HNSW: Wrap vs Reimplement
>
> HNSW is the largest single item in the proposed scope. A faithful production-quality implementation is months of work — neighbor selection heuristic, layered graph, dynamic ef, concurrent inserts, deletion semantics.
>
> - What does **Malkov & Yashunin (2018)** actually require, vs what's a common simplification?
> - What are the **known footguns** in implementing HNSW? (heuristic vs simple neighbor selection, layer assignment via geometric distribution, entry-point updates, edge pruning, undirected graph maintenance, reverse-edge bookkeeping.)
> - Concurrent insertion strategies — fine-grained locks per node? Lock-free? How do `hnswlib`, `usearch`, `qdrant` handle it?
> - Deletion: lazy tombstone vs full graph repair. Which crates support which?
> - Realistic recall/QPS numbers on **sift1m** and **glove-100** for: hnswlib, usearch, instant-distance, hnsw_rs, qdrant. Cite the ann-benchmarks results.
> - **Recommendation:** wrap (which crate?) or build. State your call and the reasoning. If you say build, give me an effort estimate in engineer-weeks.
>
> ### 3. IVF Design — Especially the ClusterSnapshot Connection
>
> rustcluster already has `ClusterSnapshot`: frozen centroids + calibration metadata + safetensors persistence + adaptive rejection thresholds + Mahalanobis boundaries. Conceptually, `ClusterSnapshot.assign` is already a flat search over centroids.
>
> - What does an IVF index look like if we view it as **`ClusterSnapshot` + inverted lists + database vectors**? Specifically:
>   - Should `IndexIVFFlat` literally hold a `ClusterSnapshot` for the coarse quantizer?
>   - Can users "upgrade" a snapshot they trained for classification into an IVF index without retraining?
>   - Does `HierarchicalSnapshot` map naturally to IMI (Inverted Multi-Index) or two-level IVF?
> - **Plain IVF vs IVF-with-residuals.** When does residual coding matter (mostly for IVF-PQ)? Should v1 IVF-Flat store raw vectors or residuals?
> - **Probing strategy.** Naive: pick `nprobe` nearest centroids, scan their lists. Smarter: heap-based multi-probe. What does FAISS actually do? Worth implementing for v1?
> - **Inverted list layout** — array-of-arrays vs flat with offsets vs interleaved. Cache implications. What's the standard?
> - **Cosine path.** rustcluster has spherical K-means. Is the right design for cosine: normalize-on-add → IP search → spherical K-means coarse quantizer? Or treat cosine as a separate metric throughout?
> - **Recall as a function of `nlist`/`nprobe`.** Standard rules of thumb (`nlist ≈ √n` etc.) — which are folklore vs measured?
>
> ### 4. Product Quantization
>
> - **Training.** Lloyd's per subvector vs joint optimization. When is OPQ (rotation pre-step) worth the training cost? Citation for the typical recall lift.
> - **Subvector count `m`.** How to pick? Practical defaults from FAISS docs and papers.
> - **SDC vs ADC** distance computation. Which does FAISS use by default? Performance/recall implications.
> - **Lookup table reuse** across queries — cache locality wins, when do they materialize?
> - **Quantizer training data size.** How many vectors needed? FAISS uses ~256K for IVF1024,PQ32 — is that overkill?
> - **Standalone `ProductQuantizer` vs only-as-IVF-PQ.** The proposal does standalone first. Is that ordering sensible, or should I bundle PQ training into IVF-PQ as a single milestone?
> - **Common implementation pitfalls** — codebook collapse, init strategies (k-means++ vs random), centroid tie-breaking.
>
> ### 5. Top-k Selection
>
> Often glossed over but real performance hot spot.
>
> - **Algorithms by k regime:**
>   - k ≤ ~16: insertion into sorted array? Branchless?
>   - k ≤ ~100: binary heap?
>   - k ≤ ~1000: heap or partial sort?
>   - k > 1000: full sort?
> - **SIMD top-k** — bitonic top-k literature. Is it worth the complexity for our use case?
> - **Fused distance + top-k** kernels — when do they win over separate compute + select?
> - What does FAISS use? hnswlib? Are there standard Rust crates (e.g., `min-max-heap`, `keepcalm`)?
>
> ### 6. Distance Kernels and faer Integration
>
> rustcluster already uses faer for PCA GEMM.
>
> - **Batched search** (queries × database) is GEMM-shaped: `Q (nq×d) @ X^T (d×n)`. Use faer GEMM directly?
> - **Single-query search** is a GEMV. Different code path? When does it matter?
> - **L2 distance via the GEMM trick** (`||q-x||² = ||q||² + ||x||² - 2 q·x`) — precompute `||x||²`, fuse with GEMM. Numerical concerns?
> - **PQ distance kernels** — table lookups dominate. SIMD gather instructions vs scalar lookup. Which architectures?
> - **Cache blocking** for large databases (n > 1M). Standard tile sizes?
>
> ### 7. Persistence and Format Design
>
> rustcluster uses safetensors + JSON for snapshots. I want to extend this, not invent a new format.
>
> - **What are the natural tensor decomposition** for each index type? (e.g., flat: just the matrix; IVF: centroids + list-offsets + list-vectors + list-ids; PQ: codebooks + codes; HNSW: graph as flattened adjacency + level vector + entry point.)
> - **Versioning.** How do FAISS, hnswlib, usearch, qdrant handle index format versioning? Forward/backward compat strategies?
> - **Index portability.** Can a save on x86 be loaded on aarch64? What breaks (endianness, alignment)?
> - **Lazy/mmap loading** — out of scope for v1 per the requirements. But will the format choice paint me into a corner? Should the file layout leave room for mmap later?
>
> ### 8. API Design — IDs, Filters, Adds/Deletes
>
> The requirements doc is silent on these and they bite v1 users.
>
> - **External IDs.** FAISS supports `add_with_ids` for non-sequential IDs. hnswlib too. What's the minimum viable v1 — sequential 0..N-1 only, or external `u64` IDs?
> - **Filtered search.** Post-filter (search top-Nk, filter, return k) vs pre-filter (skip during traversal). What do qdrant/usearch/hnswlib offer? Minimum hooks needed in v1 to not paint into a corner?
> - **Add after train (IVF)** vs **incremental add (HNSW/Flat)** — what's the typical API contract? Does `add` after `train` invalidate the index in any way?
> - **Delete semantics.** Hard delete vs tombstone. Should v1 punt entirely on delete? What do users actually need?
> - **API conventions to follow.** FAISS-style (`Index*` classes, `train` / `add` / `search`) is the de facto standard. Worth deviating?
>
> ### 9. Concurrency
>
> - **Query-level parallelism** — embarrassingly parallel across queries via rayon. Any gotchas with the GIL release pattern PyO3 needs?
> - **Concurrent inserts (HNSW).** What schemes do existing Rust HNSW crates use? Lock per node? RwLock on the whole graph? Lock-free?
> - **Reader-writer split.** Is it sane to support reads concurrent with inserts? FAISS doesn't really; qdrant does.
> - **Thread safety of `search` while another thread mutates the index** — what's the standard contract?
>
> ### 10. Benchmarking Methodology and Realistic Targets
>
> - **Datasets.** sift1m, glove-100, deep1b subset, sift10m, msmarco. Which are standard for ann-benchmarks? Where to source?
> - **Methodology.** ann-benchmarks Recall@k vs QPS curves. Single-threaded vs multi-threaded reporting. Build time treatment.
> - **Realistic v1 targets.** What's a reasonable bar for a new from-scratch Rust ANN library?
>   - Flat: should match FAISS within ~5% (it's just a GEMM).
>   - IVF-Flat: at recall@10 = 0.95, what QPS is realistic vs FAISS at the same recall?
>   - HNSW: at recall@10 = 0.95, what's reasonable vs hnswlib? vs usearch?
> - **Hardware to benchmark on.** Apple Silicon (M-series), x86 with AVX2, x86 with AVX-512. Which matters most for embedding workloads?
>
> ### 11. Module Boundary and Naming
>
> - **Submodule (`rustcluster.index`) vs sibling crate (`rustcluster-index`)** — pros/cons. Binary size growth. Release coupling. Compile time. What did sklearn / scipy / sktime do for analogous splits?
> - **Naming** — `rustcluster.index`, `rustcluster.vector`, `rustcluster.ann`, `rustcluster.search`. Which signals scope most clearly?
> - **Cargo features.** Should HNSW / PQ / IVF be optional features to keep the base small?
>
> ### 12. Milestone Ordering
>
> The proposal has: Flat → IVF-Flat → Persistence → HNSW → PQ.
>
> - Is that the right order? My instinct is to move Persistence earlier (forces good API shape) and bundle PQ with IVF-PQ (standalone PQ has limited utility on its own).
> - What's the **smallest v1** that ships meaningful value? (e.g., Flat + IVF-Flat + persistence — is that enough to be useful, or does the absence of HNSW/PQ make it a non-starter?)
> - **Critical path.** Which milestone has the most unknowns and should be de-risked first?
>
> ### 13. Final Recommendation
>
> Pull it all together:
>
> - **Wrap-vs-build calls** for HNSW, PQ, top-k selection, persistence — give a verdict on each.
> - **Recommended scope cut for v1** — what to ship, what to defer. Be opinionated.
> - **Recommended milestone order** with effort estimates in engineer-weeks per milestone.
> - **Module name.**
> - **Honest assessment of risk** — where is this most likely to go wrong? What's the probability the project ships in 8 weeks vs 6 months?
> - **Is this worth doing at all** given existing Rust crates? Or is the answer "use `usearch` + your existing `ClusterSnapshot` and call it a day"?
>
> ## Deliverables
>
> For every section:
> 1. Concrete numbers, not vague assessments. Cite sources (papers, ann-benchmarks runs, GitHub repos with line numbers where relevant).
> 2. Code examples or pseudocode for any non-obvious algorithm or API choice.
> 3. Honest maturity/risk assessment for every external dependency you recommend.
> 4. A final ranked recommendation with tradeoffs explicitly stated, not hedged.
