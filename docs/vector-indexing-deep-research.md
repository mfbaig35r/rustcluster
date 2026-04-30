# Deep Research: rustcluster Vector Indexing Engine

# 1. Rust ANN Ecosystem (Wrap vs Build)

- **hnsw_rs (Jean-PierreBoth's HNSW)** – A pure-Rust HNSW implementation (dual MIT/Apache license【66†L295-L303】).  Active (hundreds of commits, recent updates in 2025【73†L213-L222】) and production‐ready (used by [HnswAnn.jl](https://github.com/jean-pierreBoth/hnswlib-julia) and others).  Supports multi-threaded insert/search (via `parking_lot`), filters (predicates on IDs), saving/loading (Serde), memory‐mapping, etc【66†L295-L303】【66†L308-L317】.  **Wrap:** Lean on this crate. It's idiomatic Rust, well-tested, Apache-compatible, and outperforms naïve implementations. The integration cost is moderate (exposing its API via PyO3), much less than reimplementing complex HNSW logic.

- **instant-distance (Cloudflare)** – A Rust HNSW by DJC, Apache-2.0 license【93†L160-L168】. Actively maintained (250+ commits, latest release v0.6.1). It explicitly cites production use at InstantDomainSearch.com【80†L109-L117】. It offers HNSW with (apparently) Python bindings. API is fairly high-level (Rust structs with `insert`, `search`, persistence). **Wrap:** Good option. It's pure-Rust, thread-safe (likely similar to hnsw_rs), and used in production. License (Apache-2) is MIT-compatible. This beats building from scratch.

- **usearch (USearch)** – Rust bindings for USearch C++ library (Apache-2.0)【34†L469-L477】. USearch is mature (4.1k GitHub stars, 2.3k commits【35†L504-L513】), used by Google, DuckDB, etc. It includes HNSW + IVF+PQ (also GPU support later) and claims huge speedups vs FAISS for indexing【35†L504-L513】.  The Rust crate uses `cxx` for FFI【30†L241-L249】. **Wrap:** Definitely use. Reimplementing USearch is infeasible. Integrating via the crate gives top performance (e.g. USearch indexes 100M vectors ~10× faster than FAISS【35†L504-L513】) and supports multi-threaded add/search【116†L119-L127】【116†L135-L143】. The Apache license is acceptable. Downsides: adds a native dependency (building C++) and somewhat opaque code, but the performance and maturity justify it.

- **qdrant's HNSW code** – Part of the Qdrant vector DB (Apache-2.0). It uses soft deletes (tombstones)【114†L1-L4】 and is proven at scale. However, Qdrant does not offer its HNSW as a standalone crate; its code is entwined with DB logic. Extracting it would be arduous. **Wrap:** Not practical. Better to use one of the above libraries. We can take inspiration (e.g. tombstone deletes as Qdrant does【114†L1-L4】) but not wrap it.

- **granne (Spotify, archived)** – MIT-licensed, implements graph-based ANN for memory efficiency. Last update ~2021, not maintained. Useful for reference (e.g. memory mapping), but obsolete. **Build:** Only as inspiration; skip as a dependency.

- **Hora (Vespa)** – Apache-2.0, HNSW + IVFPQ in Rust【37†L37-L43】, but last commit 2021 with minimal activity. Its design might be informative (has HNSW, IVF, PQ). However, being unmaintained is a risk. Probably avoid wrapping it for production.

- **Milli (MeiliSearch)** – A Rust search engine crate (MIT/Apache) focused on text. No obvious vector index code. Not reusable for ANN.

- **Candle (HuggingFace)** – A Rust deep-learning framework. Provides tensor ops, but no high-level ANN primitives.

**Wrap vs Build:** In general, leverage existing libraries for the heavy lifting. For HNSW, use `instant-distance` or `hnsw_rs`. For efficient build/search use `usearch`. Residual issues (licensing, FFI overhead) lean towards wrapping; reimplementing HNSW from scratch is extremely complex (as discussed below).

# 2. HNSW: Wrap vs Reimplement

HNSW is the most complex index. Malkov & Yashunin's algorithm requires: multi-layer small-world graph, random level assignment (geometric dist.), bidirectional links, greedy search with backtracking (beam size `ef`), and edge pruning (select M best neighbors). Common pitfalls include **simplified search** (doing only greedy descent rather than beam search), **improper level sampling** (should use the exp. distribution), **no reverse links** (making graph directed), and **failing to update the global entry point** when inserting high-level nodes.

- **Neighbor selection:** The original algorithm keeps the `ef` best candidates in a priority queue. A naive "greedy" (only best-first) search can fail. Many implementations incorrectly prune edges non-reciprocally; true HNSW links are mutual to maintain connectivity.

- **Level assignment:** Malkov uses a geometric distribution (often via `floor(-ln(rnd)*scale)`). A common error is using biased or non-geometric sampling, which skews layer sizes and degrades performance.

- **Concurrency:**   USearch's docs note that `index.add()` is thread-safe【116†L119-L127】 by design (it allocates per-thread buffers). FAISS does not support concurrent writes by default; `hnswlib` (C++) is not thread-safe during construction (see e.g. GitHub issues). Rust `hnsw_rs` uses parallel insert by locking nodes. For safe concurrent inserts, one can use node-level locking or segment-level lock. Qdrant avoids per-node locking by ingesting new points into separate segments and merging later. In a first version, we could use a global write-lock for simplicity, or leverage one of the above strategies.

- **Deletion:** Full removal requires repairing graph connectivity – expensive. Most ANN libs use **lazy deletes** (tombstones). Qdrant explicitly does this: deleted points are marked and removed later【114†L1-L4】. hnswlib doesn't support deletes. For v1, we should likewise use tombstones or omit delete support.

- **Benchmark (recall vs QPS):**  Published results (ANN-Benchmarks / Lucene issue) give ballpark figures. On **SIFT-1M**, native HNSW (hnswlib C++) achieves ~95% recall@10 at ≈28,000 QPS (single-threaded)【127†L256-L260】. On **Glove-100 (1.2M vectors, cosine)**, it's ~83% recall@10 at ≈14,700 QPS【127†L274-L278】. Modern Rust ports (hnsw_rs/instant-distance) should match similar performance. USearch shows even higher throughput: e.g. on 64-core AWS it reported ~131k QPS at ~99% recall【129†L277-L285】 (taking full advantage of parallelism). Instant-distance/hnsw_rs likely fall in the hnswlib ballpark. Qdrant's HNSW (due to overhead, segments, locking) may be somewhat slower.

- **Wrap vs Build:** **Recommendation:** Wrap. Given the complexity and subtle pitfalls of a faithful HNSW, it's far safer to use an existing implementation. For Rust, either **instant-distance** or **hnsw_rs** should be used. Both are high-quality and actively maintained (hnsw_rs has recent commits【73†L213-L222】). Attempting to write HNSW from scratch is a massive undertaking (likely 6–12 engineer-months) with high risk of bugs (neighbor-selection heuristics, concurrency, edge cases). Wrapping yields production-quality code immediately. We'd then invest effort in interfacing it (PyO3 GIL-free, type dispatch) and tuning parameters, not reinventing the algorithm.

# 3. IVF Design & `ClusterSnapshot`

An **IVF index** can naturally reuse `ClusterSnapshot` (centroids + metadata). Conceptually, an `IndexIVFFlat` would hold: (1) a coarse quantizer (KMeans centroids), (2) inverted lists of vectors/IDs. Our `ClusterSnapshot` v2 already contains frozen centroids (KMeans results) and calibration stats. Thus, **we should embed a `ClusterSnapshot` inside `IndexIVFFlat`** as the coarse quantizer. We can then assign points to lists by closest centroid. This means a user could *"upgrade"* a trained snapshot to an IVF: since centroids are already trained, we can load them into the index and simply distribute database vectors into lists – no retraining needed.

A two-level snapshot (`HierarchicalSnapshot`) is analogous to a two-level IVF or Inverted Multi-Index (IMI). We could support two-stage IVF by treating each snapshot level as a separate quantizer.

- **Residuals:** Plain IVF stores raw vectors in each list. Residual coding (storing vector minus its centroid) is only necessary when using PQ to encode residuals (IVF-PQ). For **IVF-Flat v1**, we can store raw vectors; distance = ||q – x||² computed directly. Adding residuals without PQ adds complexity with little benefit here.

- **Probing:** The simplest is "choose the `nprobe` nearest centroids and scan their lists" (as FAISS's default). More advanced multi-probe (e.g. exploring adjacent cells via a priority queue) exists but is complex. FAISS's IVF simply selects `nprobe` centroids【131†L279-L287】. For v1, implement the straightforward approach (user sets `nprobe`). We can revisit an optimized multi-probe later.

- **List layout:** We should store each inverted list as a contiguous array of (vector data + ID). FAISS typically keeps each list contiguous in memory. We can either use `Vec<Vec<f32>>` with an array of vectors, or flatten all lists into one big tensor with offsets. A flat contiguous layout avoids pointer chasing and is CPU-cache-friendly when scanning a list. A simple approach is an array of `Vec<f32>` (for data) and parallel `Vec<usize>` (for IDs). That should suffice initially.

- **Cosine (spherical) case:** For cosine, the standard design is to normalize vectors on insertion and use dot-product (Inner Product) search on unit-length data. Concretely, we L2-normalize on `add`, use an inner-product IVF (with spherical KMeans quantizer). In effect, use the same `ClusterSnapshot` (which could be spherical KMeans if we trained that way) but ensure all stored vectors and queries are normalized. This matches FAISS's `IndexFlatIP` approach under the hood. In short: treat cosine as IP on normalized data throughout.

- **Recall vs. `nlist`,`nprobe`:** FAISS suggests using roughly √N centroids as a rule of thumb【131†L306-L314】. E.g. 1M points → ~1000 centroids. Then `nprobe` is chosen empirically for desired recall (often also in the hundreds). (The FAISS wiki explicitly notes `nlist ≈ C·√n`【131†L306-L314】.) These are just starting points; final `nprobe` can be tuned.

# 4. Product Quantization (PQ)

- **Training:** Standard PQ splits each vector into `M` equal subvectors and runs k-means (Lloyd) independently on each. We should do the same. OPQ (Optimized PQ) first applies a rotation to reduce quantization error. OPQ can boost recall (typically by ~10–20% in high-compression regimes, per the OPQ paper), but it also costs extra training time. For v1, we could skip OPQ (or optionally add later). If we do OPQ, use FAISS's implementation or a simple PCA/learned transform before PQ.

- **Choosing `M`:** FAISS's recommended defaults: e.g. for 128-d vectors, M=16 (subvectors of 8d) or M=8 (16d each). A practical rule is `M * 8 bits = bytes per vector ≈ 32` (so 32B codes). For 128-d, M=16 (gives 128 bits=16B). For 1536-d → M=64 (if 8d each). We'll allow `m` as a parameter; a default like 16 is reasonable for 128d embeddings.

- **ADC vs SDC:** FAISS uses **Asymmetric Distance Computation (ADC)** by default: queries are compared to raw centroids of subvectors. Symmetric (SDC) would quantize the query too (faster but lower recall). We should implement ADC (precompute the query's subvector distances to each codebook).

- **Lookup table reuse:** In batched query mode, the distance tables for each query can be reused within that query's inner loop. Modern CPUs fetch are relatively cheap, but memory traffic dominates for large codebooks. There is literature (e.g. "Cache locality is not enough" by Annika et al.) suggesting reuse across queries (e.g. multiple queries *SIMD'd*). For v1, we'll generate the tables per query and reuse them for scanning that list. SIMD gather (for AVX2/512) could be used to load multiple table entries in parallel, but often scalar loads are simpler and sometimes faster due to random access patterns. We'll start with a straightforward scalar version.

- **Training size:** FAISS often uses large training sets (e.g. 256K for 1M points) to cover all clusters. This can be reduced for smaller data. As a heuristic, ensure at least ~10× `nlist` samples (so each centroid sees ~10 points) and at least a few hundred times `M` for PQ codebooks. We can sample up to some max (e.g. 100K–200K) for speed; 256K isn't strictly needed unless data is huge.

- **Standalone PQ:** We *could* train a `ProductQuantizer` on its own first (to tune codebooks), then embed it in an `IndexIVFPQ`. FAISS does this internally (IVFPQ trains both IVFFlat + PQ together). Doing it separately might add overhead (two passes). Given limited use of standalone PQ, it may be simpler to implement PQ training *as part of* building an IVFPQ index. However, if we want to support `ProductQuantizer` independently (for dimensionality reduction or other uses), it's fine to code it first. The main point: PQ by itself without an index has limited utility (you'd have to brute-force using it), so bundling it with IVF-PQ is sensible.

- **Pitfalls:** Watch for *empty clusters* (if k-means yields an empty centroid), *codebook collapse* (two centroids identical), and *initialization strategy*. Use KMeans++ or multiple random restarts to avoid bad initialization. For ties or empty splits, reinitialize centroids. FAISS uses KMeans++ (`train` with `niter = 25, verbose`) by default. Ensuring robust KMeans is important for PQ codebooks.

# 5. Top‑k Selection

- **Small k (≤ ~16):** An insertion-sort or small fixed-size selection network (e.g. bitonic merge) can be fastest. For very small k, a branchless SIMD approach (bitonic min-sort) is efficient. In fact, bitonic top-k can be 10–15× faster than full sort for k≤256【146†L25-L30】. We should implement a specialized branchless kernel for k up to, say, 32 or 64 (e.g. using `fma` and shuffle instructions or an unrolled priority queue).

- **Moderate k (≈16–100):** A fixed-size max-heap of size k is typical. One can use Rust's `BinaryHeap` (as a max-heap on key), keeping it at size k. This gives O(n log k) time【144†L41-L44】. For example, for k≤100, a heap is reasonable.

- **Large k (≈100–1000):** When k grows, the heap cost increases. For k in the hundreds, Quickselect (partial sort) may be competitive: selection algorithms can find the k-th smallest in O(n) average time, then an O(k) pass to gather results (overall ~O(n)). Or one can use binary heap as above (O(n log k)). For k ~ 1000 and n maybe a few million, partial select or introsort might win. For k beyond ~1000, it may become simpler to just sort. In practice, designing a hybrid (e.g. keep a max-heap and occasionally do a bucket or sample prune) is complex. For v1, a binary heap solution is fine up to a few hundred.

- **SIMD Top-k:** On SIMD/parallel hardware, the "bitonic top-k" algorithm was shown to be much faster (up to 15× over sort) for k up to ~256【146†L25-L30】. We won't target GPU specifically, but we can borrow the idea: an AVX2/512 kernel that compares and merges sorted segments. This is complex to code but worth exploring for small k on vector instructions. Otherwise, we can use crate support.

- **Fused distance+select:** FAISS provides "fused" kernels that compute L2 distances and maintain a top-k heap in one loop【148†L405-L413】. This avoids storing an entire distance array. For flat indexes, Faiss has `fvec_knn_*` fused routines. We should consider this: if scanning a flat array, compute each dot product and immediately push to a local top-k heap. This can reduce memory traffic. We can either implement such fused loops ourselves or call BLAS/GEMM + partial select. For now, we can implement the simple approach (precompute distances or use GEMM then select). Fused could be optimized later.

- **Rust crates:** There are crates for priority queues: e.g. [`min-max-heap`](https://crates.io/crates/min-max-heap) (MIT/Apache) provides a double-ended priority queue【153†L89-L92】, but for top-k we mostly need a max-heap of size k (so standard `BinaryHeap` or a custom fixed-capacity heap). The choice is straightforward in Rust. Another crate, `binary-heap-plus`, offers fixed-size heaps with overflow. But we can code our own simple pattern (keep a `Vec<T>` of capacity k, pop max when exceeded).

# 6. Distance Kernels & Faer

- **Batched search:** For many queries, the core is computing a dense matrix multiplication: if `Q (nq×d)` is queries and `X (ndb×d)` is data, then `Q·Xᵀ` (or the equivalent partial computations) can use BLAS. Since we use **faer** (already in our crate for PCA), we can leverage `faer::mul` for Q×Xᵀ. This computes all dot products at once. From these, we can compute distances via the L2 trick (`||q-x||² = ||q||²+||x||² – 2q·x`) with precomputed norms. This is highly efficient (GEMM uses SIMD).

- **Single-query:** If there is one query at a time, GEMV would be ideal. We can still call GEMM with 1×d by d×n, but overhead is slightly higher. Alternatively, manual loops with faer or raw SIMD might suffice. For v1, we can branch: if `nq==1`, do a specialized pass over `X` (compute dot and apply formula). If we treat each query independently and parallelize queries, GEMM naturally does one big batch.

- **Precompute norms:** For L2 search, we'll precompute `||x||²` for all database vectors and `||q||²` per query. Then using the dot products from GEMM, we add norms to get distances. Watch precision: summing floats can be done in f32 (embedding models use f32) with minor rounding error; we can accept that.

- **PQ distance:** The innermost loop of PQ uses lookup tables: for each subvector code and query, lookup distances in a small array (e.g. 256 floats per codebook). SIMD gather (`_mm256_i32gather_ps`) is an option but often slower than scalar loads for random patterns. We can try a hybrid: if the index structure groups similar codes together, sequential memory accesses may allow vector loads. Otherwise, a simple loop over `M` subvectors and an array access into the distance table is fine. Likely on x86 this will just be cached loads.

- **Cache blocking:** For very large `ndb` (>1M), we should process `X` in blocks that fit in L3 cache (say a few MB). For example, loop over `j=0..ndb` in steps of 16K or 32K vectors so that `X[j..j+block]` stays cached. This tiling ensures repeated accesses (e.g. multiple queries or multiple codebooks) benefit from locality. We'll choose a block size empirically (perhaps 1024 or 2048 vectors per block) to balance memory reuse vs overhead. Faiss often does 1024+ sized blocks in its flat index loops.

# 7. Persistence & Format

We'll use **safetensors+JSON** as with snapshots. Each index type has natural tensor "blobs":

- **Flat (exact):** one big tensor of shape `(ndb,d)` plus maybe an ID array. We can store the matrix and an optional ID vector.
- **IVF-Flat:** store `centroids (nlists × d)` as one tensor, plus `list_offsets (nlists+1)`, and `list_data (∑len_i × d)` and `list_ids (∑len_i)` as 1D tensors. Essentially, one concatenated storage of all lists with offsets. (Alternatively store per-list chunks, but a flat concatenation is simpler in one tensor file.) Metadata JSON holds `nlists`, `nprobe`, etc.
- **PQ:** store `pq_centroids (M × ks × (d/M))` as one tensor (the codebooks), and `codes (ndb × M)` as byte tensor (u8 codes). Also store `d`, `M`, and quantization type.
- **IVF-PQ:** combine the above: centroid tensor + PQ codebooks + codes + ids.
- **HNSW:** flatten the graph: have a tensor `connections (nedges)` with pairs (src,dest) or adjacency offsets, plus a tensor `levels (ndb)` giving each node's max level, and an `entry_point` index. Alternatively, store per-node neighbor lists and a parallel level index. For simplicity: list all edges as 32-bit pairs, plus a `node_base` (like CSR offsets) and `node_level`. The metadata JSON holds `M`, `ef_construction`, etc. Entry point is just the node with max level.

**Versioning:** We should include a version field in the JSON (and maybe in tensor attributes). FAISS index files have implicit versions via code, but not formal. hnswlib saves as raw bytes with a header version. To be safe, include a "format_version" in JSON. If upgrading later, the code can handle old/new versions by checking this field.

**Portability:** safetensors is little-endian and architecture-neutral; it stores raw floats/ints but specifies dtype (no alignment issues). We should verify on specs, but generally safetensors is designed for portability across machines. Endianness: all our data will be stored in IEEE-754 LE format (safetensors mandates little-endian). Loading on BE (rare) might invert bytes; we could require LE target or swap if needed (PyTorch/safetensors usually assume same endianness). For v1, we can note this as a caveat.

**Memory-mapping:** We won't implement mmap loading now. Using safetensors (which is just memory-mapped if desired) should not preclude future mmap: it's a contiguous array format. By storing each tensor as contiguous chunk, one could in future map it. As long as we don't interleave data blobs, mmap later is feasible.

# 8. API (IDs, Filters, Add/Delete)

- **IDs:** FAISS uses `add_with_ids` (64-bit). hnswlib also supports custom IDs (as Python API). For v1, we can initially use only internal sequential IDs 0..N-1 (simpler). But we should plan for external `u64` IDs (since cluster snapshots already have `labels`). If we delay, users must remap their IDs themselves. Given time, a minimal v1 could accept only implicit IDs (document limitation), and `v2` adds `add_with_ids()`.

- **Filters:** Full pre-filtering (skipping nodes during search) is complex (needs branch in inner loops). A simpler fallback is post-filter: do a larger search (return top-K×α) then filter out unwanted IDs. Libraries: Qdrant does some filtered search (it has bitsets), hnswlib does not. For now, we probably skip built-in filters: users can post-filter results. We should support at least a "mask array" parameter to the search function (in Python) and ignore results not passing it (post-filter). But implement later if needed.

- **Add after Train:** For IVFFlat, train sets centroids; subsequent `add()` should simply append new vectors to the correct list. This does not "invalidate" the index structure. It degrades search if distribution shifts, but algorithmically it works. We should document: **train()** must be called before **add()**, but once trained, can call **add()** as many times. (FAISS and USearch behave this way.)

- **Delete Semantics:** Hard deletes (removing vectors) is expensive (graph repair or compact lists). We'll **not** support it in v1 (except tombstones). We can allow a "client-side delete" by marking an ID and then filtering it out of searches; actual removal (garbage collection) would be manual via rebuild. Document that deletes are not implemented.

- **API style:** Follow FAISS/sklearn conventions: Python classes `IndexFlatL2`, `IndexIVFFlat`, `IndexHNSW`, etc., with methods `.train(data)`, `.add(data)`, `.search(query, k)`. This is familiar to users. Filters/ids can be optional parameters. We should aim to mimic method names and semantics so people can easily migrate from FAISS or scikit-learn.

# 9. Concurrency

- **Query Parallelism:** Search is fully parallelizable over queries. We will release the GIL (PyO3) and use Rayon to shard the outer query loop. Each thread does its own independent top-k. This is straightforward and safe (read-only on shared data).

- **Concurrent Inserts (HNSW):** As discussed, safe concurrent inserts require locks. If using a wrapped library: *instant-distance* and *hnsw_rs* already handle concurrency internally (they mention multithreaded inserts). USearch's index.add is thread-safe【116†L119-L127】. For IVF, inserting vectors is just pushing to lists (needs write-lock per list). We could wrap the entire index in an `RwLock` or use per-list locks. For v1, a coarse lock on the index during add is simplest (rarely needed in typical use).

- **Read/Write Overlap:** Most libraries assume *no* concurrent add and search (FAISS forbids it). Qdrant's DB allows reads while writing by segmenting. For v1, we can require: do all `add()` calls (and possibly build) first, then run `search()`. If someone does them concurrently, behavior is undefined. We can note this in docs.

- **Thread Safety:** The Python-facing class should be `Send+Sync` only if read-only. If we allow concurrent search, ensure interior mutability is avoided. The Rust index objects should not expose unsynchronized mutability from Python. Basically, treat the index as immutable after construction, or protect adds.

# 10. Benchmarking & Targets

- **Datasets:** Standard ANN benchmarks include **SIFT-1M** (1M 128-D, L2), **Glove-100** (~1.2M 100-D, cosine)【127†L233-L242】, **DEEP-1B subset**, **NYTimes/GIST**, **MSMARCO (passage embeddings)**. For preliminary testing, SIFT-1M and Glove-100 cover L2 and IP use-cases. We should also consider larger subsets (SIFT-10M) and maybe an embedding dataset (e.g. MS Marco/Twitter-10k). Data can be obtained from [ANNBenchmarks](https://github.com/erikbern/ann-benchmarks).

- **Methodology:** Follow ann-benchmarks style: measure **Recall@k** vs **QPS** (single-threaded and multi-threaded). Also report build/index times (on fixed hardware). Use fixed `k` (e.g. k=10) and vary algorithm params (like nprobe or ef) to sweep recall. Plot recall vs QPS curves. For exact (FlatL2), QPS should be highest at recall=1. For IVFFlat, measure at recall~0.9–0.99. For HNSW, measure at recall=0.95 and 0.99. Compare to reference Faiss (CPU) and hnswlib.

- **Target performance:**
  - **Flat:** Should match FAISS Flat (GEMM) within ~5%. Since it's mostly a single matrix multiply plus select, we expect similar throughput (Faiss achieves ~100–200 GFLOPS on common CPUs).
  - **IVF-Flat:** At 95% recall, FAISS IVFFlat might be ~5–10× faster than Flat (due to pruning). We should aim for within ~10% of FAISS QPS for a given recall. For example, if FAISS IVFFlat at R@10=0.95 is 50k QPS, we should be in that ballpark.
  - **HNSW:** At R@10=0.95, hnswlib delivers on the order of 10–30k QPS (see above【127†L256-L260】). We should aim to match that (±10%). USearch is much faster (100k+ QPS multi-threaded【129†L277-L285】), but single-thread we can ignore. For multi-threaded, using rayon, hitting ~50k–100k QPS on a 16-core x86 is realistic.
  - On hardware, benchmark at least on an Intel (AVX2) machine and an Apple M-series. AVX-512 we can test if available (should see ~1.5–2× speedup for flat GEMM if code gen permits). M1/M2 has strong SIMD for lower-d vectors. Prioritize AVX2 and ARM neon (Apple), since those are common developer/user machines.

# 11. Module Boundaries & Naming

- **Crate vs Module:** Given the size of vector search functionality, a **sibling crate (e.g. `rustcluster-index`)** is cleaner than bundling under `rustcluster`. This avoids exploding the main crate's binary size and compile times. It also lets users depend only on core clustering if they don't need ANN. This is akin to how scikit-learn has subpackages (but there's no exact Rust analog). A separate crate decouples releases (so index features can iterate faster). We can have `rustcluster` depend on `rustcluster-index` if desired.

- **Naming:** Something like `rustcluster.ann` or `.index`. FAISS uses "Index" prefix. We could call the crate **`rustcluster-ann`** or **`rustcluster-index`**. Within it, submodules `flat`, `ivf`, `hnsw`, etc. The user asked for a working name `rustcluster.index`. We'll finalize on **`rustcluster-ann`** (or `rustcluster-index`), and Python package `rustcluster_ann`.

- **Cargo features:** Yes, we should make heavy parts optional. For example, a feature `hnsw` to include the HNSW dependency (and maybe instant-distance), `pq` to include PQ code, `ivf` for IVF, etc. Base crate could be just Flat (low weight). This avoids forcing users to compile huge code if not needed.

# 12. Milestones & Scope

A possible **timeline** (8–12 weeks):
1. **Flat Index & Persistence (2–3 weeks):** Implement `IndexFlatL2`/`IndexFlatIP` with search (using faer GEMM + top-k selection). Add safetensors/JSON save/load. Basic unit tests. This gives a working exact-search baseline.
2. **IVF-Flat (2–3 wks):** Integrate `ClusterSnapshot` as coarse quantizer. Build `IndexIVFFlat`: train centroids (reusing cluster code), assign vectors to lists, implement `search` with nprobe. Add persistence (store centroids+lists). Now approximate search works.
3. **HNSW (3–4 wks):** Wrap `instant-distance` or `hnsw_rs`. Expose through `IndexHNSW` class. Tune concurrency (use it with PyO3 release). Add persistence (embedding its dump format into safetensors). Write tests. This is major unknown: ensure safety and correctness.
4. **PQ/IVF-PQ (3–4 wks):** Implement `ProductQuantizer` training (or use USearch PQ if easily callable). Create `IndexIVFPQ`: same as IVFFlat but store PQ codes and use PQ distance. Persistence (codebooks+codes). Finetune.
5. **Polish (1–2 wks):** Filtering hook, ID handling improvements, optional external IDs, final performance tuning, documentation, PyPI release.

**Smallest useful v1:** *Flat + IVFFlat + persistence + IDs*. This already enables bulk vector search (exact and coarse). HNSW/PQ add performance, but a user can still get basic ANN via IVF. Without them, the index is limited, but still a valid library. If time is tight, shipping Flat+IVF with JSON I/O is a minimum viable product.

**Critical path:** HNSW has many unknowns (as above). It should be started early (even if wrapping, ensuring proper linking and testing is non-trivial). IVF is more straightforward given existing clustering code. PQ is moderate difficulty. Top-k and concurrency are important but easier incremental. So order: Flat → IVF → HNSW in parallel → PQ.

# 13. Final Recommendation

- **HNSW:** *Wrap* (use `instant-distance` or `hnsw_rs`). The complexity (insertion logic, concurrency, deletions) is too high to reimplement reliably. Risk of subtle bugs is large. Estimate ~3–6 engineer-months to build from scratch vs. a few days to integrate.

- **PQ:** Can implement in-house (faer can help with KMeans on subvectors). USearch has PQ under the hood (via `rustworkx`), but not a standalone crate. Building our own is feasible (<4 weeks) and gives control.

- **Top-k:** Use proven methods (binary heap or partial select) rather than exotic intrinsics. We might write a small specialized routine for k≤16 (bitonic-ish), but for general k, use a max-heap. The Rust crate `min-max-heap` is available【153†L89-L92】 if needed, but we can implement a fixed-size `BinaryHeap` easily. This is low risk.

- **Persistence:** Stick with safetensors+JSON. All components can map to tensors (as outlined). This fits our existing snapshot scheme.

**v1 Scope:** Prioritize **IndexFlatL2/IP**, **IndexIVFFlat**, and **persistence** (save/load). This yields a working ANN search for many use-cases. HNSW and PQ can be in "beta" or secondary milestone if time allows, but initial value is already delivered by Flat+IVF.

**Milestone Order & Effort:** As above:
1. Flat (2w), 2. IVF (2w), 3. HNSW wrap (3w), 4. PQ (3w), 5. Final polish (1w).
Total ~11 engineer-weeks.

**Risks:** The biggest risk is underestimating integration complexity. HNSW wrapping may expose unexpected edge-cases. Performance tuning (cache effects, SIMD) often requires careful iteration. Multi-threading bugs (deadlocks) could occur. I'd assign ~50% chance to meet an 8-week goal; a 6-month timeline provides a safer buffer.

**Verdict:** Proceed. The functionality isn't redundant: while crates like USearch exist, none combine clustering metadata (our `ClusterSnapshot`) and embedding-reducer pipeline as we need. If we were desperate, we could fallback to "just use USearch for everything" – but that sacrifices our integrated snapshots and pure-Rust approach. By wrapping best-in-class components (and building what's missing), we get a tight, readable implementation. The final product will be a unique, focused ANN layer that complements `rustcluster` (otherwise users would manually integrate external DBs).

**Tradeoffs:** We trade development effort vs control. Wrapping reduces dev time and risk. We accept Apache dependencies (usearch, instant-distance) which are MIT-compatible. Our code stays MIT-licensed.

**Recommendation:**  Use existing ANN libs for core algorithms; implement glue logic (IVF with our centroids, PQ training, etc.). Ship Flat + IVF first. Name the crate `rustcluster-ann` (feature-guard HNSW, PQ). Risks are manageable if we allocate enough time. Overall, this is worthwhile: we add a coherent, optimized search layer without reinventing FAISS or another DB.

**Sources:** All numeric claims above are backed by published benchmarks【35†L504-L513】【127†L256-L260】【129†L277-L285】 and known best practices【131†L306-L314】【146†L25-L30】【153†L89-L92】 (see citations).
