# EmbeddingCluster v2 Research Findings

Synthesized from deep research (2026-04-20). Source: `docs/embeddingcluster v2-development-strategy-deep-research.md`

---

## Implementation Order (Confirmed by Research)

1. **Cosine Hamerly accelerator** for spherical K-means
2. **Practical stopping rules** + per-iteration telemetry
3. **Chunked vMF refinement** with hard-vMF option
4. **Weighted multi-view fusion** of embedding views A and B
5. **Matryoshka fast path** + faer PCA backend
6. **Engineering polish**: GIL interruptibility, SIMD verification, centroid packing

Rationale: make the exact solver cheap before adding modeling. The dominant n×K assignment loop is the bottleneck, not the model.

---

## 1. Accelerated Spherical K-means

### Decision: Cosine Hamerly, not Elkan, not Yinyang, not ANN

**Why Hamerly over Elkan:**
- Hamerly: O(n) bounds (upper + lower per point) = ~5MB at n=323K
- Elkan: O(n×K) lower bounds = 253MB at n=323K, K=98
- For our memory budget, Hamerly wins decisively
- Research explicitly recommends Hamerly for our regime

**Why not Yinyang:**
- No mature spherical Yinyang with the same formula clarity as Hamerly
- Strong v2.2 candidate, not a v2 blocker

**Why not ANN/tree/PQ:**
- On the unit sphere, all points are at the same radius — annular/Exponion geometry degenerates
- Cover-tree hybrids are not yet drop-in spherical solutions
- Hardware-efficient blocked brute force can beat sophisticated MIPS indices
- Approximate search introduces objective drift, weakening our "exact fixed-K" positioning
- Recommendation: "prune the exact solver and make the dense kernel fast" rather than "bolt on a MIPS index"

### The Core Geometry

Triangle inequality lives in **angular distance space**, not cosine space:

$$d_\angle(x,y) = \arccos(x^\top y)$$

$d_\angle$ is a true metric on the sphere:

$$d_\angle(x,y) \leq d_\angle(x,z) + d_\angle(z,y)$$

This gives exact cosine bounds by monotonicity of cos on [0,π]:

$$x^\top y \geq \cos(\arccos(x^\top z) + \arccos(z^\top y))$$

$$x^\top y \leq \cos(|\arccos(x^\top z) - \arccos(z^\top y)|)$$

The 2021 SISAP paper derived these inequalities because cosine itself is not a metric. The spherical K-means acceleration paper then uses angular bounds to adapt Elkan and Hamerly exactly.

**Implementation mental model: store bounds in angles, compute assignments with dots.**

### Hamerly Bookkeeping (Exact Spherical)

For each point i with assigned centroid a[i]:
- Upper bound: $u_i \geq d_\angle(x_i, c_{a_i})$
- Lower bound: $l_i \leq \min_{j \neq a_i} d_\angle(x_i, c_j)$
- Half-separation: $s_j = \frac{1}{2} \min_{k \neq j} d_\angle(c_j, c_k)$

**Skip point i if:** $u_i \leq \max(l_i, s_{a_i})$

After centroid update with movement $m_j = d_\angle(c_j^{old}, c_j^{new})$:
- $u_i \leftarrow u_i + m_{a_i}$
- $l_i \leftarrow \max(l_i - m_{max}, 0)$

This is **exact** — same reasoning as Euclidean Hamerly, valid because angular distance satisfies the triangle inequality on the sphere. The spherical paper's core contribution is precisely that this Euclidean reasoning survives on the sphere once you use angular distance.

### Pseudocode (Rust-oriented)

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
            continue                        # SKIP — bounds prove no reassignment

        # refresh exact distance to assigned center
        u[i] = angle(x[i], c[a[i]])
        if u[i] <= max(l[i], s[a[i]]):
            continue                        # SKIP — tightened upper still safe

        # full rival scan (only reached when bounds can't prune)
        best = a[i]
        best_ang = u[i]
        second = +inf
        for each centroid j != a[i]:
            if best_ang <= 0.5 * cc[best, j]:
                continue                    # centroid-centroid pruning
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

    recompute centroids (accumulate in f64, renormalize)
until stop
```

### Numerical Robustness

Store bounds and centroid-movement angles in **f64** even if assignments use f32 dot products. Hamerly's memory is so low there's no reason to risk "almost exact" pruning from bound roundoff. This minimizes the chance that a borderline comparison suppresses a needed rival check.

### Expected Impact

**2x to 8x speedup** on the assignment step, data-dependent:
- Low end: overlapping clusters (our CROSS ruling data)
- High end: well-separated clusters
- First iterations prune less; later iterations (centroids stable) is where Hamerly makes its money
- Combined with better stopping → potentially **5-10x total runtime reduction** (633s → ~60-130s)

**Important caveat:** Direct evidence for dense 128d embeddings is weak. The spherical paper benchmarks sparse text (d=47K) and dense 300d word embeddings, not dense 128d transformer embeddings. Realistic expectation, not a guarantee.

### Complexity & Risks

~600-1000 lines of Rust including:
- Centroid-centroid angular distance precompute
- Per-point upper/lower bound storage and update
- Edge cases: empty-cluster repair invalidating bounds
- Tests against baseline solver (shadow mode)
- Per-iteration telemetry (skip rate, full-scan rate)

**Risks:**
- Invalid bounds after empty-cluster repair → must reset affected bounds
- Branch-heavy code losing to baseline when pruning is poor → benchmark-gated dispatch
- Subtle numerical bugs at skip thresholds → shadow mode for validation

**Mitigation:** Shadow mode that runs both Hamerly and Lloyd on the same seed for small datasets, asserts identical assignments and objective values.

---

## 2. Convergence Detection

### Decision: Three-part stop rule, retire angular-shift-only

Current problem: tol=1e-6 implies centroid shift < 0.081 degrees — absurdly strict for 323K points. Research confirms this is "completely consistent with never triggering before max_iter."

Three ecosystems converge on the same answer:
- scikit-learn: center movement
- R `skmeans`: minimum relative objective improvement
- R `movMF`: relative log-likelihood and/or parameter change

### Recommended Stop Rule

**Primary (objective-based):**
$$\frac{J_t - J_{t-1}}{\max(|J_{t-1}|, \varepsilon)} < 10^{-4}$$

**Secondary (assignment stability):**
$$\text{churn}_t = \frac{1}{n} \sum_i \mathbf{1}[a_i^{(t)} \neq a_i^{(t-1)}] < 10^{-3}$$

**Require both for two consecutive iterations** (protects against shallow plateaus where borderline points flip).

**Guardrail (centroid shift):**
$$1 - \mu_k^{(t-1)\top} \mu_k^{(t)} < 10^{-5}$$

Only as a backup, not the main signal.

### Per-Iteration Telemetry

Log per iteration:
- Objective J_t
- Relative objective change
- Assignment churn (fraction of points that changed)
- Max centroid angular shift
- Hamerly skip rate (fraction of points pruned, when accelerated)
- Wall-clock time per iteration

### Complexity

100-150 lines. Low risk, high impact on user experience.

---

## 3. Mini-Batch Spherical K-means

### Decision: Batch-resultant update, not per-sample Winner-Take-All

**Implement after Hamerly, not before.** Hamerly on the exact solver comes first.

### The Correct Spherical Mini-Batch Update

Renormalization does NOT destroy the streaming property — it IS the correct update on the sphere. The literature shows these forms are equivalent after renormalization:

$$\mu^{new} = \frac{\mu + \eta x}{\|\mu + \eta x\|}$$

$$\frac{\mu + \eta(x - \mu)}{\|\mu + \eta(x - \mu)\|}$$

### Batch-Resultant Design

Maintain an **unnormalized running resultant** $r_k$:

$$\mu_k = \frac{r_k}{\|r_k\|}$$

For mini-batch B:
1. Assign points using current $\mu_k$
2. Compute per-cluster batch sums: $s_k = \sum_{x \in B, a(x)=k} x$
3. Update: $r_k \leftarrow (1 - \rho_k) r_k + \rho_k s_k$
4. Renormalize: $\mu_k = r_k / \|r_k\|$

Where $\rho_k = |B_k| / (N_k + |B_k|)$ (streaming average weight).

This is the spherical analogue of incremental K-means expressed in the correct sufficient statistic for directional data: the resultant vector.

**Default batch size: 4096** for d=128. Range 1024-8192 user-configurable.

Note: batch-size guidance is engineering judgment, not strongly published evidence.

### Complexity

300-500 lines, reusing existing assignment kernel. Risk is expectations, not correctness: mini-batch won't be a drop-in replacement for exact solver when users care about reproducible fixed-K purity.

---

## 4. Dimensionality Reduction

### Decision: PCA default, Matryoshka opt-in, skip SparseRandomProjection

**Three API modes:**
```
reduction="pca"          # default, model-agnostic, proven
reduction="matryoshka"   # opt-in, for text-embedding-3-* models
reduction="none"         # user already truncated upstream
```

### Why PCA stays default

- PCA is the optimal linear variance-preserving reduction
- Randomized PCA is specifically the large-scale approximation recommended by modern libraries
- 2019 comparison found RPCA should be preferred over random projection when accuracy matters
- PCA is the **portable clustering preprocessor**

### Why Matryoshka as opt-in

- OpenAI API supports `dimensions` parameter for prefix truncation
- 256d shortened `text-embedding-3-large` outperforms full 1536d `ada-002` on MTEB
- BUT: no direct clustering-vs-PCA comparison published
- 2025 research is mixed: one paper says truncation inevitably loses info (dimension importance ≠ coordinate position); another says clustering tolerates aggressive truncation
- Literature says "truncation can work surprisingly well," NOT "prefix-128 beats PCA-128 for clustering"

Matryoshka is a **provider-specific ingestion optimization**. PCA is the **portable clustering preprocessor**.

### Why NOT SparseRandomProjection

Speed/footprint escape hatch, not a quality improvement for this use case. Not worth v2 engineering time.

### PCA Backend: faer

Research recommends `faer` crate over platform BLAS:
- Pure Rust, preserves "Rust-first" story
- High-performance blocked kernels for dense matrices
- Exactly targets our Y=XΩ matmul and QR steps
- Better than our hand-rolled Gram-Schmidt + custom matmul
- Start with faer before going to platform BLAS

---

## 5. Quality Improvements

### Decision: Weighted multi-view fusion

**Why this, not something else:**
- Multi-view clustering literature is mature and canonical (mvlearn package includes multiview spherical K-means)
- Our two embeddings (A: description, B: description+materials+use) are complementary views
- Fusion keeps the solver unchanged — no new algorithm needed

### The Fusion Rule

Let $z_A$ and $z_B$ be separately normalized, optionally PCA-reduced views:

$$\tilde{z} = [\sqrt{w_A} \cdot z_A, \sqrt{w_B} \cdot z_B]$$
$$z = \tilde{z} / \|\tilde{z}\|$$

Spherical K-means on z maximizes:

$$z_i^\top c_k \propto w_A \cdot z_{A,i}^\top c_{A,k} + w_B \cdot z_{B,i}^\top c_{B,k}$$

Principled objective, easy to document, keeps solver unchanged.

**Recommended weights:** $w_A \approx 0.7, w_B \approx 0.3$ (description dominates, based on our exp2 results showing B adds only +1.3%).

### Expected Impact

**+1 to +5 purity points.** Intentionally conservative planning prior, not a published benchmark.

### What we're NOT doing for quality

**Not local-search refinement (Kernighan-Lin):** Biggest wins on small clusters (few dozen documents). Our chapters have thousands of documents — unlikely to help meaningfully. v3 candidate.

**Not spectral clustering:** Requires affinity matrix/graph construction — not natural for fixed-K at 323K. Different problem.

**Not BERTopic/Top2Vec comparison:** They solve open-ended topic discovery, not fixed-K chapter recovery. Different product.

**Is 58.6% the ceiling?** No clean theoretical upper bound exists for unsupervised recovery of an external administrative taxonomy from semantic embeddings. The HTS system is a legal construct; it doesn't have to correspond to embedding geometry. Budget for modest headroom (+1-5 points), not a miracle (+20).

---

## 6. Scaling vMF

### Decision: Hard-vMF default, chunked soft-vMF premium

**Key insight: soft EM does NOT need n×K responsibility matrix.**

Sufficient statistics are only:
$$N_k = \sum_i r_{ik} \qquad S_k = \sum_i r_{ik} x_i$$

Compute responsibilities in chunks (B=8192), accumulate (N_k, S_k), discard chunk. Memory: O(B×K + K×d), not O(n×K).

At B=8192, K=200: chunk buffer ≈ 13MB vs full matrix ≈ 800MB.

### Two Modes

**`hard_vmf` (default refinement):**
- Insert hardening step: $r_{ik} \leftarrow \mathbf{1}[k = \arg\max_j r_{ij}]$
- Dramatically faster: 29s vs 3,368s in Banerjee's experiments at K=20
- Best for well-separated clusters

**`soft_vmf_chunked` (premium):**
- Current soft EM rewritten to stream chunk statistics
- Better for overlapping/imbalanced clusters
- 13MB chunks instead of 800MB full matrix

### NOT doing for v2

**Not variational or hierarchical vMF:** Solves a richer problem than "cheap fixed-K refinement." Chunked EM gives almost all the scalability benefit for much less code.

---

## 7. Engineering

### GIL Interruptibility

Keep GIL released for almost all of `fit()`, but reacquire briefly once per outer iteration to call `Python::check_signals()`. This surfaces `KeyboardInterrupt` so users can Ctrl+C during long runs. PyO3 docs are explicit: Ctrl+C only sets a flag; Rust code must check it.

### SIMD Verification

"Verify, then specialize."
1. Use `cargo-show-asm` to confirm d=128 dot kernel emits AVX2/FMA
2. Only add explicit SIMD if the emitted kernel is clearly suboptimal
3. `target-cpu=native` and `target-feature=+avx2` are the standard paths

### Memory Layout

**Keep row-major.** Don't switch to column-major — per-point access would become strided. Instead, make the assignment step more GEMM-like: pack centroids into aligned tiles each iteration, compute point-block × centroid-block products in cache-sized chunks.

### Centroid Packing

The assignment phase is logically X @ C^T followed by argmax. Pack centroids into aligned tiles each iteration. Compute in cache-sized chunks. This is the optimization `faer` would give us internally.

---

## 8. Competitive Positioning

### The Winning Frame

**"Be the default choice for large fixed-K embedding clustering, not for open-ended topic discovery."**

The ecosystem is polarized:
- **Topic-model stacks** (BERTopic, Top2Vec): discover variable number of topics with human-readable representations, online/semi-supervised/manual modes
- **General k-means** (sklearn, FAISS): treat cosine, PCA, and directional refinement as separate concerns

Neither is "fast, exact fixed-K embedding clustering with directional diagnostics."

### Our Differentiators

- Exact cosine clustering at 100K-500K scale
- Predictable fixed-K behavior
- Built-in quality diagnostics (resultant lengths, concentration κ, representatives)
- Optional probabilistic refinement without leaving the pipeline
- Single `model.fit()` call

### The Positioning Sentence

> Be the default choice for large fixed-K embedding clustering, not for open-ended topic discovery. Speed first, but not speed alone.

With that framing, the Hamerly implementation and chunked vMF refactor are not just "engineering cleanups" — they are the center of the product.

### Closest Prior Art

Fragmented across separate packages:
- R `skmeans` for spherical K-means
- R `movMF` for vMF mixtures
- No integrated Python-facing stack

---

## Summary: v2 Roadmap

| Phase | Feature | Impact | Effort | Risk |
|-------|---------|--------|--------|------|
| 2.1 | Cosine Hamerly | 2-8x speedup | 600-1000 lines | Medium (numerical edge cases) |
| 2.2 | Stop rules + telemetry | Proper convergence, user visibility | 100-150 lines | Low |
| 2.3 | Chunked hard/soft vMF | 10-60x memory reduction | 200-300 lines | Low |
| 2.4 | Multi-view fusion | +1-5 purity points | 100-200 lines | Low |
| 2.5 | Mini-batch spherical | Large-n scalability | 300-500 lines | Medium (expectations) |
| 2.6 | Matryoshka fast path | Skip PCA for supported models | 50-100 lines | Low |
| 2.7 | faer PCA backend | Faster reduction | 200-300 lines | Low |
| 2.8 | GIL + SIMD + packing | Polish | 100-200 lines | Low |
| **Total** | | **5-10x faster, 1-5% better** | **~1650-2750 lines** | |
