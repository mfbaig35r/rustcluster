# EmbeddingCluster v2 Research Findings

Research completed 2026-04-20. Key decisions for v2 development.

## Implementation Order (Confirmed)

1. **Cosine Hamerly accelerator** for spherical K-means
2. **Better stopping rules** + per-iteration instrumentation
3. **Chunked vMF refinement** with hard-vMF option
4. **Weighted multi-view fusion** of embedding A and B
5. **Matryoshka fast path** + better PCA backend (faer)

## 1. Accelerated Spherical K-means

### Decision: Hamerly, not Elkan

- Hamerly uses O(n) bounds (upper + lower per point) = ~5MB at n=323K
- Elkan uses O(n×K) lower bounds = 253MB at n=323K, K=98
- For our regime, memory asymmetry favors Hamerly decisively

### The Core Geometry

Triangle inequality lives in **angular distance space**, not cosine space:

```
d_angle(x,y) = arccos(x·y)       // true metric on sphere
d_angle(x,y) ≤ d_angle(x,z) + d_angle(z,y)   // exact triangle inequality
```

Implementation mental model: **store bounds in angles, compute assignments with dots.**

### Hamerly Bookkeeping (Spherical)

For each point i with assigned centroid a[i]:
- Upper bound: `u[i] ≥ d_angle(x[i], c[a[i]])`
- Lower bound: `l[i] ≤ min_{j≠a[i]} d_angle(x[i], c[j])`
- Half-separation: `s[j] = 0.5 × min_{k≠j} d_angle(c[j], c[k])`

Skip point i if: `u[i] ≤ max(l[i], s[a[i]])`

After centroid update with movement m[j] = d_angle(old, new):
- `u[i] += m[a[i]]`
- `l[i] = max(l[i] - max_movement, 0)`

This is exact — same reasoning as Euclidean Hamerly, valid because angular distance satisfies triangle inequality on the sphere.

### Expected Impact

**2x to 8x speedup** on assignment step, depending on cluster separation.
- Low end: overlapping clusters (our CROSS data)
- High end: well-separated clusters
- Later iterations prune more (centroids stabilize)
- Combined with better stopping → potentially **5-10x total runtime reduction**

### Complexity

~600-1000 lines of Rust. Main risks:
- Invalid bounds after empty-cluster repair
- Branch-heavy code losing to baseline when pruning is poor
- Subtle numerical bugs at skip thresholds

Mitigation: shadow mode that runs both Hamerly and Lloyd, asserts identical results.

## 2. Convergence Detection

### Decision: Three-part stop rule

Replace current `1 - dot(old, new) < 1e-6` (never triggers) with:

```
Relative objective change:  (J_t - J_{t-1}) / max(|J_{t-1}|, ε) < 1e-4
Assignment churn:           (points that changed) / n < 1e-3
For two consecutive iterations (protects against plateau oscillation)
```

Current tol=1e-6 requires centroid shift < 0.081 degrees — far too strict for 323K points.

Keep centroid angular shift as a **guardrail** at 1e-5, not the primary stop signal.

### Complexity

100-150 lines. Low risk.

## 3. Mini-Batch Spherical K-means

### Decision: Batch-resultant update

The correct update maintains an **unnormalized running resultant** r_k:

```
μ_k = r_k / ||r_k||           // published centroid

For mini-batch B:
  s_k = Σ_{x∈B, a(x)=k} x    // per-cluster batch sum
  r_k = (1 - ρ_k) × r_k + ρ_k × s_k
  μ_k = r_k / ||r_k||         // renormalize

where ρ_k = |B_k| / (N_k + |B_k|)  // streaming average weight
```

Renormalization does NOT destroy the streaming property — it IS the correct update on the sphere. The paper shows the forms `(μ + η×x) / ||μ + η×x||` and `(μ + η×(x-μ)) / ||...||` are equivalent after renormalization.

**Default batch size: 4096** for d=128. Range 1024-8192 user-configurable.

### Complexity

300-500 lines, reusing existing assignment kernel.

## 4. Dimensionality Reduction

### Decision: Keep PCA default, add Matryoshka as opt-in

```python
reduction="pca"         # default, model-agnostic, proven
reduction="matryoshka"   # opt-in for text-embedding-3-* models
reduction="none"         # user already truncated
```

Matryoshka removes PCA fit time and projection cost when the model supports it. But no published evidence that prefix-128 beats PCA-128 for clustering specifically.

PCA is the **portable clustering preprocessor**. Matryoshka is a **provider-specific ingestion optimization**.

Do NOT invest in SparseRandomProjection — it's a speed hack, not a quality improvement for this use case.

### PCA Backend: faer

Consider switching from hand-rolled Gram-Schmidt + deflated power iteration to `faer` crate for the QR and matmul. Preserves the Rust-first story while getting blocked kernels and better dense-matrix infrastructure.

## 5. Quality Improvements

### Decision: Weighted multi-view fusion

Concatenate views with weights, renormalize:

```
z̃ = [√w_A × z_A,  √w_B × z_B]
z = z̃ / ||z̃||
```

Spherical K-means on z maximizes: `w_A × cos_sim_A + w_B × cos_sim_B`

Recommended starting weights: w_A=0.7, w_B=0.3 (description dominates, materials/use adds signal).

Expected improvement: **+1 to +5 purity points** over single-view. Conservative estimate.

### NOT doing for v2:
- Local-search refinement (Kernighan-Lin) — biggest wins are on small clusters, ours are large
- Spectral clustering — requires affinity matrix, not natural for fixed-K at 323K
- BERTopic/Top2Vec comparison — different problem (topic discovery vs fixed-K clustering)

## 6. vMF Scaling

### Decision: Hard-vMF default, chunked soft-vMF premium

**Key insight:** Soft EM does NOT need the n×K responsibility matrix in memory.

Sufficient statistics are only:
```
N_k = Σ_i r_ik          // effective count
S_k = Σ_i r_ik × x_i   // weighted sum
```

Compute responsibilities in chunks (B=8192), accumulate (N_k, S_k), discard chunk. Memory: O(B×K + K×d), not O(n×K).

Two modes:
- `hard_vmf`: harden responsibilities to 0/1 after E-step. Dramatically faster (29s vs 3368s in Banerjee's experiments at K=20). Best for well-separated clusters.
- `soft_vmf_chunked`: stream chunk statistics instead of storing all responsibilities. Premium mode for overlap-heavy data.

## 7. Competitive Positioning

### The winning frame

**"Be the default choice for large fixed-K embedding clustering, not for open-ended topic discovery."**

BERTopic and Top2Vec solve "discover topics I don't know about." We solve "cluster into K groups I've defined, fast, with quality diagnostics."

Differentiators:
- Exact cosine clustering at 100K-500K scale
- Predictable fixed-K behavior
- Built-in quality diagnostics (resultant lengths, concentration κ, representatives)
- Optional probabilistic refinement without leaving the pipeline
- Single `model.fit()` call, no UMAP/HDBSCAN/c-TF-IDF stack

### Not competing with:
- BERTopic (open-ended topic discovery)
- FAISS (ANN index + quantization)
- cuML (GPU-first ecosystem)

## Summary: v2 Roadmap

| Phase | Feature | Impact | Effort |
|-------|---------|--------|--------|
| 2.1 | Cosine Hamerly | 2-8x speedup | 600-1000 lines |
| 2.2 | Stop rules + telemetry | Proper convergence | 100-150 lines |
| 2.3 | Chunked hard/soft vMF | 10x memory reduction | 200-300 lines |
| 2.4 | Multi-view fusion | +1-5 purity points | 100-200 lines |
| 2.5 | Matryoshka fast path | Skip PCA for supported models | 50-100 lines |
| 2.6 | faer PCA backend | Faster reduction | 200-300 lines |
| **Total** | | **5-10x faster, 1-5% better** | **~1500-2500 lines** |
