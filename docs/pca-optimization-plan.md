# PCA Optimization Plan: Three Paths to 60 Seconds

## The Problem

Experiment 6 proved that randomized PCA is 90% of the EmbeddingCluster pipeline runtime:

| Stage | Time | Share |
|-------|------|-------|
| PCA matmul (323K × 1536 × 138) | 570s | 90% |
| Everything else | 67s | 10% |

The 60-second target requires reducing PCA from 570s to near-zero or under 60s.

## Three Paths

### Path 1: Matryoshka — Skip PCA Entirely (Available Today)

**Impact:** 570s → 0s. Total pipeline: ~66s.

OpenAI's text-embedding-3 family supports the `dimensions` parameter for Matryoshka-style prefix truncation. Users can request 128d embeddings directly:

```python
response = client.embeddings.create(
    input=texts,
    model="text-embedding-3-small",
    dimensions=128  # ← returns 128d instead of 1536d
)
```

Then cluster without PCA:

```python
model = EmbeddingCluster(n_clusters=50, reduction_dim=None)
model.fit(embeddings_128d)  # ~5s
```

**Pros:**
- Zero engineering effort (code already supports this)
- Eliminates PCA entirely
- Reduces embedding storage from 6KB to 512 bytes per document
- Lower API cost (smaller embeddings = fewer tokens in future billing)

**Cons:**
- Only works for new embeddings (can't retroactively truncate existing 1536d)
- Model-specific (only text-embedding-3, not Cohere/Voyage/etc.)
- Literature is mixed on whether prefix-128 matches PCA-128 for clustering quality

**Action:** Document prominently in embedding clustering guide. Add to README. This is the simplest path for new projects.

### Path 2: faer PCA Backend (Engineering Work)

**Impact:** 570s → 120-190s. Total pipeline: ~130-200s.

Replace our hand-rolled rayon matmul with faer's blocked dense matmul. The bottleneck is Y = X·Omega (323K × 1536 × 138).

**Current implementation (reduction.rs):**
```rust
// Rayon row-block parallelism — each thread processes a chunk of rows
y.par_chunks_mut(k)
    .enumerate()
    .for_each(|(i, y_row)| {
        for col in 0..k {
            let mut acc = 0.0f64;
            for j in 0..d {
                acc += (data[i * d + j] - mean[j]) * omega[j * k + col];
            }
            y_row[col] = acc;
        }
    });
```

This is a naive row-by-column dot product — no cache blocking, no SIMD tiling, no kernel packing. faer's matmul uses all of these.

**faer implementation:**
```rust
use faer::Mat;

// Build faer matrices (column-major, needs transpose awareness)
let x_mat = Mat::from_fn(n, d, |i, j| data[i * d + j] - mean[j]);
let omega_mat = Mat::from_fn(d, k, |i, j| omega[i * k + j]);

// Blocked matmul — uses cache-aware tiling + SIMD
let y_mat = &x_mat * &omega_mat;  // faer handles the optimization
```

**Complexity:** ~200-300 lines to integrate faer for the matmul + QR + eigendecomp.

**Dependency:** `faer = "0.24"` in Cargo.toml. Pure Rust, no system BLAS.

**Expected speedup:** 3-5x based on faer's blocked matmul vs our naive row-parallel. The improvement comes from:
1. Cache-line-aware tiling (L1/L2 blocking)
2. SIMD kernel packing (faer uses AVX2/NEON internally)
3. Better parallelism strategy (faer uses rayon internally with work-stealing)

**Risk:** faer uses column-major layout; our data is row-major. Need to handle the transpose correctly. The QR and eigendecomp replacements are straightforward (small matrices).

### Path 3: BLAS/GPU PCA (Future)

**Impact:** 570s → 50-80s. Total pipeline: ~65s.

Link against OpenBLAS or MKL for the matmul. These are decades-optimized assembly kernels.

**Options:**
- `openblas-src` crate — system OpenBLAS or bundled
- `intel-mkl-src` crate — Intel MKL
- faer with BLAS backend — faer can delegate to BLAS for large matmuls

**Pros:**
- Potentially 10x speedup over our current implementation
- Well-proven performance

**Cons:**
- System dependency (OpenBLAS must be installed, or bundled at 30MB+)
- Platform-specific behavior (MKL is Intel-only performance)
- Wheel size increases significantly
- Build complexity (linking BLAS from maturin)
- Non-deterministic results across platforms (BLAS implementations differ)

**GPU path:**
- Metal Performance Shaders on Apple Silicon (unified memory, zero-copy)
- cuBLAS on NVIDIA
- This is a single large GEMM — exactly what GPU excels at
- But adds massive distribution complexity

**Recommendation:** Defer BLAS/GPU to v3. faer (Path 2) gets us 3-5x with zero system dependencies. BLAS/GPU is the final optimization for users who need the absolute fastest path.

## Recommendation: Build Path 1 + 2 + EmbeddingReducer

### Phase A: EmbeddingReducer (standalone PCA transformer)

**What:** Extract PCA into a standalone class with fit/transform/save/load.

**Why:** Even without making PCA faster, this eliminates redundant PCA runs. Users run PCA once, save the result, cluster unlimited times for free.

**Effort:** ~400-600 lines (Rust reducer + PyO3 + Python wrapper + save/load + tests)

### Phase B: Expose reduced_data_ from EmbeddingCluster

**What:** Cache and expose the intermediate PCA output.

**Why:** Users of the convenience pipeline can access the reduced data without restructuring their code.

**Effort:** ~50 lines (add field + property + cache in fit())

### Phase C: faer PCA backend

**What:** Replace hand-rolled matmul with faer.

**Why:** 3-5x speedup on the dominant operation.

**Effort:** ~200-300 lines

### Phase D: Matryoshka documentation

**What:** Prominently document the "request 128d from OpenAI" path.

**Why:** Zero-code path to 60-second target for new projects.

**Effort:** Documentation only

## Build Order

```
Phase D (docs)         — do immediately, zero code
Phase B (reduced_data_) — small change, big UX win
Phase A (EmbeddingReducer) — the architecture fix
Phase C (faer)          — the performance fix
```

## Expected Outcomes

| Configuration | PCA | K-means | Total |
|---------------|-----|---------|-------|
| Current | 570s | 4s | 637s |
| + EmbeddingReducer (cached) | 570s once, 0s after | 4s | 5s after first run |
| + faer backend | 120-190s | 4s | 130-200s |
| + Matryoshka (new embeddings) | 0s | 4s | 66s |
| + faer + cached | 120-190s once, 0s after | 4s | 5s after first run |
