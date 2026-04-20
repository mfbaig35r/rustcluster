# Deep Research Prompt: GPU Acceleration for rustcluster

> ## Context
>
> I maintain `rustcluster`, a Rust-backed Python clustering library published on PyPI. It implements K-means (Lloyd + Hamerly), MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering, and an experimental EmbeddingCluster pipeline (spherical K-means + randomized PCA + vMF mixtures). Everything runs on CPU today — compiled Rust with rayon parallelism, LLVM auto-vectorized kernels, PyO3 Python bindings with zero-copy NumPy interop.
>
> The library currently ships as a pure CPU wheel for Linux (x86_64/aarch64), macOS (x86_64/aarch64), and Windows (x86_64). No external BLAS dependency. No GPU requirement. Users install with `pip install rustcluster` and it just works.
>
> I want to understand the realistic options, tradeoffs, and engineering effort for adding optional GPU acceleration — without breaking the "pip install and it works" story for CPU-only users.
>
> ## Current Performance Profile
>
> The CPU-bound bottlenecks, measured on real workloads:
>
> **EmbeddingCluster (spherical K-means):**
> - 323K embeddings × 128 dimensions (after PCA) × K=98 × 50 iterations × 3 inits = 633 seconds
> - The inner loop is n×K dot products per iteration — a matrix multiply X(n×d) @ C(K×d)^T
> - Assignment step is parallelized via rayon across points
> - Centroid update is sequential (accumulate sums + renormalize) — fast, not the bottleneck
>
> **Regular K-means (Lloyd):**
> - Beats sklearn 1.3-6.4x on low-dimensional data (d < 20)
> - At d=32, K=32, n=100K: ~70ms per fit
> - Bottleneck is the same: n×K distance computations per iteration
>
> **DBSCAN/HDBSCAN:**
> - KD-tree accelerated neighbor queries for d ≤ 16
> - At d > 16: falls back to O(n²) brute-force pairwise distances
> - HDBSCAN MST construction is O(n²) — inherently sequential Prim's algorithm
>
> **Randomized PCA:**
> - One large matmul: X(n×d) @ Omega(d×k) where n=323K, d=1536, k=138
> - Currently pure Rust with rayon row-block parallelism
> - QR and eigendecomposition on small matrices (k×k) — fast, not worth GPU-ing
>
> ## What I Need Researched
>
> ### 1. GPU Backend Options for Rust
>
> I need a detailed comparison of every viable approach for adding GPU compute to a Rust library that ships as Python wheels via maturin:
>
> **CUDA (NVIDIA only):**
> - `cudarc` crate — what's the API like? How mature is it? Can it call cuBLAS for GEMM?
> - `cuda-sys` / raw CUDA bindings — too low-level?
> - Writing custom CUDA kernels in `.cu` files and linking from Rust — how does the build system work? Does maturin handle this?
> - Can CUDA wheels be built via GitHub Actions? What's the CI/CD story?
> - Does the user need CUDA toolkit installed, or can we ship the CUDA runtime in the wheel?
>
> **wgpu (cross-platform: NVIDIA/AMD/Intel/Apple):**
> - `wgpu` crate — WebGPU API implemented in Rust via Vulkan/Metal/DX12
> - Can it do GEMM-level performance? Or is the overhead too high for our matrix sizes?
> - Compute shaders in WGSL — how do you write a matrix multiply shader? What's the performance vs cuBLAS?
> - Does wgpu work headless (no window/display required)? Our library is a CLI/server tool, no GUI
> - What's the startup latency? (device init, shader compilation, buffer allocation)
> - Can we ship wgpu in a maturin wheel without the user installing Vulkan/Metal SDKs?
>
> **Metal (Apple Silicon):**
> - Apple Silicon has unified memory — GPU and CPU share the same RAM, no PCIe transfer
> - `metal` crate via `objc` bindings — how mature? Any examples of GEMM?
> - Metal Performance Shaders (MPS) — Apple's optimized BLAS-like library. Can Rust call MPS?
> - Is there a Rust crate for MPS? Or do you need raw Objective-C FFI?
> - What's the realistic speedup for a 323K × 128 × 98 GEMM on M1/M2/M3/M4 GPU?
> - Would unified memory mean true zero-copy between our Rust arrays and GPU compute?
>
> **OpenCL:**
> - `ocl` crate — still maintained? Cross-platform (NVIDIA/AMD/Intel/Apple)?
> - Apple deprecated OpenCL in favor of Metal. Is it still usable on macOS?
> - How does OpenCL GEMM performance compare to cuBLAS and Metal?
> - Is OpenCL dead for new projects?
>
> **Vulkan compute (direct):**
> - `ash` or `vulkano` crates — too low-level for our use case?
> - Performance vs wgpu abstraction?
>
> For each option, I need:
> - Maturity level (production-ready? experimental?)
> - Platform support matrix
> - Performance ceiling (% of theoretical peak for GEMM at our sizes)
> - Build/distribution complexity (can it ship in a pip wheel?)
> - User requirements (do they need to install anything?)
>
> ### 2. Which Operations to GPU-Accelerate
>
> Not everything benefits from GPU. For each operation, analyze whether GPU is worth it:
>
> **Assignment step (n×K dot products):**
> - This is a GEMM: S = X @ C^T, then argmax per row
> - At n=323K, K=98, d=128: the matrices are (323K × 128) × (128 × 98) = (323K × 98) result
> - How does this compare to typical GEMM benchmarks? Is it large enough for GPU to win?
> - What's the transfer overhead? X stays on GPU across iterations, only C (98 × 128 = 50KB) needs upload per iteration
> - What about the argmax? Fused GEMM + argmax kernel vs separate operations?
>
> **Pairwise distance matrix (DBSCAN/HDBSCAN):**
> - O(n²) pairwise distances — at n=50K, d=128: 50K × 50K = 2.5 billion distances
> - This is a massive GEMM (for Euclidean: ||x-y||² = ||x||² + ||y||² - 2x·y)
> - GPU would dominate here. But the result matrix is n² × 8 bytes = 20GB at n=50K. Does it fit?
> - Would chunked computation work? Process in tiles?
>
> **PCA matmul (X @ Omega):**
> - One-shot: (323K × 1536) × (1536 × 138) = (323K × 138)
> - Large enough for GPU. But it runs once, not per-iteration
> - Worth the GPU init overhead for a single matmul?
>
> **L2 normalization:**
> - Row-wise norm + divide — trivially parallel but low arithmetic intensity
> - Probably not worth GPU transfer overhead unless data is already on GPU
>
> **Centroid update (accumulate + renormalize):**
> - K sums over assigned points, then normalize
> - Involves scatter/gather by label — harder to parallelize on GPU
> - Probably keep on CPU
>
> **vMF EM (responsibilities):**
> - n×K matrix of posteriors — another GEMM-like operation (log-space dot products)
> - At n=500K, K=200: 100M entries. GPU would help
> - The log-sum-exp normalization is per-row — GPU-friendly
>
> ### 3. Hybrid CPU/GPU Architecture
>
> I don't want "everything on GPU." I want "GPU for the hot GEMM, CPU for everything else."
>
> - What's the optimal split? GPU does assignment (GEMM + argmax), CPU does centroid update?
> - How do you efficiently transfer labels (n integers) back from GPU each iteration?
> - Should data live on GPU for the entire fit() call? Or upload/download each iteration?
> - How does Apple Silicon unified memory change this? (no transfer cost — GPU and CPU see the same bytes)
> - What about PCIe-connected GPUs (NVIDIA discrete)? Transfer cost per iteration?
>
> ### 4. Optional GPU as a Feature Flag
>
> The library must work without GPU. GPU should be an optional enhancement.
>
> - How do other Rust/Python libraries handle this? (e.g., PyTorch's CPU/CUDA split)
> - Cargo feature flags: `gpu`, `cuda`, `metal`? How does this interact with maturin wheel building?
> - Runtime detection: can we detect GPU availability at runtime and dispatch automatically?
> - Separate wheels: `rustcluster` (CPU) vs `rustcluster-gpu` (with CUDA)? Or one wheel with optional GPU?
> - How does FAISS handle this? How does cuML handle this?
>
> ### 5. Build and Distribution
>
> This is the hardest part. GPU wheels are notoriously difficult to build and distribute.
>
> - **CUDA wheels on PyPI:** How do PyTorch, CuPy, and FAISS distribute CUDA wheels? Do they ship the CUDA runtime? Multiple wheels per CUDA version?
> - **GitHub Actions CI:** Can you build CUDA wheels in GitHub Actions? What runner images have CUDA? Cost?
> - **Wheel size:** CUDA runtime is ~500MB. Does bundling it make wheels impractically large?
> - **maturin + CUDA:** Has anyone built a maturin-based PyO3 project with CUDA support? Any examples?
> - **Metal on macOS:** Does maturin handle Metal framework linking automatically? Or manual?
> - **Cross-platform strategy:** Ship CPU-only wheel by default, separate `rustcluster-cuda` and `rustcluster-metal` packages? Or detect at install time?
>
> ### 6. Competitive Landscape
>
> - **FAISS GPU K-means:** Performance numbers at n=100K-1M, d=128, K=100? Does it support spherical mode?
> - **cuML K-means:** NVIDIA's GPU K-means. Performance? Does it support cosine/spherical?
> - **PyTorch clustering:** Can you do K-means efficiently in PyTorch by treating it as a GEMM problem? People do this in practice
> - **RAPIDS cuML HDBSCAN:** GPU HDBSCAN performance numbers?
> - **Apple MLX:** Apple's ML framework for Apple Silicon. Does it expose K-means or GEMM primitives usable from Rust/Python?
>
> For each competitor, I need: what GPU operations they accelerate, what performance they achieve, and what their distribution story is (pip install complexity).
>
> ### 7. Realistic Speedup Estimates
>
> For our specific workloads, what speedups are realistic? Not theoretical peak, but actual achievable performance including transfer overhead, kernel launch latency, etc.
>
> | Operation | Size | CPU (current) | GPU (estimated) | Speedup |
> |-----------|------|---------------|-----------------|---------|
> | Spherical K-means assign | 323K × 128 × 98 | ? ms/iter | ? ms/iter | ? |
> | Pairwise distances | 50K × 50K × 128 | ? s | ? s | ? |
> | PCA matmul | 323K × 1536 × 138 | ~2s | ? | ? |
> | vMF responsibilities | 323K × 98 | ? s | ? s | ? |
>
> Fill in realistic numbers for:
> - NVIDIA RTX 4090 (consumer high-end)
> - NVIDIA A100 (cloud/datacenter)
> - Apple M3 Max GPU (unified memory)
> - Apple M1 GPU (baseline Apple Silicon)
>
> ### 8. Recommendation
>
> Given all the above, what is the **concrete recommended path** for adding GPU support to rustcluster?
>
> - Which backend? (CUDA? wgpu? Metal? Multiple?)
> - Which operations? (assignment only? PCA too? pairwise distances?)
> - What's the minimum viable GPU feature? (the smallest thing that gives 5x+ speedup)
> - What's the engineering effort? (rough days/weeks of work)
> - What's the distribution strategy? (feature flags? separate packages?)
> - Should we wait for wgpu to mature, or go CUDA-first?
> - Is the honest answer "don't bother, Elkan/Hamerly on CPU is enough"?
>
> ## Deliverables
>
> For each section, I need:
> 1. Concrete findings with specific numbers, not vague assessments
> 2. Code examples or pseudocode where relevant
> 3. Links to real projects that have solved similar problems
> 4. Honest assessment of maturity and risk
> 5. A final ranked recommendation with tradeoffs explicitly stated
