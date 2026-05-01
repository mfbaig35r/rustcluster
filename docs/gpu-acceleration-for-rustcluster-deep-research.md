# GPU Acceleration for rustcluster

## Bottom line

For `rustcluster` as you described it, GPU acceleration is worth doing only if you treat it as a **surgical acceleration of the GEMM-shaped hot path**, not as a full “move clustering to GPU” rewrite. The best current Rust option for that job is **CUDA first** on Linux x86_64, using `cudarc` plus cuBLAS/cuBLASLt or a thin FFI layer to nvcc-built kernels. `cudarc` is active, has recent 2026 releases, exposes safe/result/sys layers, and includes `cublas` and `cublaslt` modules. By contrast, `wgpu` is real and headless, but its portable-shader GEMM ceiling is materially lower, and the project itself is still working on startup pain points such as precompiled shaders. On macOS, the best second backend is **Metal plus MPS**, using the newer `objc2-metal` family rather than the older `metal` crate, which is now explicitly deprecated for new development. citeturn43view0turn43view2turn24search7turn23view0turn23view1turn16search3turn16search0turn16search2

The operational recommendation is therefore straightforward: keep the existing CPU wheels exactly as they are, add an **optional accelerator-specific path** for the few operations that dominate runtime, and leave the rest on CPU for now. The highest-value GPU targets are the spherical K-means assignment GEMM, randomized PCA’s one large matmul, and the shared GEMM-like core you can reuse for vMF responsibilities. DBSCAN and HDBSCAN are a bad MVP target because accelerating only pairwise distances does not solve the rest of the pipeline, especially neighborhood graph handling and MST construction. citeturn11view5turn13view1turn11view4

Packaging is the real constraint, not kernels. Cargo features are compile-time switches; `maturin` builds whatever Cargo artifact you asked for, and Linux wheels must satisfy manylinux-style portability rules. The Python packaging ecosystem is still actively trying to improve accelerator-specific wheel distribution, which is exactly why the entity["organization","PyTorch Foundation","linux foundation project"] blog now talks about “wheel variants” as the future of specialized hardware packaging. That is a strong signal that you should **not** try to cram CPU, CUDA, and every future accelerator story into one default wheel. citeturn35search0turn35search1turn17search4turn35search9turn36view0

## Backend options in Rust

### CUDA with `cudarc`

This is the only option I would call **production-capable for your MVP**. `cudarc` describes itself as a minimal and safe API over the CUDA toolkit, has frequent releases, and explicitly exposes wrappers for the CUDA driver/runtime as well as `cublas`, `cublaslt`, `cudnn`, `curand`, `nccl`, and `nvrtc`. Its API design is exactly what you want in a Rust library: use the safe layer for 90% of the work, and drop to result/sys only where you need to. That makes it a good fit for “call cuBLAS for GEMM, keep Rust in charge of orchestration.” citeturn43view0turn43view2

For kernels beyond GEMM, you have two workable CUDA stories. The higher-level path is `cudarc` plus runtime-loaded PTX or NVRTC when you need a small reduction kernel. The lower-level path is to compile `.cu` files from `build.rs` and link them into the Rust crate. The `cc` crate documents CUDA support directly: `.cuda(true)` invokes nvcc, and Cargo build scripts are designed to run exactly this sort of native build step. `maturin` does not “special case” CUDA here; it simply runs Cargo, so if Cargo can build the artifact, `maturin` can package it. The hard part is not build invocation but portable wheel distribution and runtime library handling. citeturn18search0turn18search1turn18search11turn17search4turn35search9

For your use case, CUDA’s strongest advantage is that you can **outsource the hardest optimization problem — high-performance GEMM — to cuBLAS/cuBLASLt** and keep your own custom kernels small and limited to rowwise argmax, normalization, or other thin post-processing. That minimizes engineering risk while landing the biggest speedup.

### Raw CUDA bindings

Raw `cuda-*-sys` crates are viable only if you deliberately want to build your own CUDA abstraction layer. The current rebooted `cuda-sys` repository says plainly that it is a binding to the CUDA driver and runtime APIs and that CUDA itself is not included; the developer must install it. The latest `cuda-driver-sys` docs also report essentially no Rust-level documentation coverage. That is not a reason to avoid raw FFI in narrow internal modules, but it is too low-level to make it your primary integration story for a library that already has a polished CPU implementation and Python packaging surface. citeturn30search5turn30search6

### `wgpu`

`wgpu` is real, portable, and headless. The crate documents native support across Vulkan, Metal, D3D12, and OpenGL, and the headless path is simple because you can request an adapter with `compatible_surface: None`; the windowless tutorial states explicitly that you do not need a window to create an instance, adapter, or device. If your question were only “can I ship headless GPU compute in a wheel?” the answer would be yes. citeturn24search7turn23view0turn23view2

The problem is the **performance ceiling for GEMM**, which is your dominant workload. Portable WebGPU/WGSL shaders are still much harder to tune to vendor-BLAS levels. A recent public SGEMM experiment on an RTX 5060 Ti reported about **1.48 TFLOP/s**, roughly **8% of theoretical FP32 peak**, and explicitly framed that as a best result after several rounds of optimization. The broader WebGPU inference literature also shows that generalized “one-for-all” kernels can be several times slower than device-customized ones, and that web/native shader compilation and dispatch overhead matter enough that the ecosystem is still pushing on JIT tuning and precompiled shaders. `wgpu` itself has an open tracking issue for precompiled shaders specifically to reduce startup time. For a clustering library whose big win comes from GEMM, that is not where I would spend the first engineering months. citeturn20view0turn21view0turn23view1

My honest assessment is: **use `wgpu` only if cross-vendor portability is more important than top-end GEMM speed**. For `rustcluster`, it is not.

### Metal with MPS

On Apple Silicon, this is the best non-CUDA path. The old `metal` crate is still available, but docs.rs now marks it deprecated for new development and recommends `objc2-metal` instead. Rust bindings also exist for Metal Performance Shaders through `objc2-metal-performance-shaders`, including `MPSMatrixMultiplication`. Apple’s own MPS docs describe that type as the standard matrix multiply primitive, and Apple’s Metal docs say shared storage mode is the default for buffers on Apple silicon GPUs. citeturn16search3turn16search0turn16search2turn15search18turn15search0turn15search3

That shared-memory model is important, but it does **not** mean that your existing NumPy-backed host buffers automatically become zero-copy GPU inputs. It means that once data is in a Metal buffer or in an array framework built for Apple’s memory model, CPU and GPU access the same pool of physical memory without PCIe staging. Apple’s MLX documentation states this plainly: Apple silicon has a unified memory architecture, CPU and GPU access the same pool, and operations can run on CPU or GPU without moving arrays between separate memory locations. In practice, for `rustcluster`, that argues for a later Apple-specific backend that owns GPU-native buffers rather than trying to treat arbitrary host slices as if they were already GPU resources. citeturn39view2turn34search2turn34search6

The realistic verdict is **good second backend, not first backend**. It is the right answer for macOS arm64 users once the CUDA path proves the design, but the Rust-side ergonomics are still less settled than CUDA’s vendor library path.

### OpenCL

I would not start a new clustering backend here. The `ocl` crate is functional and its repository still exists, but the current crate line is old, the latest 0.19.x releases are no longer fresh, and the project asks users to make sure an OpenCL library is installed on the system. Apple still hosts OpenCL documentation, but it lives in archive-oriented documentation and the current platform direction is clearly Metal, not OpenCL. Cross-platform in theory is not enough; you want strong current vendor investment plus a modern Rust ecosystem, and OpenCL does not give you that combination. citeturn25view0turn26search4turn27search0turn27search11

### Direct Vulkan compute

Direct Vulkan compute is technically powerful, but it is too low-level for your MVP. `vulkano` is a safe wrapper around Vulkan, `ash` is the raw binding, and `vulkano-shaders` will compile GLSL to SPIR-V for you. But the moment you need BLAS-class GEMM, you run into the same issue as `wgpu`: portable compute shaders are not cuBLAS. NVIDIA’s own Vulkan ML material makes the point that high-performance matrix multiply really wants dedicated primitives such as cooperative matrices rather than hand-rolled SIMT decomposition. That is precisely why direct Vulkan compute is interesting for ML engines and compiler projects, but not the fastest route to shipping a small optional accelerator inside a Python clustering package. citeturn28search0turn28search2turn28search7turn28search17

## Which operations are worth offloading

The assignment step in spherical K-means is the clear MVP. Your shape is:

- `X`: 323,000 × 128  
- `C^T`: 128 × 98  
- output: 323,000 × 98

That is about **8.1 GFLOP** per iteration. More importantly, the data movement is favorable. Uploading `X` once is about **158 MiB** in float32. Uploading centroids each iteration is only about **49 KiB**. Downloading labels each iteration is about **1.3 MiB** if you use `u32`. In other words, the per-iteration PCIe traffic is tiny if `X` stays resident on the device. This is exactly the sort of repeated GEMM where a GPU wins decisively. The straightforward implementation is GEMM followed by a rowwise argmax kernel; I would not try to write a custom fused GEMM+argmax kernel for the MVP. The separate reduction is easier to validate and almost certainly “fast enough” at `K=98`. 

Regular K-means also benefits, but not all of it equally. Your low-dimensional `d < 20` path is already very fast on CPU, and even at `n=100K, d=32, K=32` you reported about 70 ms per fit. Those are the cases where kernel launch, device init, and copy overhead can erase much of the GPU gain. The honest strategy is to keep **small and low-dimensional K-means on CPU by default**, and only dispatch to GPU above a shape threshold. FAISS’s long-standing k-means benchmarks are useful as a sanity check here: even older GPUs cut brute-force training time dramatically when the problem is large enough, but they also show CPU and transfer overhead becoming important once the problem is spread across too many GPUs or the GEMMs become too small. citeturn11view4

Randomized PCA is the second-best target. Your one-shot `X @ Omega` is about **136.9 GFLOP** and reads an `X` matrix that is roughly **2.0 GiB** in float32. That is still large enough that a good GPU backend will win comfortably, even if it is “only” a single matmul. If the GPU backend is already initialized because the caller chose `backend="cuda"` or `backend="metal"`, then PCA should go on the same backend. If the rest of the pipeline is CPU-only, the value is smaller but still real, because the operation is big enough to amortize one device setup and one large transfer. 

vMF responsibilities are also worth doing as soon as the assignment machinery exists. The computational core is again a dot-product score matrix plus a rowwise normalization, so most of the infrastructure is the same as spherical K-means assignment: GEMM, then rowwise reduction and normalization. That makes it a strong **phase-two** feature after assignment and PCA.

L2 normalization is not a first target. Rowwise norm-and-divide is trivially parallel but has low arithmetic intensity. It becomes worth doing **only when the data is already on GPU** because you are about to do assignment or vMF there anyway.

Centroid updates should stay on CPU for MVP. Scatter-heavy accumulation by label is much less elegant on GPU than GEMM, and you yourself called it fast enough today. I would change that only after measuring the end-to-end fit with GPU assignment.

DBSCAN and HDBSCAN should not be in version one of the GPU work. Yes, the brute-force distance phase can be reframed as norms plus tiled dot products, and yes, the GPU can dominate that math. But at `n=50K`, the full pairwise matrix is about **10 GB in fp32** or **20 GB in fp64**, which means you are now in the business of tiling, thresholding, sparse graph construction, and downstream cluster logic. RAPIDS’s HDBSCAN speedups are impressive because they accelerate the full pipeline, not just one tiled distance kernel. That is a very different project from “optional GPU acceleration for `rustcluster` hot GEMMs.” citeturn11view5turn13view1

## A hybrid architecture that fits rustcluster

The right architecture is **GPU for score generation, CPU for control flow and update logic**.

The basic loop should look like this:

1. Keep the original host copy of `X` for compatibility and CPU updates.  
2. If a GPU backend is selected and the shape clears a threshold, upload `X` once at the start of `fit()`.  
3. Each iteration, upload the current centroids.  
4. Run GEMM on the GPU to produce scores.  
5. Run a GPU rowwise argmax kernel to produce labels.  
6. Download labels to CPU.  
7. Do centroid accumulation, normalization, convergence checks, and bookkeeping on CPU.  

On discrete GPUs, this keeps the expensive transfer — `X` — as a one-time cost and makes the per-iteration transfer budget tiny. On Apple Silicon, the same structure still makes sense, but the penalty for moving between CPU and GPU execution contexts is much lower because buffers live in shared system memory once you commit to GPU-native allocations. Apple’s shared-storage and MLX unified-memory docs both support that design direction. citeturn15search0turn15search3turn39view2

A thin backend trait is enough for the first version:

```rust
pub trait AssignBackend {
    fn init(x_f32: &[f32], n: usize, d: usize) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn assign(
        &mut self,
        centroids_f32: &[f32], // K x d
        k: usize,
        labels_out: &mut [u32],
    ) -> anyhow::Result<()>;
}

pub enum Backend {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(CudaAssign),
    #[cfg(feature = "metal")]
    Metal(MetalAssign),
}
```

And the fit loop can remain recognizably yours:

```rust
for init in 0..n_init {
    let mut centers = init_centers(...);
    for iter in 0..max_iter {
        backend.assign(&centers, k, &mut labels)?;
        update_centers_cpu(x_host, &labels, &mut centers, k, d);
        normalize_centers(&mut centers);
        if converged(...) { break; }
    }
}
```

That design buys you three things at once. It preserves your CPU implementation as the source of truth. It gives you a single place to add CUDA now and Metal later. And it keeps the public Python surface simple enough that `backend="auto"` can just mean “use an accelerator if present and if the shape is large enough, otherwise fall back to CPU.”

## Packaging and CI reality

This is the part that decides whether the project feels elegant or painful.

Cargo features are compile-time, additive switches. The Cargo book is explicit about that, and `maturin` is simply building a Cargo artifact and packaging the result. That means one wheel cannot magically decide to compile in CUDA later. Whatever you publish has already committed to a backend set. On Linux, `maturin` also reminds you that portable wheels must obey manylinux-style dynamic-linking rules, which is exactly where GPU runtimes become awkward. citeturn35search0turn35search1turn17search4turn35search9

That is why major projects still struggle here. The 2025 PyTorch wheel-variants post says the current install matrix still forces users to pick accelerator versions and custom indexes, and presents hardware-specific wheel variants as the future solution precisely because specialized binary packaging is so messy today. RAPIDS’ own packaging story is likewise a warning sign: current docs still emphasize Linux and WSL2 support, and NVIDIA had to do dedicated work on reducing CUDA binary size before shipping simplified cuML wheels on PyPI. citeturn36view0turn39view3turn39view4turn37search6

For `rustcluster`, the cleanest distribution model is:

- `rustcluster`: unchanged CPU wheel, published exactly as today.
- `rustcluster-cuda12`: optional plugin package, Linux x86_64 first.
- later `rustcluster-metal`: optional macOS arm64 plugin package, if you decide the Apple path is worth productizing.

If you want a nicer UX, expose extras from the base package so users type:

```bash
pip install rustcluster
pip install "rustcluster[cuda12]"
```

where the extra simply depends on the plugin distribution.

This approach is much safer than a single all-purpose wheel. It preserves the “pip install and it works” CPU story, isolates CUDA packaging risk to opt-in users, and allows you to ship and deprecate backend packages independently.

From a build perspective, custom CUDA compilation works fine with Cargo build scripts:

```rust
// build.rs
fn main() {
    cc::Build::new()
        .cuda(true)
        .file("src/kernels/assign_argmax.cu")
        .compile("rustcluster_cuda_kernels");
}
```

The builder machine will need nvcc or an equivalent toolkit setup; `cudarc` also documents build-time CUDA-version feature selection and nvcc-based toolkit detection. End-user source builds therefore require CUDA development tools, but end-user **binary wheel installs should not rely on source builds at all** if you want a humane UX. citeturn18search0turn18search1turn18search11turn43view0

For CI, entity["company","GitHub","software platform"] Actions now documents GPU-powered larger runners, specifically T4-backed Ubuntu and Windows configurations. That means you can keep normal wheel compilation on ordinary runners or controlled build images, and reserve actual runtime tests and smoke benchmarks for a GPU runner or self-hosted machine. The catch is organizational availability and per-minute billing, so it is useful — but not free and not universal. citeturn42view0turn42view1

## Competitive signals and realistic speedups

The strongest external signal comes from entity["company","Meta","technology company"]’s FAISS. FAISS documents spherical k-means directly, and its low-level benchmark page shows brute-force k-means on 1M vectors in 256D to 20k centroids taking **11 minutes on CPU**, **55 seconds on one P100 in float32**, and **34 seconds on one P100 in float16**. Those are old numbers on now-old hardware, but they establish the important point: once clustering becomes “big repeated assignment GEMMs,” GPU brute force is not just viable, it is often the simplest fast path. citeturn31search0turn31search1turn11view4

cuML points the same way. RAPIDS’ current docs still summarize typical speedups in the **10–50×** range, its earlier K-means blog reported **100×+** acceleration on million-sample runs versus scikit-learn, and the 2025 zero-code-change release claimed up to **50×** for accelerated scikit-learn workflows and much larger gains for HDBSCAN. The important caveat for your specific use case is that I did not find a clean official cuML KMeans story for spherical/cosine mode; there is at least a public RAPIDS issue asking for cosine distance support in KMeans, which suggests that your spherical embedding workflow is not a perfect drop-in match for cuML’s KMeans surface today. citeturn39view3turn13view0turn13view1turn32search2

Apple’s MLX is relevant mostly as a platform signal, not as a Rust integration answer. MLX exposes matmul and is explicitly optimized for Apple silicon unified memory, but it is an array framework rather than a ready-made clustering library. It is excellent evidence that Apple silicon can make this style of workload go fast, but it does not remove the need for a Rust-side backend if you want `rustcluster` itself to own the accelerator path. citeturn11view2turn39view2turn10search3

### Estimated speeds for your shapes

The table below is an **engineering estimate**, not a claimed benchmark result. I derived it from your exact workload sizes, the published hardware capabilities of the RTX 4090, A100, and Apple silicon memory-bandwidth tiers, plus public clustering/GEMM benchmark envelopes from FAISS, cuML, and WebGPU SGEMM work. For Apple, I assume the higher-end M3 Max configuration with 40 GPU cores and 400 GB/s bandwidth; for baseline M1 I use Apple’s own statement that M1 Max’s 400 GB/s is nearly 6× M1 bandwidth. These are **kernel-time estimates** for a tuned vendor-library backend, not full-application wall time. End-to-end fit speedups will be smaller because labels, centroid updates, convergence checks, and Python/Rust orchestration remain on CPU in the recommended MVP. citeturn39view0turn39view1turn11view1turn38search2turn11view4turn13view0turn13view1turn20view0

| Operation | Size | Current CPU | RTX 4090 | A100 | M3 Max | M1 |
|---|---:|---:|---:|---:|---:|---:|
| Spherical K-means assign | 323K × 128 × 98 | ~4.2 s/iter from your 633 s run | ~4–8 ms/iter | ~3–6 ms/iter | ~10–25 ms/iter | ~30–70 ms/iter |
| Pairwise distances, tiled | 50K × 50K × 128 | ~60–180 s inferred for brute-force CPU | ~0.5–1.5 s | ~0.3–1.0 s | ~2–6 s | ~6–15 s |
| PCA matmul | 323K × 1536 × 138 | ~2 s from your profile | ~15–35 ms | ~10–25 ms | ~40–90 ms | ~120–250 ms |
| vMF responsibilities | roughly assignment-shaped | ~4–6 s inferred | ~6–15 ms | ~4–10 ms | ~15–40 ms | ~40–90 ms |

The practical interpretation is:

- **CUDA overall**: for your EmbeddingCluster path, a carefully scoped CUDA backend should realistically deliver **10–40× end-to-end speedup** depending on how much time is truly in assignment versus update/bookkeeping.
- **Metal overall**: on Apple silicon, **5–20× end-to-end** is realistic for the same path, with better UX and lower transfer pain than discrete GPUs, but a lower absolute ceiling than good NVIDIA hardware.
- **`wgpu` overall**: for these same GEMM-heavy kernels, it is unlikely to match either of those without a very large tuning effort, and even then you would still trail vendor BLAS. citeturn20view0turn23view1turn11view4turn13view0

## Recommended path

The concrete recommendation is:

**Build one CUDA-backed optional plugin first, and accelerate only three operations:**
the spherical K-means assignment GEMM, randomized PCA’s projection matmul, and the shared GEMM-plus-rowwise-reduction path you can reuse for vMF responsibilities.

That is the smallest surface area that still makes a visible product difference. It directly targets the hot loops you measured, reuses the same backend abstractions across multiple algorithms, and avoids overcommitting to GPU-ifying the entire library.

I would rank the options like this:

**CUDA plus `cudarc` plus cuBLAS/cuBLASLt** is the best MVP. It has the best performance ceiling, the best library support for GEMM, and the clearest route to a backend that is “small but serious.” Use `.cu` or NVRTC only for the thin custom kernels that cuBLAS does not give you. Do not try to invent your own GEMM. citeturn43view0turn43view2turn18search0

**Metal plus MPS** is the best second backend. Do it after the CUDA path stabilizes and only if you want first-class Apple silicon acceleration. Use `objc2-metal` and `objc2-metal-performance-shaders`, not the deprecated `metal` crate. Keep the API identical to the CUDA backend so the Python side can stay simple. citeturn16search3turn16search0turn16search2

**Do not use `wgpu` for the MVP.** It is a good portability technology and a very real compute API, but not the best answer to “I need a GEMM fast enough to collapse a 633-second clustering run.” It is the wrong first project if speedup-per-engineering-week is the metric. citeturn24search7turn23view0turn20view0turn23view1

**Do not spend the first release on DBSCAN or HDBSCAN GPU support.** If you eventually want that, treat it as a separate project after the GEMM backend lands and proves its value. cuML’s HDBSCAN speedups are real, but they come from accelerating much more than a single distance kernel. citeturn11view5turn13view1

In timeline terms, a realistic effort estimate is:

- **Prototype CUDA MVP:** about 1–2 engineering weeks  
  backend trait, one-time device upload, GEMM assignment path, rowwise argmax, Python option switch, thresholds
- **Productionize CUDA path:** another 2–4 weeks  
  wheel builds, import-time fallback behavior, CI, docs, shape heuristics, testing on no-GPU systems, driver/runtime edge cases
- **Metal second backend:** another 2–4 weeks  
  MPS plumbing, buffer ownership model, parity testing on Apple silicon
- **Cross-vendor `wgpu` path instead:** easily 6–10+ weeks for a result that still underdelivers on the specific GEMM-heavy workloads you care about

So the honest answer is **not** “don’t bother.” The honest answer is:

**bother, but keep the scope brutally narrow.**  
If you do that, GPU acceleration for `rustcluster` is realistic, strategically clean, and likely to pay off immediately on the workloads you actually measured.