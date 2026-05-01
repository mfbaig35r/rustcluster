# Deep Research Prompt: BLAS Backend for rustcluster.index

> ## Context
>
> I maintain `rustcluster`, a Rust-backed Python clustering library on PyPI
> (v0.6.0). Built with maturin + PyO3, ships pure-CPU wheels for
> Linux/macOS/Windows, no BLAS dependency.
>
> v1.0 added `rustcluster.index` — a flat exact vector search submodule
> (IndexFlatL2, IndexFlatIP) with `add_with_ids`, `search`, `range_search`,
> `similarity_graph` (a fused all-pairs above-threshold kernel), and
> safetensors-based persistence. Built around faer 0.24 for SIMD GEMM.
>
> v1.1 closed perf gaps via parallelism (matmul → `Par::Rayon`,
> per-query loops → rayon par_iter) and tile-size tuning. Across
> n ∈ [10k, 100k] and d ∈ {128, 384, 1536} on Apple Silicon:
> - `similarity_graph` at d=128 is **2.3× faster** than the FAISS batched
>   range_search loop.
> - At d=1536 it's **0.77× of FAISS** (i.e., 1.3× slower).
> - `search` and `range_search` are 1.3-3.7× slower than FAISS standalone
>   (parallel BLAS + fused kernel is what FAISS has that we don't).
>
> The remaining gap at d=1536 is faer's pure-Rust GEMM vs hand-tuned BLAS.
> faer is excellent but lags Apple's Accelerate, OpenBLAS, and MKL by
> 20-30% for large-K matmul shapes — exactly what raw embedding dimensions
> like `text-embedding-3-small` (1536d) produce.
>
> **The question this prompt is scoping:** should v1.2 ship with an
> optional, multi-platform BLAS-backed build to close that gap?
>
> ## What I Want to Build
>
> "Option 2" from `docs/index-performance-plan.md`: numpy-style
> per-platform BLAS bundling, behind a Cargo feature, ideally as a single
> wheel that does the right thing per OS/architecture.
>
> Concretely:
> - `cargo build --features blas` swaps faer's `matmul` for a BLAS sgemm
>   call inside `ip_batch` (and the per-tile matmul in `similarity_graph`).
> - `pip install rustcluster` (the default wheel) auto-uses BLAS where
>   available without the user having to choose.
> - Performance approaches FAISS at d=1536 across all common platforms.
> - The "pip install just works" property is preserved — no system BLAS
>   install required, no separate package to install.
>
> Constraints I care about:
> - Solo-maintained project. CI complexity has a real cost.
> - Wheel size matters but is not the most important thing.
> - Must work on Linux x86_64, Linux aarch64, macOS x86_64, macOS aarch64,
>   Windows x86_64. (These are what numpy/scipy build for.)
> - Should NOT require users to install anything beyond `pip install rustcluster`.
>
> Where you state a number, cite the source. Where you make a recommendation,
> commit to it — don't just lay out tradeoffs.
>
> ## What I Need Researched
>
> ### 1. The reference implementation: numpy / scipy
>
> numpy and scipy are the gold standard for "Python library that bundles
> BLAS." Document specifically:
>
> - **What BLAS do they bundle on each platform?** (OpenBLAS variant on
>   Linux, Accelerate on macOS via dynamic linking, OpenBLAS on Windows?
>   Has anything changed recently — e.g., scipy_openblas as a separate
>   package?)
> - **How do they pin the version?** Reproducible builds, security updates.
> - **Wheel size impact.** What does OpenBLAS contribute to a numpy wheel?
> - **What's `scipy-openblas64` / `numpy-openblas`?** I've seen these as
>   separate PyPI packages — when do they apply?
> - **The build toolchain.** numpy switched to meson; what does that mean
>   for a maturin-based project that wants to follow their pattern?
> - **ABI considerations.** Does numpy expose its BLAS symbols, or are
>   they private? If we link our own OpenBLAS, do they collide?
> - **Threading.** numpy's BLAS uses pthread/OpenMP. Does this conflict
>   with Python threads, GIL, rayon? What does numpy do about it?
>
> Cite specific links to numpy's `.github/workflows/wheels.yml`,
> `meson.build`, or wheel-building documentation.
>
> ### 2. How FAISS distributes wheels
>
> FAISS-CPU is the most direct comparison point. Document:
>
> - The wheel-building pipeline for `faiss-cpu` (the PyPI package).
> - Which BLAS variant ships in the precompiled wheels per platform.
> - Wheel size.
> - Their CMake/swig build matrix.
> - Anything we should explicitly not copy.
>
> ### 3. usearch's choice (the alternative we'd be rejecting)
>
> usearch ships with **SimSIMD**, a custom SIMD library hand-tuned for
> distance kernels. They explicitly chose to NOT depend on BLAS.
>
> - Why? What's their stated reasoning?
> - SimSIMD performance vs OpenBLAS / Accelerate at typical embedding
>   shapes (d=128, 384, 1536).
> - Could rustcluster use SimSIMD instead of going down the BLAS road?
>   (Rust bindings? License — Apache-2.0? FFI overhead?)
> - What does usearch lose by skipping BLAS, if anything?
>
> If SimSIMD looks viable, that's a contender alongside the BLAS path —
> evaluate honestly.
>
> ### 4. Rust BLAS ecosystem for maturin/PyO3 projects
>
> The relevant crates:
> - **`blas-src`** — provider-agnostic BLAS facade. Pick at compile time.
> - **`accelerate-src`** — links Apple Accelerate framework.
> - **`openblas-src`** — bundles or links OpenBLAS.
> - **`intel-mkl-src`** — bundles MKL (licensing concerns).
> - **`netlib-src`** — reference implementation, slow.
> - **`cblas-sys` / `blas-sys`** — raw FFI declarations.
> - **`gemm` / `gemm-f32`** — the crate faer is built on.
>
> For each, give: maturity (last release, GitHub activity), license,
> bundles vs links dynamically, build dependencies (Fortran compiler?),
> Windows support quality, aarch64 support quality.
>
> Specifically for `blas-src` + `openblas-src`:
> - Can it produce a self-contained Linux wheel without a system
>   OpenBLAS install? (i.e., the OpenBLAS `.so` ships inside the wheel)
> - Same question for Windows.
> - Same question for macOS — or do we use accelerate-src there?
>
> ### 5. ABI clash with numpy's bundled OpenBLAS
>
> This is the concern that worries me most. If both numpy and rustcluster
> bundle their own OpenBLAS in their wheels, what happens when both are
> imported into the same Python process?
>
> - Does Linux's symbol resolution merge them? Pick the first loaded?
>   ODR-violation territory?
> - Does numpy's recent move to "namespace its OpenBLAS symbols" (via
>   the `scipy-openblas-libs` infrastructure) help here?
> - Are there documented patterns for this — what do other libraries
>   that bundle BLAS (FAISS, lightgbm, xgboost) do?
> - Has anyone hit this in practice and what was the resolution?
> - Worst case: two OpenBLAS instances, each with its own thread pool,
>   competing for cores. How bad is this in practice?
>
> Specifically for our case: rustcluster gets imported alongside numpy
> in 100% of users' code. We MUST avoid breaking numpy's BLAS.
>
> ### 6. Apple Silicon Accelerate framework
>
> macOS ships Accelerate as part of the OS. Always available, no bundling
> needed.
>
> - **Auto-link via `accelerate-src` or framework attribute** —
>   what's the simplest path?
> - **Performance.** At d=1536, sgemm via Accelerate's AMX vs OpenBLAS:
>   what's the realistic speedup? Cite benchmarks.
> - **Code signing.** Does linking against `/System/Library/Frameworks`
>   require any wheel-signing or notarization considerations?
> - **macOS x86_64 vs arm64.** Both supported by Accelerate? Performance
>   parity?
> - **Threading.** Accelerate uses GCD; does it conflict with rayon
>   or Python threads?
>
> ### 7. Linux BLAS landscape
>
> - **OpenBLAS** — the obvious default. Last release, active maintenance,
>   manylinux compatibility, aarch64 (Graviton) support.
> - **MKL** — fastest on Intel CPUs but licensing is restrictive
>   (intel-mkl-src bundles the redistributable). Worth the complexity?
> - **BLIS** — alternative; performance? distribution model?
> - **Apple Silicon on Linux** — does this exist in production? (Asahi
>   Linux, anyone shipping aarch64 Linux desktops with Apple Silicon?)
> - **Manylinux constraints.** Which manylinux versions can ship
>   bundled OpenBLAS? Are we forced to manylinux_2_28+ or can we hit
>   manylinux2014?
>
> ### 8. Windows BLAS landscape
>
> - **OpenBLAS for Windows** — distribution model. DLL bundled in wheel?
>   Static link? Both work?
> - **MKL on Windows** — same question, plus licensing.
> - **What does numpy/scipy actually ship?** This is probably the
>   reference answer.
> - Edge cases (Windows ARM64?).
>
> ### 9. cibuildwheel + maturin integration
>
> - Has anyone built a maturin-based project with bundled BLAS via
>   cibuildwheel?
> - What does the GitHub Actions matrix look like?
> - Are there template repos to copy from?
> - Build time and CI-cost implications.
>
> ### 10. Runtime BLAS detection (alternative to bundling)
>
> Could we instead detect BLAS at import time and dlopen whatever's
> available?
>
> - Pros: smaller wheel, uses whatever the user already has.
> - Cons: behavior depends on environment, harder to support.
> - Has anyone done this? (sklearn does something like this with its
>   threadpoolctl.)
> - Is the per-import cost acceptable?
>
> Compare to bundling. State which is the better engineering choice
> for our case.
>
> ### 11. Single-wheel vs multi-package distribution
>
> Three patterns to evaluate:
>
> **A. Single wheel, BLAS bundled.** What numpy does. Best UX, biggest
> wheel.
>
> **B. Two packages: `rustcluster` (pure Rust) and `rustcluster-blas`.**
> Like `tensorflow-cpu` vs `tensorflow-gpu` historically. User opts in
> to BLAS.
>
> **C. Single wheel, BLAS optional Cargo feature, default off.** User
> who wants BLAS rebuilds from source.
>
> For each, state: total wheel size if applicable, user experience,
> maintenance complexity, performance for the default user. Recommend one.
>
> ### 12. Realistic performance expectations
>
> If we ship with a BLAS backend, what's the realistic d=1536 number
> on each platform? Cite benchmarks where possible:
>
> - macOS Apple Silicon (M1/M2/M3) with Accelerate
> - Linux x86_64 with OpenBLAS (Skylake-class, AVX-512)
> - Linux x86_64 with MKL
> - Linux aarch64 (Graviton 3) with OpenBLAS
> - Windows x86_64 with OpenBLAS
>
> Specifically: at our `similarity_graph` workload (n=20k, d=1536, threshold
> matched), where do we land vs FAISS?
>
> ### 13. Maintenance burden
>
> Honest assessment of ongoing cost:
>
> - CI matrix expansion (current matrix vs new matrix size).
> - BLAS version upgrades (security CVEs in OpenBLAS happen).
> - Per-platform debugging (a Windows-only BLAS issue is painful).
> - Documentation updates.
>
> Estimate engineer-hours per year of maintenance once shipped.
>
> ### 14. Final recommendation
>
> Pull it all together:
>
> - **Verdict: should we add BLAS support, or stay pure-Rust?**
>   If the maintenance burden + ABI clash risk + wheel-size growth
>   outweigh the d=1536 speedup, the right answer is to NOT ship BLAS
>   and document the gap. State your verdict explicitly.
> - **If yes to BLAS: which pattern (A/B/C from §11)?**
> - **Effort estimate** (engineer-weeks for initial implementation +
>   stable CI).
> - **Risk assessment.** Where is this most likely to go wrong?
> - **A specific concrete plan** — what to do first, what's milestone 1.
> - **Decision criteria for stopping.** When do we know it's not worth
>   pushing further?
>
> ## Deliverables
>
> For every section:
> 1. Concrete numbers, not vague assessments. Cite sources (papers,
>    benchmarks, GitHub repos with file paths and line numbers).
> 2. Code examples or pseudocode for any non-obvious build/link choice.
> 3. Honest maturity/risk assessment for every external dependency
>    you recommend.
> 4. A final ranked recommendation with tradeoffs explicitly stated,
>    not hedged.
