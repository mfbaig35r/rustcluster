# Deep Research Prompt: Hand-Tuned Rust SIMD Kernels for rustcluster.index

> ## Context
>
> I maintain `rustcluster`, a Rust-backed Python clustering library on PyPI.
> v1.0 added `rustcluster.index` — flat exact vector search (IndexFlatL2,
> IndexFlatIP) plus a fused all-pairs `similarity_graph` kernel. Built
> around faer 0.24 for SIMD GEMM. Full pure-Rust, no BLAS dependency.
>
> v1.1 closed perf gaps via parallelism. Across n ∈ [10k, 100k] and
> d ∈ {128, 384, 1536} on Apple Silicon:
> - `similarity_graph` at **d=128 is 2.3× faster than FAISS** (the
>   batched range_search loop).
> - At **d=1536 it's 0.77× of FAISS** (i.e., 1.3× slower).
> - `search` and `range_search` are 1.3-3.7× slower than FAISS standalone.
>
> Phase-1 profiling (committed under `examples/profile_*.rs`) showed the
> remaining gap at d=1536 is **99.9% the SIMD microkernel itself** —
> there's no overhead at the call site to optimize. Pure-Rust `gemm`
> (faer's underlying matmul) lags hand-tuned BLAS by 20-30% at large `d`.
>
> A separate research prompt is investigating the BLAS-backend route
> (`docs/blas-backend-research-prompt.md`). This prompt is its
> alternative: **what would it take to write our own SIMD kernels in Rust
> instead of bundling BLAS?**
>
> usearch took this path with **SimSIMD** — a ~3000-LOC C++ library of
> hand-tuned distance kernels. They explicitly chose SimSIMD over BLAS
> and ship a single small wheel that beats FAISS for vector-search
> workloads. The question for rustcluster is whether a similar choice
> works for us, ideally as a Rust-native library.
>
> ## What I Might Build
>
> A focused SIMD distance-kernel layer — **not** a general BLAS, just
> the operations that vector search uses:
>
> 1. **Single-vector dot product / L2 / cosine** at any `d`, AVX2/AVX-512/NEON.
> 2. **Batched dot products** `Q (nq, d) @ X^T (d, n) → S (nq, n)` for
>    `search` and `range_search`. Could call a Rust GEMM here, or fuse.
> 3. **Fused GEMM + threshold + emit** for `similarity_graph` — compute
>    a tile of inner products and immediately run them through the
>    threshold filter, never materializing the full score matrix. This
>    is FAISS's secret to beating parallel BLAS at top-k (22ms vs the
>    44ms numpy parallel BLAS does for the same matmul alone).
> 4. **Possibly Apple AMX direct access** for Apple Silicon (undocumented
>    but stable across M-series, ~3× speedup at d≥512 for f32 sgemm).
>
> Constraints:
> - **Stays pure Rust** (or with a thin C/C++ FFI dep at most). Must keep
>   the "pip install just works" property — single small wheel,
>   no bundled BLAS, no manylinux pain.
> - **f32 only** (matches v1).
> - Must work on Linux x86_64, Linux aarch64, macOS arm64, macOS x86_64,
>   Windows x86_64.
> - Solo-maintained project — multi-week scope is acceptable, multi-year
>   is not.
>
> Where you state a number, cite the source. Where you make a
> recommendation, commit to it.
>
> ## What I Need Researched
>
> ### 1. SimSIMD: what it actually is and how it works
>
> Document specifically:
>
> - **Architecture** — module layout, kernel naming conventions.
> - **Code size** — how many LOC, broken down by per-architecture
>   intrinsics vs shared scaffolding.
> - **Operations covered** — `dot_f32`, `cos_f32`, `l2sq_f32`, plus
>   what else (BF16, int8, sparse)?
> - **Per-architecture coverage** — AVX2, AVX-512, NEON, SVE, others.
>   Which paths are well-tuned vs "good-enough fallback"?
> - **Performance vs OpenBLAS / Accelerate** — at d ∈ {128, 384, 1536}
>   for f32. SimSIMD's own benchmarks plus any third-party numbers.
> - **License** — Apache-2.0? MIT-compatible?
> - **API style** — header-only? Linkable lib? How do consumers use it?
> - **Maintenance** — release cadence, who works on it, recent CVEs or
>   correctness issues.
>
> ### 2. SimSIMD via Rust bindings: wrap vs port
>
> The `simsimd` crate exists on crates.io. Evaluate:
>
> - **Maturity, last release, GitHub stars, downloads.**
> - **API quality** — idiomatic Rust? Or thin C-ABI?
> - **Build dependencies** — does it require a C/C++ compiler? Can it
>   build with vendored sources or does it want system SimSIMD?
> - **Wheel-distribution implications** — when used in a maturin/PyO3
>   project, does the resulting wheel bundle SimSIMD's binary, or does
>   the user need it installed?
> - **Performance vs hand-rolled Rust** — is the FFI overhead negligible
>   for our shapes (small per-call, repeated across tiles)?
>
> Verdict: **wrap the existing C++ SimSIMD via the `simsimd` crate, or
> port it to Rust?** Be opinionated. The trade-off is "ship faster but
> take on a C++ dep" vs "stay pure Rust but spend months porting."
>
> ### 3. Pure-Rust SIMD landscape (if porting)
>
> If we port SimSIMD or write our own kernels from scratch, the Rust
> SIMD options are:
>
> - **`std::simd`** — portable SIMD, currently nightly-only. When does
>   it stabilize? Is there a workaround for stable Rust?
> - **`std::arch::*`** — architecture-specific intrinsics on stable.
>   `_mm256_*` for AVX2, `_mm512_*` for AVX-512, `vld1q_*` for NEON.
>   Behind `#[target_feature]`, ergonomics rough.
> - **`pulp`** — what faer uses. Auto-vectorization wrapper, ~stable.
>   Could we use pulp's lower-level API directly to write distance
>   kernels?
> - **`wide`** — portable SIMD wrapper crate. Quality, performance.
> - **`packed_simd`** — older, abandoned?
> - **`generic-simd`** / **`safe_arch`** / **other crates** —
>   honest assessment of which are alive and usable today.
>
> For each, give: stable-Rust support, target_feature ergonomics, code
> ugliness for a typical dot-product loop, performance vs `std::arch`
> intrinsics directly.
>
> Recommend ONE primary approach for a hand-rolled kernel.
>
> ### 4. The fused GEMM + threshold + emit kernel
>
> This is the architecturally interesting piece — what FAISS does that
> a normal BLAS user can't.
>
> For our `similarity_graph` workload, the inner work is:
>
> ```text
> for each tile (i_block, j_block):
>     compute X[i_block] @ X[j_block]^T → tile-shaped score matrix
>     for each (i, j) in tile:
>         if score[i, j] >= threshold:
>             emit (i, j, score[i, j])
> ```
>
> Today we materialize the score tile (a `tile × tile` Array2<f32>)
> then run a separate threshold loop. A fused kernel would maintain a
> small per-row score buffer in registers, emit edges inline, and
> never write the full score matrix to memory.
>
> - **What does FAISS's fused kernel actually look like?** Code-level
>   detail — register blocking, what's kept in registers vs L1.
> - **Realistic speedup** vs "GEMM then threshold" at d=1536 on (a)
>   Apple Silicon, (b) AVX-512.
> - **Implementation effort** in pure Rust intrinsics.
> - **Could we share register-blocked microkernels with `pulp` or roll
>   our own?**
>
> The same fused-kernel approach also wins for `search` (top-k) and
> `range_search` — the FAISS 22ms-vs-44ms result is exactly this.
>
> ### 5. Apple AMX direct access
>
> Apple Silicon has a hidden matrix accelerator (AMX) that ~3× outpaces
> NEON for f32 sgemm. Officially undocumented but reverse-engineered
> and stable across M1/M2/M3.
>
> - **What Rust crates exist?** (`amx-aarch64`, others?). Maturity.
> - **Stability commitment from Apple** — has the AMX ISA changed
>   between M-series generations? Is Accelerate the only Apple-blessed
>   way to use it?
> - **Risk** — if Apple deprecates or breaks AMX, what's the fallback?
> - **Performance at d=1536** on M1/M2/M3.
> - **Effort to integrate** — wrap an existing crate vs roll our own.
>
> Specific question: if the Linux x86_64 path is "hand-tuned AVX-512
> kernels" and the macOS arm64 path is "AMX direct," is that
> maintenance burden acceptable? Or should macOS just use NEON-only
> kernels and accept being 3× behind Accelerate on Apple?
>
> ### 6. Wrap vs build, definitively
>
> Three concrete paths:
>
> **A. Wrap SimSIMD via the existing `simsimd` Rust crate.**
> Lowest effort. Inherits SimSIMD's coverage, performance, maintenance.
> Adds a C++ build dependency (need to evaluate how invasive).
>
> **B. Port SimSIMD's algorithms to pure Rust.** Medium effort, ~3-6 months
> for full coverage. Stays pure Rust. Bigger maintenance cost long-term.
>
> **C. Hand-roll our own focused kernels** for just the operations
> rustcluster needs (single-vector dot/L2/cos, plus the fused GEMM
> kernel). Smaller scope than SimSIMD because we only need 3-4
> operations, not 20+. Maybe 2-3 months.
>
> For each: realistic effort, realistic maintenance, realistic
> performance ceiling. Be opinionated.
>
> Specific concern about A: how much does adding a C++ build dependency
> cost vs the current zero-C-dep state? Is it materially different
> from "BLAS bundled" in distribution complexity, or is SimSIMD's
> small footprint genuinely cheap to ship?
>
> ### 7. Comparison: SimSIMD-style vs BLAS backend
>
> The other research prompt (`blas-backend-research-prompt.md`) is
> evaluating the BLAS path. Compare directly across these axes:
>
> | | SimSIMD-style | BLAS backend |
> |---|---|---|
> | Distribution complexity | ? | ~hard (manylinux + ABI clash with numpy) |
> | Wheel size growth | ? | ~30 MB on Linux |
> | Performance ceiling | ? | ~1.0× FAISS |
> | Effort to ship | ? | 1-2 weeks |
> | Maintenance | ? per arch | annual (CVEs, version pins) |
> | Pure Rust property | preserved | broken (C deps either way) |
> | Fused-kernel ability | yes | no (BLAS is opaque) |
> | Could BEAT FAISS | yes | no |
>
> Fill in the SimSIMD column. The "could beat FAISS" row is the
> distinctive thing — fused kernels matter.
>
> ### 8. License and distribution
>
> - **SimSIMD license** — Apache-2.0?
> - **Compatibility with rustcluster's MIT license** — a wrapped Apache
>   dep is fine; what about a port (we'd be re-implementing under MIT,
>   but inspired by Apache code — is that clean)?
> - **Wheel size** — does adding SimSIMD bindings grow the wheel by
>   meaningful amounts? Compare to current pure-Rust wheel size.
> - **Build matrix** — does this work with our existing maturin +
>   cibuildwheel setup, or does it require new toolchain?
>
> ### 9. Testing and correctness
>
> Hand-tuned SIMD code is famously error-prone. What's the testing
> story?
>
> - **Per-architecture testing** — how do we verify NEON kernels on
>   x86 CI runners? GitHub Actions has aarch64 runners now.
> - **Property-based testing** — verify SIMD output equals scalar
>   reference within float tolerance, randomized.
> - **What does usearch / SimSIMD do for testing?**
>
> ### 10. Realistic performance projections
>
> Based on usearch / SimSIMD's published benchmarks plus what's
> achievable with hand-tuned Rust SIMD:
>
> - At **d=1536, n=20k, similarity_graph** (our production-shaped
>   workload): what wall-clock could we realistically hit on:
>   - Apple Silicon M2 / M3 with NEON kernels?
>   - Apple Silicon M2 / M3 with AMX direct?
>   - Linux x86_64 with AVX-512 hand-tuned?
>   - Linux x86_64 with AVX2 only (older CPUs)?
>   - Linux aarch64 (Graviton) with NEON?
> - Compare to current 870ms (faer) and FAISS's 540ms.
>
> ### 11. Effort and risk
>
> Concrete estimates:
>
> - **Path A (wrap existing simsimd crate):** engineer-weeks for
>   integration, plus risk assessment (what could go wrong?).
> - **Path C (hand-roll focused kernels):** engineer-weeks per
>   architecture for production-quality kernels. Where does effort
>   actually go — kernel writing, testing, debugging?
> - **Path B (port SimSIMD):** lowest probability of being the right
>   choice; estimate it briefly to confirm.
>
> Where is each path most likely to fail? Be specific.
>
> ### 12. The library-author question
>
> Step back from rustcluster's specific workloads. For a Python vector
> search library aimed at general users (people with OpenAI / Cohere
> embeddings at d=1024-3072), which is the better long-term choice:
>
> - Pure-Rust SIMD kernels (this prompt's path)
> - Bundled BLAS (the other prompt's path)
>
> Specifically: **how does the field's center of gravity feel?**
> usearch chose SimSIMD; numpy/scipy chose BLAS. Which is more
> aligned with where Rust ML / vector search is heading?
>
> ### 13. Final recommendation
>
> Pull it all together:
>
> - **Verdict: build hand-tuned SIMD, wrap SimSIMD, or stay pure-faer
>   and accept the d=1536 gap?**
> - **If yes to SIMD: which path (A / B / C from §6)?**
> - **Effort estimate** in engineer-weeks.
> - **Risk** — where this fails.
> - **Sequencing vs the BLAS option** — if both research outputs say
>   "yes, do it," which should we do first?
>
> ## Deliverables
>
> For every section:
> 1. Concrete numbers, not vague assessments. Cite sources (papers,
>    benchmarks, GitHub repos with file paths and line numbers).
> 2. Code examples or pseudocode for any non-obvious kernel choice.
> 3. Honest maturity/risk assessment for every external dependency.
> 4. A final ranked recommendation with tradeoffs explicit, not hedged.
