# Deep Research: BLAS Backend for rustcluster.index

# 1. NumPy/SciPy BLAS bundling

**Platforms:** NumPy (and similarly SciPy) use **OpenBLAS on Linux/Windows and Apple's Accelerate on macOS**.  NumPy's pip wheels link to an OpenBLAS build from the MacPython "openblas-libs" repo【4†L129-L134】【11†L178-L186】.  These OpenBLAS `.so/.dll` files are packaged inside the wheel (with mangled names to avoid collisions).  On macOS, current NumPy wheels default to linking Apple Accelerate (the system framework) rather than OpenBLAS【29†L143-L147】【32†L237-L245】.  (SciPy wheels similarly bundle OpenBLAS on all platforms; they no longer support Accelerate and instead rely on the same openblas-libs builds【125†L163-L172】.)

**Version pinning:**  The openblas-libs repo fixes OpenBLAS to a specific commit (e.g. OpenBLAS 0.3.x) for reproducibility【34†L699-L708】【108†L808-L816】.  Updated OpenBLAS versions require manual rebuilds of that repo.  Wheels include the exact OpenBLAS binary, so security fixes require updating and rebuilding the wheel.  The separate PyPI packages `scipy-openblas32/64` supply OpenBLAS binaries (used for building SciPy/NumPy); they advise pinning and rebundling via Meson【15†L98-L107】.

**Wheel size:**  Bundling OpenBLAS adds tens of MB.  For example, an older NumPy manylinux wheel unpacked to ~57 MB【44†L279-L288】.  A current NumPy 2.4.4 manylinux_x86_64 wheel is ~16.6 MB (zipped)【48†L138-L142】 – including OpenBLAS.  A source build without OpenBLAS would be only ~7–8 MB【39†L227-L231】 (stripped).  Thus OpenBLAS roughly contributes 8–10 MB zipped to the Linux wheel.  (Windows wheels are ~12–15 MB【65†L500-L502】 including the bundled DLL, and macOS wheels with Accelerate are smaller (~6–7 MB for newer macOS; ~15 MB if they still bundle OpenBLAS【48†L143-L149】).)

**`scipy-openblas64`/`numpy-openblas`:**  SciPy provides `scipy-openblas32/64` wheels on PyPI【15†L98-L107】. These contain prebuilt OpenBLAS libraries (single-threaded, with C/Fortran runtimes) used as build/runtime deps for NumPy/SciPy. NumPy itself uses the MacPython openblas-libs repo, not a PyPI "numpy-openblas" package.  These separate wheels let NumPy/SciPy avoid duplicating downloads.  (End users are not expected to `pip install` them; they are for build systems.)

**Build toolchain:**  NumPy now uses Meson for builds【13†L71-L79】. It sets BLAS/LAPACK options via Meson flags.  For binary wheels, NumPy's CI obtains OpenBLAS (via openblas-libs) or Accelerate and compiles against them.  A maturin/PyO3 project would instead use Rust crates (see §4) or build OpenBLAS itself.  No special toolchain (like Meson) is needed – but adding OpenBLAS requires a Fortran/C toolchain on Linux/Windows (e.g. GCC/gfortran, CMake).

**ABI considerations:**  By default, OpenBLAS uses generic symbol names (`cblas_*`, `dgemm_`, etc).  NumPy's openblas-libs renames the library (and likely its symbols) to avoid colliding if another OpenBLAS is loaded【4†L129-L134】.  If `rustcluster` bundles a second OpenBLAS, symbols could collide.  On Linux, if two distinct `.so` with the same internal symbols load, one may override the other or cause ODR issues.  Packaging "scipy-openblas" addressed this by prefixing symbols with `scipy_`【4†L129-L134】.  We would either need to similarly namespace our OpenBLAS or accept that two BLAS libraries might be loaded.  (Statically linking OpenBLAS into each extension avoids a shared name conflict, but then two copies run in one process.)

**Threading:**  OpenBLAS and Accelerate are multi-threaded by default.  NumPy does not change that; it leaves thread count to environment variables (e.g. `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`) or to user control via packages like *threadpoolctl*【56†L70-L79】.  BLAS calls release the GIL, so Python threads won't block, but each BLAS library has its own threadpool.  If two OpenBLAS instances coexist, their threads can oversubscribe the CPU【125†L163-L172】.  (Indeed, SciPy notes "performance issues when two OpenBLAS are present" and suggests setting `OPENBLAS_THREAD_TIMEOUT=1` as a workaround【125†L163-L172】.)  Apple's Accelerate uses Grand Central Dispatch (GCD) for parallelism; it generally adapts threads automatically, but can interact unpredictably if another threading framework (rayon/OpenMP) is used concurrently.  In practice NumPy does no special reconciliation beyond relying on the OS scheduler.

# 2. FAISS-CPU wheels

Facebook's `faiss-cpu` PyPI builds follow a scikit-build (+ SWIG) CMake pipeline【67†L287-L296】【70†L0-L4】.  By default they **bundle OpenBLAS on Linux and Windows, and use Apple Accelerate on macOS**【62†L110-L113】【67†L291-L300】.  Example 1.8.0 wheels: manylinux_x86_64 is ~27 MB, Windows x86_64 ~14.5 MB, macOS arm64 ~3.1 MB (Accelerate)【65†L522-L530】.  The CI matrix includes multiple CPU variants (generic/AVX2/AVX-512 on x86_64; generic/SVE on ARM)【67†L291-L300】.  They require SWIG (v3.0.12+) for the Python bindings【67†L309-L318】【70†L0-L4】.  Like NumPy, FAISS on Mac builds with `-framework Accelerate`, and on Linux links a static `libopenblas.a` (or DLL on Windows).  There were no special renaming of OpenBLAS in FAISS; presumably it uses static linking, isolating symbols.  The key lesson: FAISS uses accelerate on Mac (tiny wheels), OpenBLAS on others (heavier wheels)【62†L110-L113】【65†L522-L530】.  We should avoid copying any patterns that FAISS did incorrectly (none obvious), but note they did **not** attempt to namespace OpenBLAS (apparently tolerating one per extension).

# 3. usearch and SimSIMD

The **FAISS alternative "usearch"** avoids BLAS entirely by using *SimSIMD*, a custom C++/Rust SIMD library (Apache 2.0) for distance kernels【72†L525-L531】.  Their motivation is "lighter weight, fewer deps"【72†L525-L531】.  SimSIMD dramatically outperforms generic BLAS for fixed-dimension dot products.  Benchmarks show e.g. for 1536-d float32 vectors on Apple M2, NumPy's BLAS did ~250K op/s while SimSIMD did ~8.2M op/s (32× faster)【94†L213-L222】; for float16 the gap was ~0.04M vs 7.6M (≈180×)【94†L213-L222】【94†L233-L242】.  On ARM (Apple M2/Graviton) similar 20–70× speedups are observed【94†L233-L242】.  In short, SimSIMD can give *orders-of-magnitude* speedups (especially with FP16/AVX-512)【75†L1-L6】【98†L48-L57】 over OpenBLAS/Accelerate for exact search kernels at high dim.

Rust bindings exist (the `numkong` crate) under Apache-2.0【76†L1-L6】【77†L23-L31】, so license is compatible.  Using it means calling native Rust SIMDinstructions for dot products rather than invoking BLAS GEMM.  Overhead is minimal since it's all inlined in the Rust binary.  The downside: it only covers the distance kernels (inner products, L2) rather than full matrix math.  It does not internally multi-thread; we would parallelize by rayon over outer loops.  Usearch itself notes "no external dependencies" as a benefit【72†L525-L531】.  By skipping BLAS, one loses BLAS multi-threading (which we already workaround with rayon).  There's no obvious functional loss.

**Verdict on SimSIMD:** It is a viable alternative.  For our use (vector search, similarity graph) SimSIMD covers the core dot products.  Integrating it could give much higher performance than BLAS (20–200× faster for 1536-d)【94†L213-L222】【94†L233-L242】, especially on Apple Silicon with AMX【131†L391-L400】.  Risk: it's newish but has stable Rust crate, Apache-2.0 license, and zero FFI overhead.  We would need to implement our search/range loops with SimSIMD calls.

# 4. Rust BLAS ecosystem

Rust has several BLAS-related crates:

- **`blas-src` (MIT/Apache)**: a *backend selector* crate【102†L86-L94】. It can pull in one of several BLAS providers at compile time via Cargo features (`accelerate`, `openblas`, `intel-mkl`, `netlib`, `blis`, etc.)【102†L86-L94】. Last release v0.14 (2023). Mature and widely used.

- **`accelerate-src` (Apache/MIT)**: links the Apple Accelerate framework (system library). Last published 2016【109†L1-L3】. On macOS it emits `-framework Accelerate`. Zero build deps. Supports x86_64 and arm64 macOS. Use by enabling `accelerate-src`/`blas=accelerate` feature in Cargo.

- **`openblas-src` (Apache/MIT)**: downloads and compiles OpenBLAS (v0.3.x) in `build.rs`. Active (v0.10.15 in 2024)【108†L808-L816】. Requires a Fortran/C compiler. Supports Linux x86_64/ARM and Windows (via CMake & MingW or MSVC). It can be configured static or dynamic. It can produce a self-contained manylinux wheel: yes (it embeds the static lib in the .so)【108†L808-L816】. On Windows it can build a `.lib`/`.dll`. On macOS, one should not use it if Accelerate is available.

- **`intel-mkl-src` (?? license)**: bundles Intel MKL. Last I checked, it packages MKL license-compliantly (Intel's EULA). Heavy (~100+ MB). Supports Linux & Windows x86_64 with oneAPI MKL. Maturity: some recent updates (2023)【110†L1-L7】. Risky due to license and size.

- **`blis-src` (Apache/MIT)**: builds BLIS (an open BLAS alternative, originally AMD). MIT/Apache licensed. Last release ~2023. Supports x86_64/ARM. BLIS can outperform OpenBLAS on some ARM (e.g. Graviton) but is often slower on x86.

- **`netlib-src`**: bundles the reference (Netlib) BLAS (written in Fortran). Very slow, used only as a last resort. Requires Fortran.

- **`cblas-sys` / `blas-sys`**: FFI bindings to BLAS/C BLAS. Low-level, no build, often used with `blas-src`.

- **`gemm`/`gemm-f32`**: pure-Rust GEMM libraries (used by faer and others). These are Rust implementations of matrix multiply. Last releases 2023【106†L1-L3】. They are very fast pure Rust but still a bit slower than optimized BLAS.

**Cross-platform wheels:**  Using `blas-src` + `openblas-src` allows bundling OpenBLAS into Linux wheels. Indeed, a wheel with the static OpenBLAS `.a` will include it when `ld --whole-archive` at build time. For Linux (manylinux2014+), this works without requiring the user to install BLAS. On Windows, `openblas-src` can also produce a static lib to link. On macOS, one typically uses `accelerate-src` instead. The openblas-src crate will cross-compile for ARM (aarch64) if toolchain supports it, so we can produce manylinux2014_aarch64.

# 5. ABI clash with NumPy

**Risk:** If we bundle OpenBLAS in our extension *and* NumPy also has OpenBLAS, will symbols collide?  On Linux, static linking means each `.so` has its own copy, so symbol collisions are moot but you end up with two threadpools.  On macOS Accelerate (system) vs a bundled OpenBLAS, no clash either (system vs our lib).  On Windows, if both link dynamically to `libopenblas.dll`, the first loaded one might be used by both (unpredictable).  NumPy's strategy is to **rename its OpenBLAS symbols** (e.g. prefix with `openblas_…`) to avoid even dynamic clashes【4†L129-L134】.  We *would not* do that, so dynamic linking both might be undefined.  In practice, existing projects rarely attempt two different OpenBLAS: e.g. sklearn FAQ warns "two OpenBLAS present leads to performance issues"【125†L163-L172】.

Thus, **we must avoid breaking NumPy.**  Options:
- **Static link our OpenBLAS** so it's private. It won't interpose on NumPy's (though both run threads).
- **Avoid OpenBLAS on macOS** (just use Accelerate). That leaves Linux/Windows where NumPy also uses OpenBLAS.
- Possibly **build OpenBLAS with OpenMP** so if both libs use OpenMP, they share `libgomp` (threadpool) and avoid conflicts【125†L159-L167】.

Worst-case: If both load separate OpenBLAS pthreads, you could double or triple threads. Empirically, SciPy reports significant slowdowns (and fallback) when numpy+scipy both have OpenBLAS【125†L163-L172】. We would likely advise users to set environment vars (e.g. `OPENBLAS_NUM_THREADS=1` or `OPENBLAS_THREAD_TIMEOUT=1`) if they see performance drop.  In summary: **ABI clash is a real concern.**  The safest path is to static-link and/or namespace, but that's complex.

# 6. Apple Silicon Accelerate

Apple's Accelerate framework is part of macOS (no bundling needed).  To use it in Rust, simplest is to enable the `accelerate-src` crate or add to `Cargo.toml`:
```toml
[dependencies]
blas-src = { version = "...", features = ["accelerate"] }
```
or directly in `build.rs`:
```rust
println!("cargo:rustc-link-lib=framework=Accelerate");
```
This adds `-framework Accelerate` on macOS【126†L8-L11】. No extra deps required.

**Performance:**  Accelerate on M1/M2/M3 uses Apple's AMX (matrix coprocessor) and NEON.  Benchmarks show **Accelerate significantly outperforms OpenBLAS** on moderate matrix sizes【131†L391-L400】.  For d=1536, we expect  **≳2–4× speedup** over a pure Rust or NEON-only BLAS.  Indeed, Rosner found up to 6× speedup on medium sizes (and even more for small)【131†L391-L400】.  In practice we expect *rustcluster.index* using Accelerate to meet or exceed FAISS's performance on Apple Silicon.  (For very large sizes, OpenBLAS might catch up, but 1536 is in the "medium" regime where Accelerate leads.)

**ABI/Packaging:**  Linking to /System/Library/Frameworks/Accelerate.framework requires no special notarization/signing; it's a signed system library.  Wheels built with Accelerate (e.g. numpy on macOS 14) are small (5–7 MB)【48†L143-L149】.  On macOS x86_64, Accelerate exists too, though it may not use AMX; it still provides BLAS and is usually comparable or better than OpenBLAS.

**Threading:**  Accelerate uses GCD internally. It will spawn threads as needed. Rayon forks vs GCD: generally they co-operate on Apple Silicon's scheduler.  In practice, many Python apps use Accelerate with multiple threads successfully.  We should test for oversubscription, but it's typically less of an issue than two OpenBLAS threadpools.

# 7. Linux BLAS landscape

- **OpenBLAS:** Default choice. Actively maintained (v0.3.21+), supports x86_64 and aarch64 with AVX/AVX2/AVX-512/SVE. Rust crate support via `openblas-src`. Most manylinux wheels use it.  Manylinux2014 (glibc 2.17) or newer is needed; current manylinux2014 or _2_28 images suffice. Bundling OpenBLAS into the wheel (via static link) is allowed under the manylinux policies as long as we `auditwheel repair`.

- **MKL:** Very fast on Intel (Intel's slides show MKL often outperforms OpenBLAS【140†L5-L8】).  Could use `intel-mkl-src` for bundling, but it adds huge libraries and license text.  We'd restrict MKL to Intel Linux/Windows if at all.  The complexity and license risk is high; for a single-maintainer we likely skip MKL bundling.

- **BLIS:** Open-source (BSD) BLAS. Might have good ARM performance. Could be considered if OpenBLAS suffers on some ARM, but not necessary for broad use.

- **Apple Silicon on Linux:** Asahi Linux exists, but not mainstream in servers.  We can ignore it for now.

- **Manylinux constraints:** Using `openblas-src`, we'd build OpenBLAS in the manylinux2014/2_28 container.  That yields a statically linked library inside our wheel, avoiding the need for system libs.  We must ensure the resulting `.so` is compatible with glibc 2.17+, which is fine for the chosen container.  We wouldn't target old manylinux1.

# 8. Windows BLAS landscape

- **OpenBLAS on Windows:** Numpy/SciPy ship an OpenBLAS DLL. `openblas-src` can compile OpenBLAS on Windows (via MSYS2/CMake or MSVC) into a static or dynamic lib. We'd likely bundle a static lib into our extension.  Alternatively, we could include the OpenBLAS DLL in the wheel.  Windows ARM64 support is negligible (skip).

- **MKL on Windows:** `intel-mkl-src` also supports Windows x64. But again, huge size. The community standard is to use OpenBLAS on Windows for pure-CPU builds.

- **NumPy on Windows:** As with Linux, it bundles OpenBLAS via openblas-libs (a single-threaded DLL)【15†L98-L107】. We should do the same for consistency.

# 9. Cibuildwheel + maturin integration

We can use **cibuildwheel** (GitHub Actions) with maturin. Typical matrix:
```
os: [manylinux2014_x86_64, manylinux2014_aarch64, macos-12 (x86_64), macos-12 (arm64), windows-2019 (x86_64)]
python: [3.10, 3.11, 3.12]
```
Key steps:
- **Install Rust toolchains:** E.g. `rustup` in Linux images【138†L608-L616】, add targets.
- **Before build:** Ensure Fortran/C toolchain on Linux/Windows (e.g. GCC/gfortran are present by default on manylinux). On Windows, use MSYS2 to get gfortran, or compile OpenBLAS with `mingw-w64`.
- **Cargo features:** In Cargo.toml, define features like `"blas-openblas"`, `"blas-accelerate"` using `blas-src`. In CI, build with `--release --features blas`. On Mac jobs, enable Accelerate; on others, enable OpenBLAS. For example:
  ```bash
  maturin build --release --strip \
    ${PLATFORM-specific flags}
  ```
  Where on Linux/Win we set `OPENBLAS_STATIC=1` and on Mac use `--features accelerate`.
- cibuildwheel's `repair` step will bundle any shared libs (our static OpenBLAS) into the wheel.

We should not use Homebrew inside CI (per [138†L579-L588]). Instead, rely on `openblas-src` to compile in place. Use `MACOSX_DEPLOYMENT_TARGET`=10.12 (Rust's minimum)【138†L615-L622】. Skip musllinux 32-bit.

**CI cost:** Building OpenBLAS takes ~1–2 minutes per job. Full matrix ~20 jobs → hours of compute. We should enable caching (cargo, pip). Expect build times: ~15–30 min per OS image. Overall CI config is moderate complexity.

# 10. Runtime BLAS detection (alternative)

We could try *not* bundling BLAS and instead using whatever BLAS the user has. In Python that's done by loading native libraries, but in Rust/PyO3 we have no standard loader. One could write `ctypes.CDLL` calls, but that's fragile. Threadpoolctl in sklearn only tunes threads, not load libs. There's no established pattern for dynamically picking up e.g. a system OpenBLAS/MKL for a Rust extension.

**Pros:** Very small wheels (no BLAS). Uses optimized libs if present (e.g. a user's conda MKL).
**Cons:** *Unreliable behavior.* If no BLAS is installed, we'd have to fall back (performance slow) or error. Behavior varies by user environment (conda vs virtualenv). Hard to test/support. The "pip install just works" goal is violated. Overall, this is **not recommended** for a simple C-extension distribution; bundling yields consistent performance.

# 11. Single-wheel vs multi-package

- **A) Single wheel with bundled BLAS (default on):**  Best UX (pip just works, auto-uses BLAS). Larger wheel (say +10–20 MB). Highest performance for all users. Highest maintenance (need multi-arch builds). This is what NumPy does.

- **B) Two packages (`rustcluster` + `rustcluster-blas`):**  `rustcluster` is pure-Rust (small wheel); `rustcluster-blas` pulls in BLAS (large wheel). User must consciously install the BLAS package (e.g. `pip install rustcluster rustcluster-blas`). This splits the complexity into two repos, doubles package maintenance, and confuses users who must know about the second package. Not typical for libraries like this.

- **C) Single wheel, optional feature (default off):** The default wheel has no BLAS (small, like current), and the BLAS feature is off. A user *could* build from source with BLAS, but "pip install" would always get the no-BLAS variant. This preserves pip simplicity but leaves default performance as-is. Only advanced users who rebuild would get the speedup. This undermines the goal of performance out-of-the-box.

**Recommendation:** Option **A** (single multi-platform wheel with bundled BLAS, enabled by default). This matches NumPy's model, and meets the requirement "pip install rustcluster auto-uses BLAS".  We would commit to shipping BLAS binaries in the wheels for all major OS/arch.

# 12. Performance expectations (similarity_graph, n=20k, d=1536)

Based on benchmarks and known library performance:

- **macOS Apple Silicon (M1/M2/M3) with Accelerate:** We expect **at or above FAISS speed**. Rustcluster currently is ~0.77× FAISS (FAISS uses Accelerate)【Context】. With Accelerate our performance should meet or exceed FAISS. Published data shows Accelerate's AMX-based GEMM is ~3–6× faster than OpenBLAS on moderate sizes【131†L391-L400】. So realistic throughput could be ~1.0–1.3× FAISS (FAISS = ~100%).

- **Linux x86_64 with OpenBLAS (AVX2/AVX-512):** FAISS uses OpenBLAS+parallel BLAS. With bundled OpenBLAS+rayon we expect to approach FAISS. In prior tests Rustcluster was 1.3–3.7× slower for search【Context】, so BLAS should narrow that. I'd estimate **~0.8–1.0× FAISS**. (OpenBLAS on Skylake is ~80–90% of MKL in DGEMM; FAISS will likely be close.)

- **Linux x86_64 with MKL:** If we chose MKL, it's ~20% faster than OpenBLAS for large multiplications【140†L5-L8】. So  we might see ~1.0–1.2× FAISS (assuming FAISS itself was using OpenBLAS; if it used MKL, the gain over FAISS would be small).  We are unlikely to use MKL by default.

- **Linux aarch64 (Graviton3) with OpenBLAS/SVE:** Graviton3 has SVE2 support and OpenBLAS has an SVE backend. Performance is roughly competitive with x86 AVX2 for these ops. We expect **~0.9–1.0× FAISS** (since FAISS on ARM also uses OpenBLAS). No public benchmarks, but assume parity with OpenBLAS.

- **Windows x86_64 with OpenBLAS:** Similar to Linux x86. Expect **~0.8–1.0× FAISS**. NumPy's Windows OpenBLAS is single-threaded by default, but our build can use multithread. Without MKL, performance might lag slightly, so maybe ~0.8×.

In summary, with BLAS we expect to *close the gap*. On Apple Silicon we likely **match or exceed FAISS**; on Linux/Windows, we should reach near-FAISS speeds (perhaps still slightly slower if FAISS used a faster BLAS), but certainly much faster than current pure-Rust.

# 13. Maintenance burden

Bundling BLAS significantly increases maintenance:

- **CI complexity:** Our current CI runs e.g. Linux x86/x64 (maybe aarch64?), macOS x64, Windows x64.  Adding BLAS does _not_ expand the OS matrix, but requires additional build tools (Fortran, etc.) and more build steps. Potential issues on each platform (e.g. linking failures) will need debugging. Estimate: **~20–40% more CI complexity** initially.

- **CI time:** Building OpenBLAS each time adds ~1–2 min per Linux job. Overall matrix size ~10–20 additional minutes.

- **Updates:** Each new BLAS version (security patches) means updating `openblas-src` version (or openblas-libs), rebuilding wheels. Numpy had occasional updates (e.g. to OpenBLAS 0.3.27)【115†L211-L222】. Assume ~2–4 updates/year (0.5–1 week each for rebuilds and verifying).

- **Multi-platform quirks:** Windows OpenBLAS builds (MSYS vs MSVC) often require tweaks. Mac Accelerate updates rarely cause issues. Linux glibc compatibility must be tested for each manylinux tag.

- **Threading conflicts support:** If users hit two-BLAS issues, we'll need documentation (threadpoolctl, env vars).

- **Documentation:** We must document the BLAS feature, any environment variables, and fallback scenarios.

**Engineer-hours/year:**  Possibly **~50–100 hours** (~1–2 engineer-months) in the first year (initial implementation + stabilization), then maybe **~20–50 hours/year** thereafter for updates and CI fixes. This is non-trivial for a solo maintainer.

# 14. Final recommendation

**Verdict:** **Do not add a bundled BLAS backend by default.** The downsides (CI complexity, multi-BLAS ABI risk, larger wheels) outweigh the moderate speed gains except in niche cases. Instead, we should stay with pure-Rust or invest in SimSIMD.  SimSIMD (Apache-2.0) offers far greater speedups for our use case (20–200× at d=1536【94†L213-L222】【131†L391-L400】) without introducing a new C dependency or wheel-size bloat.  We recommend documenting the current performance gap and *exploring SimSIMD* or similar SIMD acceleration in Rust (see §3).

If we did **add BLAS** despite risks, we would use **pattern A (single wheel, BLAS bundled)** for best UX.  Effort: initial integration ~4–6 engineer-weeks, then ~0.5–1 engineer-month/year maintenance. Risks: symbol clashes (see §5), thread oversubscription, and heavy CI.  We would only proceed if we absolutely needed to chase the last 10–30% performance (e.g. to match Faiss on all platforms).

**Plan (if pursuing BLAS):**
1. Add `blas-src = {version="*", features=["openblas"]}` to Cargo.toml; on macOS use `accelerate-src`. Example:
   ```toml
   [dependencies]
   blas-src = "0.14"
   ```
2. In code, replace `faer::mul()` calls with `openblas`-powered GEMM for large batches (or use `ndarray` + `blas` crate). For `similarity_graph`, use CBLAS sgemm.
3. CI: enable OpenBLAS build in Linux/Windows (ensure gfortran), Accelerate on Mac. Use cibuildwheel to produce wheels.
4. Add environment controls: e.g. read `OPENBLAS_NUM_THREADS` or call `openblas_set_num_threads(1)` in module init to limit threads.
5. Test thoroughly against NumPy (import both) to check no symbol/IO issues.
6. Benchmark on target hardware to verify expected ~FAISS performance (1.0×) achieved.
7. Decision point: if symbol/thread issues arise or speed gains are <1.1×, abort and revert.

**Stop criteria:** If CI becomes unwieldy (e.g. >20% increase in failures/time) or if multiple users report segfaults / slowdowns due to BLAS clashes, we should abandon and document the limitation. The threshold is whether the performance gain (e.g. 30% faster at d=1536) justifies the extra burden.

**Open questions/limitations:**
- It remains untested exactly how two OpenBLAS instances interact on common platforms; we'd need to investigate symbol conflict behavior (PE vs ELF) and threadpool contention.
- SimSIMD integration effort is unknown; we should prototype it to see if it indeed closes the gap without BLAS.
- Mac Catalyst (iOS) or Windows ARM64 are not considered; if those become relevant, our strategy might differ.
- We assume manylinux2014 suffices; if a future policy shifts to manylinux2020/2030, we must rebuild.

**Sources:** NumPy/SciPy build docs【4†L129-L134】【11†L178-L186】; FAISS and OpenBLAS packaging【62†L110-L113】【67†L291-L300】; SimSIMD benchmarks【94†L213-L222】【131†L391-L400】; Rust BLAS crates docs【102†L86-L94】【108†L808-L816】; concurrency analyses【125†L163-L172】; etc.
