# The Wall: Eleven Weeks Against Forty Years of GEMM

*Or: what I learned trying to replace faer with hand-written pure-Rust SIMD, and why the production hot path is still faer.*

---

The third time the benchmark printed `0.39x of baseline` I closed the laptop and sat with it.

I'd just spent two days writing a fused similarity-graph kernel, a re-organization of how rustcluster computes pairwise similarities, designed to skip a step that the v1.1 code spent half its time on. It was supposed to be the architectural lever for v1.2: the thing that would let pure-Rust SIMD beat FAISS at d=1536. The plan budgeted seven to eleven engineer-weeks against this premise.

My fused kernel ran the workload in 1736 ms. The unfused baseline ran it in 672 ms. I'd made the code 2.6× *slower*.

I tried larger sub-tile sizes. 1173 ms. 906 ms. 796 ms. 742 ms. The best I could manage was 9% slower than the thing I was supposed to replace. Every variant of the architectural lever I'd budgeted eleven weeks around was, in fact, slower.

The right thing, at that moment, would have been to stop. The plan had explicit decision criteria for exactly this situation. Step 0b, the spike I was running, was supposed to be a falsifiable test of the central premise, and the test had returned a negative result.

I kept building.

What follows is what I learned in the nine weeks after that.

## What I was trying to do

`rustcluster.index` ships flat exact vector search and a similarity-graph kernel for embedding workloads. People run it on text-embedding-3-small (1536 dimensions), Voyage embeddings, Cohere outputs, anything that produces a dense f32 vector of moderate width. It's pure Rust on top of `faer`, a pure-Rust GEMM library, plus a manual GEMV path for the single-query case. The whole thing is small, fast, and well-tested.

By the time I started v1.2, the state of play at d=1536 on Apple Silicon M-series was a mixed picture. At d=128, the production sweet spot, `similarity_graph` ran *2.3× faster than FAISS*. The pure-Rust path was actually winning. But at d=1536, where modern embedding models live, things flipped. `similarity_graph` came in at 0.77× FAISS. `search` was at 0.32×. `range_search` at 0.37×.

One caveat about those FAISS numbers: they come from the original v1.2 planning benchmark, not the same-day v1.1/v1.2 rerun later in this post. The same-day numbers later are v1.1 faer vs v1.2 fused, head-to-head on one machine. FAISS was the planning goalpost, not the final controlled comparison.

The d=1536 gap was the gap to close.

The v1.2 plan, written before any of this happened, identified two architectural levers. First: a focused 4×4 register-blocked microkernel, hand-written in Rust, might match or beat faer's general-purpose GEMM at the small shapes that dominate similarity-graph workloads. Second: a fused threshold-emit kernel that never materializes the full pairwise score tile, threshold-checking in registers and emitting only edges that pass. FAISS-style search kernels can fuse distance computation with selection or emission logic. Faer can give me a fast matrix product, but it cannot give me hooks inside the accumulator loop. This was supposed to be the lever that beat FAISS itself, not just match it.

The plan estimated seven to eleven engineer-weeks. It also included two mandatory falsifiable spikes before any of those weeks could be spent. I'll get to those in a minute.

What I didn't appreciate, when I wrote the plan, was that I was joining a long line of people who had assumed the same thing.

## A short history of trying to beat BLAS

In 1979, four numerical analysts (Lawson, Hanson, Kincaid, and Krogh) published the first specification for the Basic Linear Algebra Subprograms. BLAS Level 1 was a small set of vector operations: dot product, axpy, scaling. The intent was modest. They wanted a portable interface so that high-level numerical software could be written without committing to a specific machine. Vendors would implement BLAS for their hardware. Everyone else would write against the interface and never have to look at the assembly underneath.

By 1988, BLAS Level 2 added matrix-vector operations. By 1990, Level 3 added matrix-matrix multiplication. GEMM entered the canon.

What happened after that is the part of the story most people forget. Through the 1990s, attempting to roll your own matrix multiply outside of BLAS quietly became a mistake. Vendors shipped hand-tuned assembly. Open-source implementations like ATLAS auto-tuned themselves to whatever cache hierarchy the build machine had. The folkloric advice, which you can find in performance-engineering forums to this day, was *don't*. Use the vendor BLAS. Move on with your life.

Then in 2002, a postdoc broke the wall.

Kazushige Goto was working at the Texas Advanced Computing Center, mostly on machines nobody else thought were worth tuning for. He hand-wrote assembly for the Pentium III's cache hierarchy. He wasn't a numerical analyst by training; he was an experimentalist. Pick a machine, profile it, write the assembly, measure. GotoBLAS, the library that came out of his work, became one of the fastest BLAS implementations of its era, competitive with Intel's MKL, which was staffed by full-time engineers paid to do nothing else. Goto's edge wasn't novel mathematics. It was treating the cache hierarchy as a primitive, not an afterthought.

Goto wrote the paper that anchored everything that came after. *Anatomy of High-Performance Matrix Multiplication*, published in ACM TOMS in 2008, laid out the structure that's now standard: five loops around an inner microkernel, with each loop blocking for a different level of the cache hierarchy. Outer loops block over rows of the result. Inner loops block over the shared dimension. The microkernel itself is the only part that touches SIMD. Everything around it is cache choreography.

In 2014, the UT Austin FLAME group, with Field Van Zee and Robert van de Geijn central to the work, published BLIS, the *BLAS-like Library Instantiation Software*, which formalized Goto's pattern into a portable framework. Now you didn't have to be Goto to write a fast GEMM. You picked a microkernel, plugged it into the BLIS scaffolding, and the cache-blocking machinery did the rest. To this day, most fast CPU GEMM implementations you've used are variations on this structure.

Then OpenBLAS forked from the GotoBLAS lineage in 2011 and continued development. Intel's MKL remained the proprietary heavyweight, later folded into the oneAPI ecosystem. Apple's Accelerate ships with macOS and, on Apple Silicon, can route matrix operations through Apple's mostly undocumented AMX coprocessor. And in the 2020s, Sarah Quinones built faer in pure Rust, leaning on a SIMD abstraction crate called pulp, and pushed Rust linear algebra into the same performance conversation as native BLAS implementations.

Five generations. Forty-five years. Each generation took someone years to build. The people who built them all shared the same property: cache hierarchy was a primitive in their head, not an afterthought.

When I started v1.2, I was twelve weeks against forty-five years.

## The discipline of falsifying first

The plan, to its credit, knew what it didn't know.

Before any of the seven-to-eleven engineer-weeks could be committed, two spikes had to return positive results.

Step 0a was an f32 dot product, implemented four ways: scalar reference, pulp via the `WithSimd` trait, hand-written `std::arch` NEON intrinsics for Apple Silicon, and faer matmul shaped as a 1×d × d×1 dot. Bench at d ∈ {128, 384, 1536}. The decision criterion: pulp within 15% of std::arch and we proceed; pulp 15-30% behind and we proceed but plan a hot-path variant; pulp more than 30% behind and we fall back to writing raw intrinsics, which is roughly twice the engineering effort.

Step 0b was the fused architecture spike. Wire a *temporary* fused similarity-graph kernel that uses faer matmul as the inner compute, and bench it against the existing path. The question was simpler than 0a: does the fusion architecture itself buy us anything, even if the inner math is faer's? If yes, the microkernel work in milestones 1-4 was bonus. If no, the microkernel work was load-bearing.

The point of the discipline is that an entire eleven-week project could be killed in five days. There's a Tristan Handy line that's been rolling around in my head: the only way to make sure you're solving the right problem is to keep asking whether you are. Spike protocols are the operational form of that question.

Step 0a was a useful surprise.

Step 0b should have been a load-bearing one.

## What pulp gives you, and what it doesn't

Pulp is the SIMD abstraction crate that faer itself uses. The pitch is appealing: write your kernel once using `pulp::WithSimd`, and the same source compiles to AVX2, AVX-512, and NEON, with runtime dispatch picking the best path at startup. If pulp delivered, our microkernel could match `std::arch` performance from a single source. No `#[cfg(target_arch)]` blocks. No triple maintenance burden across architectures.

I implemented an f32 dot product four ways and ran it on Apple Silicon. Here's the table.

| | scalar | pulp (naive) | pulp ×4 | std::arch NEON | faer matmul |
|---|---|---|---|---|---|
| d=128 | 48.60 ns | 11.91 ns | 6.72 ns | 6.73 ns | 6.76 ns |
| d=384 | 119.68 ns | 40.92 ns | 15.88 ns | 15.76 ns | 16.98 ns |
| d=1536 | 673.26 ns | 231.42 ns | **68.54 ns** | **66.73 ns** | 70.55 ns |

Two findings.

First: a naive single-accumulator pulp dot at d=1536 is **3.47× slower** than hand-tuned NEON. Not a small gap. The kernel is FMA-latency-bound: every iteration is waiting for the previous FMA to retire before starting the next, because there's only one accumulator and Apple Silicon's FMA latency is several cycles. The pitch I'd had in my head, *pulp gives you SIMD for free across architectures*, was misleading. Pulp gives you SIMD *primitives*. The kernel-level optimization is yours to do. In this case, four independent accumulators to hide the FMA latency.

Second: pulp ×4, with manually unrolled four-accumulator structure, matches `std::arch` NEON within 2.7%. The single-source-multi-arch property *does* hold, once you take pulp seriously as an abstraction over primitives, not as a magic wand. You write the unrolling discipline the same way you would in raw intrinsics. Pulp dispatches the result to the right architecture.

There's a small footgun worth pricing in if you write pulp kernels: pulp 0.22 has no `add_f32s` for plain vector-vector add. It has FMA. It has horizontal sum. But to combine multiple accumulators at the end of a kernel, you call `reduce_sum_f32s` separately on each and add the scalars. The abstraction is FMA-shaped, not arithmetic-shaped. This is not a complaint. It makes sense for what pulp is for. But it's a thing you'll trip over the first time.

The Step 0a result was clean. Pulp validated. Plan A was viable. The microkernel work was real, but writeable.

It also gave me a false sense that the rest of the plan would behave the same way.

## The day my fusion hypothesis died

The fusion hypothesis was structurally simple. The current similarity_graph_ip code, the v1.1 path, computes the full (h × w) score tile via faer matmul, then iterates the tile to find threshold passers. The score tile gets written to memory, then read back. For a 384×384 tile at f32, that's 590 KB written and 590 KB read.

The fused architecture would never write the full tile. It would compute a sub-tile band of rows, threshold-emit immediately, and discard the band. The per-band score buffer would fit L1. The emitted edges would be sparse (most pairs don't pass threshold). The memory bandwidth saved would be the full-tile write-then-read, replaced with a tiny edge buffer.

A caveat about the workload, before the table. This was a sparse-emission benchmark. Threshold = 0.5 on random unit-norm vectors at d=1536 produces approximately zero edges, so the experiment measured the cost of avoiding tile materialization more than the cost of dense edge emission. I'll come back to this, because it matters.

For Step 0b, I kept faer as the inner compute. Sliced the outer tile into row bands of `sub` rows. Otherwise identical to v1.1. The only variable was whether avoiding the full-tile materialization bought us anything.

I ran it.

| variant | best (ms) | vs baseline |
|---|---|---|
| baseline (full 384×384 tile materialized) | 672.6 | 1.00× |
| spike sub=8 | 1736.7 | 0.39× (−61%) |
| spike sub=16 | 1173.6 | 0.57× (−43%) |
| spike sub=32 | 906.6 | 0.74× (−26%) |
| spike sub=64 | 796.0 | 0.84× (−16%) |
| spike sub=128 | 742.7 | 0.91× (−9%) |

Every variant was slower. The best, which used the largest sub-tiles I could fit, was still 9% slower than the unfused baseline. The architectural lever, in this configuration, was a brake.

Three reasons, in the order I admitted them.

The first I noticed immediately. Faer's GEMM is already cache-blocked internally. When I shrank the call shape from 384×384 to 64×384 or 8×384, I dropped *below* faer's internal sweet-spot block size. Faer has per-call setup overhead, and I was paying it many more times per outer tile. This is exactly Goto's lesson, but I was applying it badly, by re-tiling on top of code that had already tiled internally. The structure I'd added was not a refinement of faer's structure; it was friction against it.

The second I admitted later. The score tile already fits L2 on Apple Silicon. A 384×384 tile at f32 is 590 KB. M-series L2 is 12+ MB shared per cluster. The "L1-fitting sub-tile" advantage I'd imagined was theoretical. The DRAM traffic I had planned to save probably wasn't the dominant cost. The full score tile was large enough to care about, but not large enough to force the kind of memory-system pain I had imagined.

The third I didn't admit until much later. Real fusion means the threshold check inside the FMA loop, not the loop above it. You can't get the architectural win by re-tiling GEMM calls externally. The win lives inside the register-blocked accumulator loop. Compute, compare, emit, all in registers. No intermediate buffer at any level. Faer doesn't expose that level of control. To do real fusion I'd have to write the GEMM myself.

The honest reading of the spike result, which I'm typing now and didn't quite type then, was bigger than the plan's Plan-C-is-ruled-out conclusion. If the score tile already fits L2, the fusion lever doesn't pay off even with a perfect microkernel. The dominant cost was microkernel speed. And as Step 0a had just shown, our best case was matching `std::arch` NEON. Which is exactly where faer already lived.

I had two negative spikes and a plan that no longer made sense.

I built it anyway.

## Goto's lesson, learned twice

The 4×4 register-blocked microkernel came together cleanly. Sixteen SIMD accumulators across the inner loop, four q-vectors and four x-vectors loaded per iteration, sixteen FMAs issued. Half the load-to-FMA ratio of a row-by-row dot. Around a hundred lines of pulp code. Tested via proptest against a scalar reference, with a Wilkinson-style floating-point error bound that tolerates the rounding-order differences between scalar accumulation and multi-accumulator SIMD.

I benched it against faer at three square tile sizes. The result, viewed in isolation, looked like a win.

| tile | rustcluster_simd | faer | ratio |
|---|---|---|---|
| 96 | 0.23 ms | 0.58 ms | **0.39× (2.5× faster than faer)** |
| 256 | 2.46 ms | 2.29 ms | 1.07× |
| 384 | 4.00 ms | 3.85 ms | 1.04× |

The microkernel beats faer at small tile sizes. Faer's per-call setup overhead dominates there, and our lean code wins by 2.5×. At larger tiles we tie, within 4-7%.

This is the kind of result that would have been the post's headline if I'd stopped here. *Look, pure Rust matches faer at the shape that matters.* I wanted to write that post.

The shape that matters wasn't the only shape.

The bench used **square** tile shapes. Both dimensions, h and w, at the same tile size. Reasonable for similarity_graph, where outer tiles are square. The kernel processes a tile-by-tile diagonal of self-similarity. But search is a different shape entirely. A typical search workload has `q_h ≈ 32` queries against `x_w = 20000` database vectors, all at d=1536. That's a deeply rectangular GEMM. Tall and thin, or short and wide, depending on which side you stand on.

I wired the new microkernel into `ip_batch` (the function that backs both search and similarity_graph) and ran the search benchmark.

Search ran in **142 ms**. Faer took **69 ms**.

The microkernel that matched faer on square tiles was *two times slower* than faer on rectangular ones.

The diagnosis took an hour. The naive (i, j) loop over the 4×4 microkernel reads four query rows and four database rows per inner-loop iteration. The query rows are stable across many j iterations, so they end up in cache. But the database rows (all 20,000 of them) get re-read once per i_block. For x_w = 20000 at d = 1536, that's 117 MB of database vectors *re-read* q_h/4 times. Gigabytes of redundant DRAM traffic, none of which the inner microkernel could fix.

This is where I had to learn, then re-learn, what cache hierarchy means.

The fix was an outer cache-blocking loop. Load `CACHE_X_BLOCK` rows of x into L2 once, iterate all of Q against that block, then move to the next block. It's Goto's pattern, ported into my microkernel. He published this in 2008. BLIS formalized it in 2014. I wrote it from scratch in 2026 because I hadn't internalized it as a primitive.

I ran a parameter sweep to pick the cache block size.

| block | search (ms) | range (ms) |
|---|---|---|
| 16 | 81 | 87 |
| 32 | 86 | 84 |
| 48 | 85 | 80 |
| 64 | **81** | **81** |
| 128 | 81 | 79 |
| 256 | 84 | 80 |
| 512 | 92 | 92 |
| 1024 | 107 | 108 |
| 2048 | 128 | 120 |

The sweet spot is 64 rows × 1536 floats × 4 bytes = 384 KB. Larger blocks suffer from L2 contention with sibling threads on the parallel path. Smaller blocks lose the amortization benefit of reusing each X chunk across many Q rows. Either way you push, you give up something.

The 2× search regression closed to 17%.

Mature GEMM is a wall because every generation has had to rediscover the same primitives. Goto did it once, by hand, and he was the first. BLIS encoded it as a framework, so the next generation didn't have to. Faer inherited the framework. I tried to write GEMM without the framework and rediscovered, in a parameter sweep, the same primitive Goto had identified twenty-four years earlier.

There is a kind of humility you only earn this way.

## The numbers that ended it

I wired everything together. Microkernel. Cache blocking. Fused threshold-emit kernel for similarity_graph. All the proptest parity checks. All 359 Python integration tests. Everything passed correctness. The workspace crate had 26 internal tests, each one a falsifiable claim about a specific kernel.

Then I ran the production benchmarks. Same machine. Same workload. The v1.2 fused path, default features, vs v1.1 (the old path, behind a feature flag I'd kept around as an escape hatch).

| Operation | v1.2 fused | v1.1 (faer) | Delta |
|---|---|---|---|
| `similarity_graph` | 714 ms | 656 ms | 1.09× slower |
| `search` top-10 | 80 ms | 69 ms | 1.16× slower |
| `range_search` 0.5 | 77 ms | 64 ms | 1.20× slower |

The plan target was *≥1.0× FAISS at d=1536*. v1.1 was already 0.77× FAISS. v1.2 came in slower than v1.1 across every shape. The target wasn't just missed; the gap *got worse*.

Three things to say honestly here.

The 17-20% gap looked less like one missing trick and more like accumulated kernel maturity. Faer has years of tuning baked into it: software prefetching, packed kernels, sophisticated parallelism strategies. Closing the last 17% from where I sat would have needed substantially deeper expertise. Goto-level work is measured in architectures, years, and obsession. I was not going to casually close that gap in another sprint.

The fused architecture didn't win because, at d=1536, the score tile already fits L2. The memory bandwidth I planned to save *wasn't being spent*. Step 0b had told me this at the beginning. It took until M7, nine weeks later, for me to *believe* it.

The benchmark itself was the worst case for fusion. Threshold = 0.5 on random unit-norm vectors at d=1536 produces approximately zero edges. The v1.1 scan over the materialized score tile was already cheap, just a tight loop of compare-and-skip. Fusion's promise (saving the cost of writing-then-reading the tile) wasn't being purchased on the workload I was measuring. For workloads with denser thresholds, where many edges actually emit, the fusion architecture might still pay off. I didn't measure that. The benchmark I'd promised the plan I'd hit was the one I missed.

I had built a kernel that was correct, well-tested, and 17-20% slower than the thing it was supposed to replace. On the benchmark that was supposed to be the win.

## The revert, and what the simd library is for

Three options at the decision point.

Ship the regression and document it. A 17-20% regression is not catastrophic. Most users wouldn't notice. The library passed every correctness test. The architecture was technically v1.2-ready. I could write a release note that said "v1.2 ships the simd library; performance unchanged for now" and call it shipped.

Keep optimizing. Software prefetching. Larger register tiles. BLIS-style packed kernels. Maybe two weeks of deeper kernel work, maybe four. Outcome uncertain. Risk of investing more without ever clearing the bar.

Revert. Keep the simd library as a foundation. Don't ship a regression on the production hot path.

I picked the third.

The reason is not heroic. A 17-20% regression is a real cost paid by every user on every workload. Not catastrophic, but real. And it wasn't a temporary state on the path to a win. It was, by my best estimate, the *end state* of what eleven engineer-weeks could buy. To clear the bar I'd need a different category of effort.

The library still exists. `rustcluster-simd/` ships in the workspace, with `dot_f32`, `l2sq_f32`, `batched_dot_tile` (with the cache-blocking I learned the hard way), and the fused threshold-emit kernel for both inner-product and L2 metrics. It has 26 internal tests, including proptest parity against scalar reference at every kernel. As a workspace member, dev-dep only. The production hot path doesn't use it.

What it's there for is the next generation. When someone (maybe me, maybe a contributor, maybe the next iteration of pulp) closes the kernel-quality gap, the wiring points are documented and the regression-avoidance strategy is on file. The library is a draft against a wall that future drafts will have to scale anyway. Not a win. A foundation.

## What I learned

**Spike before you commit.** The five-day Step 0a + 0b protocol almost killed v1.2 on day five. It didn't, because I didn't take Step 0b's signal seriously enough. *That's* the failure mode, not the protocol. Build the protocol; respect what it tells you. The hardest discipline isn't running the spike. It's letting the result update your plan.

**Pulp gives you SIMD primitives, not optimized kernels.** The "single source, multi-arch" property is real, but the unrolling discipline is the same as if you wrote raw intrinsics. Plan accordingly. This is true of every "high-level SIMD abstraction" I've seen: `wide`, `std::simd`, `safe_arch`. The abstraction handles dispatch. The optimization is yours.

**Square benchmarks lie about rectangular shapes.** My milestone-4 bench used h = w = tile, where I matched faer cleanly. Search is q_h = 32, x_w = 20000. The 2× regression was invisible at the shape I'd benchmarked. *Always bench the shape your production workload runs.* If you have multiple workloads with different shapes, bench each one. The cost is one afternoon. The cost of skipping it was nine weeks of confidence in a wrong number.

**Cache hierarchy is a primitive, not an optimization.** Goto's 2002 hand-tuned BLAS was about cache hierarchy. BLIS's 2014 framework was about cache hierarchy. My 2026 cache-blocking sweep was about it too. Every generation that joins the wall has to internalize this primitive *before* writing the inner kernel, not after. I learned it after.

**Mature GEMM is a wall.** Forty-five years. Five generations. Each one took years. Beating it is not a quarter of engineering effort. Respect the baseline. If you're going to climb, plan for years.

**Negative results are publishable, useful, and rare.** Most performance posts ship wins. The work of falsifying a hypothesis with discipline, measuring rigorously, and reverting honestly *is* the contribution. A library that ships correct, tested, foundation code without claiming a perf win is more honest than one that ships a 17% regression and calls it v1.2.

Every name on the wall (Lawson, Hanson, Goto, the BLIS team, Quinones) is someone who spent years on one library. I spent eleven weeks. The wall is fine. I'll come back to it when I have years.

## Thanks

To Sarah Quinones, for building faer and choosing pulp as the SIMD abstraction. The baseline I couldn't beat is the one that made the attempt possible at all. Pure-Rust numerical computing in 2026 is what it is because of her.

To the pulp authors, for an abstraction that does what it says when you take it seriously.

To FAISS, for being the goalpost. Somebody has to be.

To Kazushige Goto, whose work is still hiding inside every cache-blocking sweep I will ever run.

The code is at `github.com/mfbaig35r/rustcluster`, branch `feat/index-pulp` for the v1.2 work, and `rustcluster-simd/` for the kernels themselves. The wall is patient.

---

*Endnotes:*

- *The benchmarks throughout were run on Apple Silicon (M-series). Cross-architecture validation on x86 with AVX2 / AVX-512 was scoped out of v1.2; that's part of the future-work list.*
- *Run-to-run variance for whole-workload benchmarks (similarity_graph) was ~5%. Reported numbers are best-of-N with N=4-5 after a warmup run.*
