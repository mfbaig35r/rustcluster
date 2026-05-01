# Blog post outline (v2): rustcluster v1.2 retrospective

**Working title (pick one):**
- *The Wall: Eleven Weeks Against Forty Years of GEMM*
- *Beating BLAS: Notes from a Pure-Rust SIMD Rewrite That Didn't Win*
- *Goto's Long Shadow: Why I Couldn't Beat Faer in Eleven Weeks*

**Voice target:** the Backward Index post — narrative-led, scene-based, two interwoven threads (historical + technical), first-person but the writer doesn't get in front of the material.

**Length:** 4000-5500 words, ~10-12 minute read.

---

## Opening scene (no header, or "October 2025: somewhere in Cupertino")

Cold open. Specific moment, specific terminal output. Resist the urge to summarize.

Sketch:

> The third time the benchmark printed `0.39x of baseline` I closed the laptop and sat with it.
>
> I'd just spent two days writing a "fused similarity-graph kernel" — a re-organization of how rustcluster computes pairwise similarities, designed to skip a step that the v1.1 code spent half its time on. It was supposed to be the architectural lever for v1.2: the thing that would let pure-Rust SIMD beat FAISS at d=1536. The plan budgeted seven to eleven engineer-weeks against this premise.
>
> My fused kernel ran the workload in 1736 ms. The unfused baseline ran it in 672 ms. I'd made the code 2.6× *slower*.
>
> I tried larger sub-tile sizes. 1173 ms. 906 ms. 796 ms. 742 ms. The best I could manage was 9% slower than the thing I was supposed to replace. Every variant of the architectural lever I'd budgeted eleven weeks around was, in fact, slower.
>
> The right thing, at that moment, would have been to stop. The plan had explicit decision criteria for exactly this situation. Step 0b — the spike I was running — was supposed to be a falsifiable test of the central premise, and the test had returned a negative result.
>
> I kept building.
>
> What follows is what I learned in the nine weeks after that.

---

## Section 1: What I was trying to do (~400 words, no historical thread yet)

Set up the technical context. What rustcluster.index is, the v1.1 baseline, the v1.2 ambition.

Beats:
- rustcluster.index ships flat exact vector search + similarity_graph for embedding workloads. Real users running it on text-embedding-3-small (1536-dim) and similar.
- v1.1 was built on faer — a pure-Rust GEMM library that uses `pulp` internally for SIMD dispatch.
- The state of play at d=1536 (Apple Silicon M-series): `similarity_graph` at 0.77× FAISS. `search` at 0.32× FAISS. `range_search` at 0.37× FAISS. Sweet spot was d=128, where rustcluster was *2.3× faster* than FAISS. The d=1536 case was the gap to close.
- The v1.2 plan: replace the faer hot path with hand-written pure-Rust SIMD kernels using `pulp`. Two architectural levers: a 4×4 register-blocked microkernel, and a fused threshold-emit kernel that never materializes the score tile. Beat FAISS at d=1536 by closing both gaps.
- The plan included a falsifiable spike protocol: two mandatory measurements before committing to milestone work.

End the section with a clean transition to the historical thread.

> What I didn't appreciate, when I wrote that plan, was that I was joining a long line of people who had assumed the same thing.

Cite: `docs/v1.2-plan-pure-rust-simd.md`, `docs/v1.2-execution-plan.md`.

---

## Section 2: A short history of trying to beat BLAS (~600 words, all historical thread)

This is the load-bearing section. Establish that GEMM has a forty-year tradition of humbling its replacers. The reader should leave this section with a healthy fear of mature numerical code.

Beats (in roughly chronological order, weave for narrative):

- **1979.** Lawson, Hanson, Kincaid, and Krogh publish the BLAS Level 1 specification — a small set of vector operations: dot product, axpy, vector scaling. The intent: a portable interface so high-level numerical software could be written without committing to a specific hardware vendor. Vendors implement BLAS for their machines; everyone else writes against the interface. By 1988, BLAS Levels 2 (matrix-vector) and 3 (matrix-matrix) are standardized. GEMM enters the canon.

- **The 1990s consensus.** By the mid-90s, attempting to roll your own matrix multiply outside of BLAS is widely considered a mistake. Vendors ship hand-tuned assembly. The lesson is folkloric: *don't.*

- **2002, the postdoc.** Kazushige Goto, working at the Texas Advanced Computing Center, hand-tunes assembly for the Pentium III's cache hierarchy. He's not a numerical analyst by training. His approach is experimental: pick a machine, profile it, write the assembly, measure. GotoBLAS becomes one of the fastest BLAS implementations of its era — competitive with Intel's own MKL, which is staffed by full-time engineers paid to do nothing else. Goto's edge: he understood the cache hierarchy as a primitive, not an afterthought.

- **2014, formalization.** A team at UT Austin publishes BLIS — *BLAS-like Library Instantiation Software* — which abstracts Goto's hand-tuned-per-machine pattern into a portable framework. The "five loops around a microkernel" structure (block over rows of A, block over rows of B, block over k, block over rows of A again, then the microkernel) becomes the canonical cache-blocking pattern. To this day, every fast GEMM you've used follows it.

- **2020s, the Rust angle.** OpenBLAS forks GotoBLAS in 2011 and continues development. Intel's MKL closes-source-then-open-sources. Apple's Accelerate ships with macOS, hiding AMX behind sgemm. Faer arrives — a pure-Rust GEMM library by Sarah Quinones, building on a SIMD abstraction crate called pulp. Faer reaches 70-80% of OpenBLAS performance on most shapes. The crowning achievement of pure-Rust numerical computing in 2026.

- **The pattern across all five generations.** Every implementation that joined the wall took someone years to build, and the people who built them were the kind of people for whom *cache hierarchy* was a primitive. Not an afterthought. Not something you read about in a paper and apply. A primitive.

End the section with the observation that animates the rest of the post:

> When I started v1.2, I was twelve weeks against forty years.

Cite: BLAS docs, faer's repo, "Anatomy of High-Performance Matrix Multiplication" (Goto & van de Geijn 2008), the BLIS paper (Van Zee & van de Geijn 2014).

---

## Section 3: The discipline of falsifying first (~400 words, methodology — both threads)

Bridge: the plan had Goto's lesson baked in, even if I didn't realize it at the time.

Beats:
- The v1.2 plan budgeted 7-11 engineer-weeks. Before any of those weeks could be spent, two mandatory spikes had to return positive results.
- **Step 0a (1-2 days):** measure pulp dot vs std::arch dot vs faer matmul vs scalar. Hard go/no-go thresholds: within 15% of std::arch → proceed; >30% behind → fall back to raw intrinsics (a 20-week plan instead of 11).
- **Step 0b (2-3 days):** wire a *temporary* fused kernel using faer matmul as the inner compute. Bench against the existing path. Question: does the fusion architecture alone close the gap, even with faer doing the math? If yes, microkernel work is bonus. If no, the microkernel work is load-bearing.
- The point of the discipline: an entire 11-week project could be killed in 5 days if either spike returned a clearly-negative result.
- This isn't novel methodology. Every reader who's been in the engineering trenches has seen the inverse — a quarter spent on a premise that wasn't checked. The novelty is doing it.

Sub-point: there's a Tristan Handy line that's been rolling around in my head — *the only way to make sure you're solving the right problem is to keep asking whether you are*. Spike protocols are the operational version of that.

End the section by setting up the next two: "Step 0a was a useful surprise. Step 0b should have been a load-bearing one."

Cite: `docs/v1.2-execution-plan.md` decision matrix.

---

## Section 4: What pulp gives you, and what it doesn't (~600 words, technical thread)

Step 0a result. This section is concrete and code-heavy — uses a table to land the punchline.

Beats:
- Pulp is the SIMD abstraction crate that faer itself uses. The pitch: "write your kernel once using `pulp::WithSimd`, run on AVX2, AVX-512, and NEON via runtime dispatch." If pulp delivered, our microkernel could match `std::arch` performance from a single source.
- The probe: implement an f32 dot product four ways. Scalar reference. Pulp via `WithSimd`. Raw `std::arch` AVX2+FMA (x86) and NEON (aarch64). Faer matmul shaped as 1×d × d×1. Bench at d ∈ {128, 384, 1536} on Apple Silicon.

Drop the table:

| | scalar | pulp (naive) | pulp ×4 | std::arch NEON | faer matmul |
|---|---|---|---|---|---|
| d=128 | 48.60 ns | 11.91 ns | 6.72 ns | 6.73 ns | 6.76 ns |
| d=384 | 119.68 ns | 40.92 ns | 15.88 ns | 15.76 ns | 16.98 ns |
| d=1536 | 673.26 ns | 231.42 ns | 68.54 ns | 66.73 ns | 70.55 ns |

Two findings, in this order:
1. **Naive single-accumulator pulp is 3.47× slower than hand-tuned NEON** at d=1536. FMA latency, not throughput, is the binding constraint when you only have one accumulator. The pitch — "pulp gives you SIMD for free across architectures" — is misleading. Pulp gives you SIMD *primitives*. The kernel-level optimization (multiple independent accumulators to hide FMA latency) is yours to do.
2. **Pulp ×4 (manually unrolled with 4 accumulators) matches NEON within 2.7%.** The single-source-multi-arch property *does* hold, once you take pulp seriously as an abstraction over primitives, not as a magic wand.

Tangent worth including: pulp 0.22 has no plain `add_f32s` for vector-vector add. To combine multiple accumulators at the end of a kernel, you call `reduce_sum_f32s` separately on each and add the scalars. This is a small mistake-cost worth pricing in if you're writing pulp kernels — the abstraction is FMA-shaped, not arithmetic-shaped.

End the section with: pulp validates. Plan A is viable. The microkernel work is real.

> Step 0a took two days and returned a useful surprise. It also gave me a false sense that the rest of the plan would behave the same way.

Cite: `examples/probe_pulp_dot.rs`, commit `f96841c`.

---

## Section 5: The day fusion died (~700 words, both threads, the longest section)

Step 0b result. This is the section where the project's central premise died. The post needs to hold this moment without flinching.

Beats:
- Setup: similarity_graph_ip produces all (i, j) pairs above a threshold. The v1.1 code computes the full (h × w) score tile via faer, then scans it to find threshold passers. The fusion hypothesis: process sub-tile bands and threshold-emit per band, never materializing the full tile. Should be faster because the per-band score buffer fits L1.
- Spike: keep faer as the inner compute, slice the outer tile into row-bands of `sub` rows. Otherwise identical to v1.1. The only variable: whether avoiding the full-tile materialization buys us anything.

Drop the table:

| variant | best (ms) | vs baseline |
|---|---|---|
| baseline (full 384×384 tile materialized) | 672.6 | 1.00× |
| spike sub=8 | 1736.7 | 0.39× (−61%) |
| spike sub=16 | 1173.6 | 0.57× (−43%) |
| spike sub=32 | 906.6 | 0.74× (−26%) |
| spike sub=64 | 796.0 | 0.84× (−16%) |
| spike sub=128 | 742.7 | 0.91× (−9%) |

**Every variant was slower than baseline.** The best — which used the largest sub-tiles I could fit — was 9% slower. The worst was 2.6× slower. The architectural lever, in this configuration, was a brake.

Three reasons, in order of how long it took me to admit each:

1. **Faer GEMM is already cache-blocked internally.** When I shrank the call shape from 384×384 to 64×384, I dropped *below* faer's internal sweet-spot block size. Faer has per-call setup overhead, and I was paying it many more times per outer tile. This is exactly Goto's lesson — cache hierarchy is a primitive — but I was applying it badly, by re-tiling on top of code that had already tiled internally.

2. **The score tile already fits L2 on Apple Silicon.** A 384×384 score tile at f32 is 590 KB. M-series L2 is 12+ MB. The "L1-fitting sub-tile" advantage I'd been imagining was theoretical — the actual cost was dominated by the matmul itself, which faer was already optimizing. The DRAM traffic I'd been planning to save *wasn't there to save*.

3. **True fusion means the threshold check in the FMA loop, not the loop above it.** You can't get the architectural win by re-tiling GEMM calls externally. The win lives inside the register-blocked accumulator loop — compute, compare, emit, all in registers, no intermediate buffer. Faer doesn't expose that level of control. To do it, I'd have to write the GEMM myself.

The honest framing — and this is the post's hardest moment to write — is that Step 0b *should* have updated the plan's expected gains, not just the path. The plan's Plan C ("fused kernel with faer inner") was correctly ruled out. But the implication of the result was bigger: if the score tile already fits L2, the fusion lever doesn't pay off even with a perfect microkernel. The dominant cost on this workload is microkernel speed. And as Step 0a had just shown, our microkernel best case was matching std::arch NEON, not exceeding it. Which is exactly where faer already lived.

I had two negative spikes and a plan that no longer made sense.

I built it anyway.

---

## Section 6: Goto's lesson, learned twice (~500 words, both threads)

Bridge into the build phase. Frame what we built around what Goto would have recognized.

Beats:
- Built the 4×4 register-blocked microkernel anyway. Got a result that, in isolation, looked good:

| tile | rustcluster_simd | faer | ratio |
|---|---|---|---|
| 96 | 0.23 ms | 0.58 ms | **0.39× (2.5× faster than faer)** |
| 256 | 2.46 ms | 2.29 ms | 1.07× |
| 384 | 4.00 ms | 3.85 ms | 1.04× |

- The microkernel beats faer at small tile sizes — faer's per-call setup overhead dominates there, and our lean code wins. At large tiles, we tie. This is the kind of result that would have been the post's headline if I'd stopped there. *Look, pure Rust matches faer at the shape that matters.*
- The shape that matters wasn't the only shape. The benchmark above used **square** tile shapes — h=w=tile, both ranging from 96 to 384. Reasonable for similarity_graph, where outer tiles are square. But search uses `q_h ≈ 32`, `x_w = 20000` — extremely rectangular.
- Wired into ip_batch (M5 in the plan), my 4×4 register-blocked microkernel ran search at **142 ms vs faer's 69 ms**. *Two times slower.*
- This is the section where I had to learn, then re-learn, what cache hierarchy means. Naive (i, j) iteration over the 4×4 microkernel re-reads all of x once per i_block. For x_w = 20000 at d = 1536, that's 117 MB of x re-read q_h/4 times. Gigabytes of redundant DRAM traffic, none of which the inner microkernel could fix.
- Fix: outer X-cache-block loop. Load `CACHE_X_BLOCK` rows of x into L2 once, iterate all of Q against them, repeat. Goto's pattern, ported to my microkernel.
- Cache block size sweep:

| block | search (ms) | range (ms) |
|---|---|---|
| 16 | 81 | 87 |
| 32 | 86 | 84 |
| 64 | **81** | **81** |
| 128 | 81 | 79 |
| 256 | 84 | 80 |
| 1024 | 107 | 108 |

- Sweet spot at 64 rows × 1536 × 4 = 384 KB. Larger blocks suffer L2 contention with sibling threads on the parallel path; smaller blocks lose Q-row reuse.
- The 2× regression closed to 17%.

End the section with: this was the cache-hierarchy primitive Goto had built into BLIS in 2014. I had to relearn it experimentally in 2026 because I hadn't internalized it as a primitive. *Mature GEMM is a wall because every generation has had to rediscover the same primitives.*

Cite: M4 commit `ede1317`, M5 commit `b38a566`, cache-blocking commit `088e073`.

---

## Section 7: The numbers that ended it (~500 words, technical)

The honest end-state, no spin.

Beats:
- After everything — microkernel, fused kernel, cache blocking, parameter sweeps — the end-to-end picture at n=20k, d=1536 on Apple Silicon:

| Operation | v1.2 fused | v1.1 (faer) | Delta |
|---|---|---|---|
| `similarity_graph` | 714 ms | 656 ms | 1.09× slower |
| `search` top-10 | 80 ms | 69 ms | 1.16× slower |
| `range_search` 0.5 | 77 ms | 64 ms | 1.20× slower |

- The plan target was *≥1.0× FAISS at d=1536*. v1.1 was already 0.77× FAISS. v1.2 came in slower than v1.1. The target was not just missed; the gap got worse.
- Three things to say honestly here:
  1. The 17-20% gap appears inherent to kernel maturity. Faer has years of tuning baked in — software prefetching, packed kernels, sophisticated parallelism strategies. Closing the last 17% from where I sat would have needed substantially deeper expertise (BLIS-style packing, prefetch patterns I haven't profiled). It would also have been work I couldn't budget — call it 1-2 weeks of expert-grade kernel engineering with uncertain payoff. Goto did it over years on a single architecture.
  2. The fused architecture didn't win because, at d=1536, the score tile already fits L2. The memory bandwidth I planned to save *wasn't being spent.* Step 0b had told me this. It took until M7 for me to *believe* it.
  3. The benchmark itself was the worst case for fusion. Threshold = 0.5 on random unit-norm vectors at d=1536 produces approximately zero edges. The v1.1 scan over the materialized score tile was already cheap. Fusion's promise — saving the cost of writing-then-reading the tile — wasn't being purchased on the workload I was measuring.

The honest line:

> I had built a kernel that was correct, well-tested, and 17-20% slower than the thing it was supposed to replace. On the benchmark that was supposed to be the win.

Cite: revert commit `8daaf82`, `docs/v1.2-prototype-result.md` outcome section.

---

## Section 8: The revert, and what the simd library is for (~400 words)

Why we kept the library and removed the wiring.

Beats:
- Three options at the decision point: ship the regression, keep optimizing, revert.
- Why revert: a 17-20% regression isn't catastrophic, but it's a real cost paid by every user on every workload. The library passes 26 tests including proptest parity checks. It's correct. It's just not faster yet.
- What stays: `rustcluster-simd` ships with `dot_f32`, `l2sq_f32`, `batched_dot_tile` (with cache blocking), `fused_threshold_emit_tile_ip/_l2`. As a workspace member, dev-dep only. Unused by the production hot path.
- What's there for next time: when someone — maybe me, maybe a contributor, maybe the next generation of pulp — closes the kernel-quality gap, the wiring points are documented and the regression-avoidance strategy is on file. The work isn't lost; it's a draft against a wall that future drafts will have to scale anyway.
- Frame it generously: "The simd library is what it is — a foundation. The faer hot path is what it is — mature." Both are right.

---

## Section 9: What I learned (~500 words, both threads)

The lessons section. This is the section the post is read for. Make every bullet earn its line.

Beats:

1. **Spike before you commit.** The 5-day Step 0a + 0b protocol almost killed v1.2 on day 5. It didn't, because I didn't take Step 0b's signal seriously enough. *That's the failure mode, not the protocol.* Build the protocol; respect what it tells you.

2. **Pulp gives you SIMD primitives, not optimized kernels.** The "single source, multi-arch" property is real, but the unrolling discipline is the same as if you wrote raw intrinsics. Plan accordingly. (For bonus points: this is true of every "high-level SIMD abstraction" I've seen — `wide`, `std::simd`, `safe_arch`. The abstraction handles dispatch, not optimization.)

3. **Square benchmarks lie about rectangular shapes.** My M4 bench used h = w = tile, where I matched faer. Search is q_h = 32, x_w = 20000. The 2× regression was invisible at the shape I'd benchmarked. *Always bench the shape your production workload runs.*

4. **Cache hierarchy is a primitive, not an optimization.** Goto's 2002 hand-tuned BLAS was about cache hierarchy. BLIS's 2014 framework was about cache hierarchy. My 2026 cache-blocking sweep was about cache hierarchy. Every generation that joins the wall has to internalize this primitive *before* writing the inner kernel, not after. I learned it after.

5. **Mature GEMM is a wall.** Forty years. Five generations. Each one took years. Beating it is not a quarter-of-engineering effort. Respect the baseline.

6. **Negative results are publishable, useful, and rare.** Most performance posts ship wins. The work of falsifying a hypothesis with discipline, measuring rigorously, and reverting honestly *is* the contribution. A library that ships correct, tested, foundation code without claiming a perf win is more honest than one that ships a 17% regression and calls it v1.2.

End by tying the historical thread to the technical one:

> Every name on the wall — Lawson, Hanson, Goto, the BLIS team, Quinones — is someone who spent years on one library. I spent eleven weeks. The wall is fine. I'll come back to it when I have years.

---

## Section 10: Thanks and notes (~150 words)

Generous close.

Beats:
- Thanks to Sarah Quinones (faer) for building the baseline that makes pure-Rust numerical computing what it is in 2026, and for choosing pulp as the SIMD abstraction (which gave my v1.2 work a chance at all).
- Thanks to the pulp authors for the abstraction — used correctly, it does what it says.
- Thanks to FAISS — somebody has to be the goalpost, and they've been a generous one.
- Thanks to Kazushige Goto, posthumously — his 2002 hand-tuning is in every fast GEMM I've used, and the lesson is in every cache-blocking sweep I'll ever do.
- Code: `github.com/mfbaig35r/rustcluster`, branch `feat/index-pulp` for the v1.2 work. The simd library is `rustcluster-simd/`. PRs welcome — the wall is patient.

---

## Drafting plan (1 week)

- **Day 1**: Opening scene + Section 1 (setup) + Section 2 (BLAS history). The historical thread is load-bearing — get it right first.
- **Day 2**: Section 3 (methodology) + Section 4 (Step 0a). Easier — concrete, table-driven.
- **Day 3**: Section 5 (the day fusion died). The hardest section. Don't flinch; don't oversell.
- **Day 4**: Section 6 (Goto's lesson, learned twice) + Section 7 (the numbers that ended it). Technical climax.
- **Day 5**: Section 8 (revert) + Section 9 (lessons). Tighten the lessons until each one earns its line.
- **Day 6**: Section 10 (thanks). Read-through. Verify every number against the actual probe output. Verify commit hashes.
- **Day 7**: Polish, fact-check, publish.

## Sourcing checklist

- [ ] Re-run `examples/probe_pulp_dot.rs` from a clean state; confirm Section 4 numbers.
- [ ] Re-run probe_batched_dot to confirm Section 6 numbers.
- [ ] Re-run probe_search (re-create from git history at commit `088e073`) to confirm Section 7 numbers.
- [ ] Verify commit hashes referenced in text (`f96841c`, `36668bc`, `ede1317`, `b38a566`, `088e073`, `8daaf82`).
- [ ] Verify pulp 0.22 method-name claims (`add_f32s` doesn't exist, etc.) against the current pulp version.
- [ ] Verify BLAS history dates (1979 BLAS Level 1, 1988 Level 3, GotoBLAS ~2002, BLIS 2014). One sentence's worth of fact-checking each.
- [ ] FAISS 540 ms reference number — flag as "from the original v1.2 plan prompt" or measure directly (~half day).

## Notes on voice (vs. the v1 outline)

- v1 opened with TL;DR. v2 opens with a scene.
- v1 had section headers like "Step 0a — what pulp gives you, and what it doesn't". v2 has "What pulp gives you, and what it doesn't" (drop the Step 0a; the reader doesn't care about the milestone numbering).
- v1 was chronological by milestone. v2 is structured around the BLAS-history thread, with the technical thread weaving in.
- v1 had "Lessons" as a numbered list. v2 keeps the numbered list but the lessons are tied back to the historical thread (Goto's primitive, the wall, etc.).
- v1 closed with "What's next." v2 closes with thanks. (Tristan Handy doesn't usually end with future work — he ends with a generous frame and a community-oriented call.)
