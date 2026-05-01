//! Phase-1 instrumented profile of similarity_graph_ip at d=1536.
//!
//! Runs the same inner-loop logic as
//! `rustcluster::index::similarity_graph::similarity_graph_ip`, but with
//! per-phase timers so we can see where time actually goes:
//!
//! - `gemm`     — `ip_batch` call (faer matmul + setup + per-tile alloc)
//! - `emit`     — threshold check + edge push into the local Vec
//! - `concat`   — final merge of per-tile EdgeList chunks
//!
//! Run with:
//!
//!     cargo run --release --example profile_similarity_graph --no-default-features
//!
//! Decision criterion (from docs/index-faer-tuning-plan.md): if `gemm`
//! is >85% of the total, the BLAS-vs-faer kernel-quality gap is the
//! bottleneck and there's nothing for us to optimize at the call site.
//! If `gemm` is <70%, the overhead is real and Phase 2 is justified.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use faer::Par;
use rustcluster::index::kernel::ip_batch;

const N: usize = 20_000;
const D: usize = 1536;
const THRESHOLD: f32 = 0.5;
const TILE: usize = 384;
const RUNS: usize = 5;

fn make_data() -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(1536);
    let mut data: Vec<f32> = (0..N * D)
        .map(|_| rng.gen_range(-1.0_f32..1.0_f32))
        .collect();
    for i in 0..N {
        let row = &mut data[i * D..(i + 1) * D];
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    Array2::from_shape_vec((N, D), data).unwrap()
}

fn main() {
    println!("Building data: n={}, d={}, normalized.", N, D);
    let data = make_data();
    println!("Tile size: {}", TILE);

    // Per-phase nanosecond accumulators (atomics so rayon workers can add).
    static GEMM_NS: AtomicU64 = AtomicU64::new(0);
    static EMIT_NS: AtomicU64 = AtomicU64::new(0);

    let mut concat_ns: u64 = 0;
    let mut total_ns: u64 = 0;
    let mut total_edges: usize = 0;

    let pairs: Vec<(usize, usize)> = {
        let n_tiles = N.div_ceil(TILE);
        let mut p = Vec::with_capacity(n_tiles * (n_tiles + 1) / 2);
        for it in 0..n_tiles {
            for jt in it..n_tiles {
                p.push((it * TILE, jt * TILE));
            }
        }
        p
    };
    println!("Tile pairs: {}\n", pairs.len());

    for run in 0..RUNS {
        GEMM_NS.store(0, Ordering::Relaxed);
        EMIT_NS.store(0, Ordering::Relaxed);

        let t0 = Instant::now();

        let chunks: Vec<Vec<(u64, u64, f32)>> = pairs
            .par_iter()
            .map(|&(i0, j0)| {
                let i1 = (i0 + TILE).min(N);
                let j1 = (j0 + TILE).min(N);
                let xi = data.slice(s![i0..i1, ..]);
                let xj = data.slice(s![j0..j1, ..]);

                let g0 = Instant::now();
                let ip = ip_batch(xi, xj, Par::Seq);
                let g_ns = g0.elapsed().as_nanos() as u64;
                GEMM_NS.fetch_add(g_ns, Ordering::Relaxed);

                let h = i1 - i0;
                let w = j1 - j0;
                let mut local: Vec<(u64, u64, f32)> = Vec::with_capacity(h * w / 32 + 8);

                let e0 = Instant::now();
                for li in 0..h {
                    let gi = (i0 + li) as u64;
                    let lj_start = if i0 == j0 { li + 1 } else { 0 };
                    for lj in lj_start..w {
                        let gj = (j0 + lj) as u64;
                        let score = ip[(li, lj)];
                        if score >= THRESHOLD {
                            local.push((gi, gj, score));
                            local.push((gj, gi, score));
                        }
                    }
                }
                let e_ns = e0.elapsed().as_nanos() as u64;
                EMIT_NS.fetch_add(e_ns, Ordering::Relaxed);

                local
            })
            .collect();

        let c0 = Instant::now();
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        let mut all: Vec<(u64, u64, f32)> = Vec::with_capacity(total);
        for c in chunks {
            all.extend(c);
        }
        let c_ns = c0.elapsed().as_nanos() as u64;

        let t_ns = t0.elapsed().as_nanos() as u64;

        let gemm = GEMM_NS.load(Ordering::Relaxed);
        let emit = EMIT_NS.load(Ordering::Relaxed);

        // Note: gemm + emit are summed across rayon workers (wall-clock per
        // tile × n_workers). Wall-clock total t_ns is what the user sees.
        // To compare apples-to-apples we report both: per-worker breakdown
        // (sums to roughly threads × wall-clock) and the wall-clock fractions
        // implied by Amdahl-style scaling.
        let workers = rayon::current_num_threads() as u64;
        let wall_gemm = gemm / workers;
        let wall_emit = emit / workers;

        println!(
            "run {}: total wall {:.0}ms  edges={}",
            run + 1,
            t_ns as f64 / 1e6,
            all.len()
        );
        println!(
            "         gemm:   per-worker {:.0}ms  ≈ wall {:.0}ms  ({:.0}%)",
            gemm as f64 / 1e6 / workers as f64,
            wall_gemm as f64 / 1e6,
            100.0 * wall_gemm as f64 / t_ns as f64
        );
        println!(
            "         emit:   per-worker {:.0}ms  ≈ wall {:.0}ms  ({:.0}%)",
            emit as f64 / 1e6 / workers as f64,
            wall_emit as f64 / 1e6,
            100.0 * wall_emit as f64 / t_ns as f64
        );
        println!(
            "         concat: wall       {:.0}ms  ({:.0}%)",
            c_ns as f64 / 1e6,
            100.0 * c_ns as f64 / t_ns as f64
        );
        println!(
            "         remainder (rayon overhead, alloc, etc.): {:.0}ms ({:.0}%)\n",
            (t_ns as i64 - wall_gemm as i64 - wall_emit as i64 - c_ns as i64) as f64 / 1e6,
            100.0 * (t_ns as i64 - wall_gemm as i64 - wall_emit as i64 - c_ns as i64) as f64
                / t_ns as f64
        );

        if run == 0 {
            // Skip the first run from the average — JIT warmup, allocator priming.
            continue;
        }
        concat_ns += c_ns;
        total_ns += t_ns;
        total_edges += all.len();
    }

    let avg_runs = (RUNS - 1) as u64;
    let total_avg_ms = total_ns as f64 / 1e6 / avg_runs as f64;
    println!("=== averages over runs 2..{} ===", RUNS);
    println!("total wall:    {:.0}ms", total_avg_ms);
    println!("avg edges:     {}", total_edges / avg_runs as usize);
    println!("(see per-run breakdown above for gemm/emit/concat splits)");
    println!();
    println!("Phase-1 decision criteria from docs/index-faer-tuning-plan.md:");
    println!("  gemm wall-fraction > 85%  →  ABANDON Phase 2-3 (BLAS-vs-faer kernel gap)");
    println!("  gemm wall-fraction < 70%  →  PROCEED to Phase 2");
}
