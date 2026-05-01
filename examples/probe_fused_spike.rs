//! Step 0b of the v1.2 plan: validate that *fusion alone* — with faer
//! still doing the inner compute — closes a meaningful fraction of the
//! d=1536 FAISS gap.
//!
//! Compares the production `similarity_graph_ip` (full `h×w` score tile
//! materialized per outer tile) against `similarity_graph_ip_fused_spike`
//! (row-band sub-tile, per-band score buffer fits L1, never
//! materialized full).
//!
//! Run with:
//!
//!     cargo run --release --example probe_fused_spike --no-default-features
//!
//! Decision (from docs/v1.2-execution-plan.md):
//! - if fusion + faer reaches ≥0.95× FAISS at d=1536 → microkernel work
//!   is bonus, not load-bearing. Plan C / C+ candidate.
//! - otherwise the microkernel work in Plan A remains load-bearing.

use std::time::Instant;

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustcluster::index::ids::IdMap;
use rustcluster::index::similarity_graph::{
    similarity_graph_ip, similarity_graph_ip_fused_spike_with, EdgeList,
};

const N: usize = 20_000;
const D: usize = 1536;
const THRESHOLD: f32 = 0.5;
const RUNS: usize = 4;
const SUB_TILES: &[usize] = &[8, 16, 32, 64, 128];

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

fn make_ids(n: usize) -> IdMap {
    let mut m = IdMap::new();
    m.extend_sequential(n).unwrap();
    m
}

fn sorted_pairs(g: &EdgeList) -> Vec<(u64, u64, f32)> {
    let mut v: Vec<(u64, u64, f32)> = g
        .src
        .iter()
        .zip(g.dst.iter())
        .zip(g.scores.iter())
        .map(|((&a, &b), &s)| (a, b, s))
        .collect();
    v.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    v
}

fn assert_parity(label: &str, baseline: &[(u64, u64, f32)], other: &[(u64, u64, f32)]) {
    assert_eq!(
        baseline.len(),
        other.len(),
        "{label}: edge count mismatch ({} vs {})",
        baseline.len(),
        other.len()
    );
    for (i, (b, o)) in baseline.iter().zip(other.iter()).enumerate() {
        assert_eq!(
            (b.0, b.1),
            (o.0, o.1),
            "{label}: pair mismatch at i={i}: {:?} vs {:?}",
            (b.0, b.1),
            (o.0, o.1)
        );
        assert!(
            (b.2 - o.2).abs() < 1e-4,
            "{label}: score mismatch at ({}, {}): {} vs {}",
            b.0,
            b.1,
            b.2,
            o.2
        );
    }
}

fn time_runs(
    label: &str,
    runs: usize,
    mut f: impl FnMut() -> EdgeList,
) -> (Vec<f64>, EdgeList) {
    let mut last: Option<EdgeList> = None;
    let mut wall_ms: Vec<f64> = Vec::with_capacity(runs);
    for r in 0..runs {
        let t0 = Instant::now();
        let g = f();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        wall_ms.push(ms);
        println!("    run {}: {:>7.1} ms  edges={}", r + 1, ms, g.len());
        last = Some(g);
    }
    let _ = label;
    (wall_ms, last.unwrap())
}

fn report(label: &str, runs: &[f64]) -> f64 {
    // Skip first run (warmup), report best of remaining.
    let measured = &runs[1..];
    let best = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = measured.iter().sum::<f64>() / measured.len() as f64;
    println!(
        "  {:32} best {:>7.1} ms   mean {:>7.1} ms",
        label, best, mean
    );
    best
}

fn main() {
    println!("Step 0b: fused-architecture spike (faer inner compute)");
    println!("--------------------------------------------------------");
    println!("n={N}, d={D}, threshold={THRESHOLD}, runs={RUNS} (first is warmup)");
    println!();

    println!("Building data...");
    let data = make_data();
    let ids = make_ids(N);
    println!("done.\n");

    // Baseline.
    println!("Baseline: similarity_graph_ip (full tile materialized)");
    let (baseline_runs, baseline_g) =
        time_runs("baseline", RUNS, || similarity_graph_ip(data.view(), &ids, THRESHOLD, true));
    let baseline_sorted = sorted_pairs(&baseline_g);
    println!();

    // Spike at each sub-tile size.
    let mut spike_results: Vec<(usize, Vec<f64>, EdgeList)> = Vec::new();
    for &sub in SUB_TILES {
        println!("Spike: sub-tile = {sub} rows");
        let (runs, g) = time_runs(&format!("spike sub={sub}"), RUNS, || {
            similarity_graph_ip_fused_spike_with(data.view(), &ids, THRESHOLD, true, sub)
        });
        let sorted = sorted_pairs(&g);
        assert_parity(&format!("spike sub={sub}"), &baseline_sorted, &sorted);
        spike_results.push((sub, runs, g));
        println!();
    }

    println!("Parity check: all spike variants emit identical edges to baseline.\n");

    // Summary.
    println!("=== Summary (best of runs 2..{}) ===", RUNS);
    let baseline_best = report("baseline (full-tile materialized)", &baseline_runs);
    for (sub, runs, _) in &spike_results {
        let label = format!("spike sub={}", sub);
        let best = report(&label, runs);
        let speedup = baseline_best / best;
        println!(
            "    {:32} {:>5.2}x vs baseline   ({}{:.0}%)",
            "",
            speedup,
            if speedup >= 1.0 { "+" } else { "" },
            (speedup - 1.0) * 100.0
        );
    }

    println!();
    println!("Decision (from docs/v1.2-execution-plan.md, Step 0c):");
    println!("  Apple Silicon (M-series) FAISS-relative target at d=1536:");
    println!("    baseline today: 0.77x FAISS");
    println!("    if best spike speedup >= 1.30x  → spike alone reaches >=1.0x FAISS,");
    println!("                                       Plan C / C+ candidate");
    println!("    if best spike speedup ~ 1.10-1.30x → fusion helps but microkernel still load-bearing");
    println!("    if best spike speedup < 1.05x  → fusion alone insufficient at faer microkernel quality");
    println!();
    println!("Document the result in docs/v1.2-prototype-result.md (Step 0b section).");
}
