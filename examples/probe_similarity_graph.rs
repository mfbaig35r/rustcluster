//! Milestone 7 acceptance bench: similarity_graph_ip on the v1.2 fused
//! path vs the v1.1 faer-fallback path.
//!
//! Run twice — once with default features for the fused path, once with
//! `--features faer-fallback` for the v1.1 baseline:
//!
//!     cargo run --release --example probe_similarity_graph --no-default-features
//!     cargo run --release --example probe_similarity_graph --no-default-features --features faer-fallback
//!
//! Acceptance criterion: fused path ≥ 1.0× FAISS at d=1536. Since
//! v1.1 was 0.77× FAISS, we need ≥ 30% speedup vs v1.1.

use std::time::Instant;

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustcluster::index::ids::IdMap;
use rustcluster::index::similarity_graph::similarity_graph_ip;

const N: usize = 20_000;
const D: usize = 1536;
const THRESHOLD: f32 = 0.5;
const RUNS: usize = 5;

fn make_data() -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(1536);
    let mut data: Vec<f32> = (0..N * D)
        .map(|_| rng.gen_range(-1.0_f32..1.0))
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

fn main() {
    let path = if cfg!(feature = "faer-fallback") {
        "v1.1 path (faer-fallback)"
    } else {
        "v1.2 fused path"
    };

    println!("M7 bench: similarity_graph_ip — {}", path);
    println!("------------------------------------------------------");
    println!("n={N}, d={D}, threshold={THRESHOLD}, runs={RUNS} (first is warmup)\n");

    let data = make_data();
    let ids = make_ids(N);

    let mut wall_ms: Vec<f64> = Vec::with_capacity(RUNS);
    for r in 0..RUNS {
        let t0 = Instant::now();
        let g = similarity_graph_ip(data.view(), &ids, THRESHOLD, true);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        wall_ms.push(ms);
        println!("    run {}: {:>7.1} ms  edges={}", r + 1, ms, g.len());
    }

    let measured = &wall_ms[1..];
    let best = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = measured.iter().sum::<f64>() / measured.len() as f64;
    println!();
    println!("Best of runs 2..{}: {:.1} ms", RUNS, best);
    println!("Mean of runs 2..{}: {:.1} ms", RUNS, mean);

    println!();
    println!("Reference: FAISS at this shape ≈ 540 ms (Apple M-series, from prompt).");
    println!("Acceptance: best <= 540 ms is ≥ 1.0x FAISS.");
}
