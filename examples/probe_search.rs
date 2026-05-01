//! M5/M7 perf probe: search and range_search at d=1536 on the v1.2
//! fused path vs the v1.1 faer-fallback path.
//!
//! Run twice:
//!
//!     cargo run --release --example probe_search --no-default-features
//!     cargo run --release --example probe_search --no-default-features --features faer-fallback
//!
//! Plan targets at d=1536:
//!   search       baseline 0.32× FAISS, target ≥ 0.50× FAISS
//!   range_search baseline 0.37× FAISS, target ≥ 0.50× FAISS

use std::time::Instant;

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustcluster::index::{IndexFlatIP, SearchOpts, VectorIndex};

const N: usize = 20_000;
const D: usize = 1536;
const NQ: usize = 1000;
const K: usize = 10;
const RANGE_THRESHOLD: f32 = 0.5;
const RUNS: usize = 5;

fn make_data(n: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0_f32..1.0)).collect();
    for i in 0..n {
        let row = &mut data[i * d..(i + 1) * d];
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn time_op<F: FnMut()>(label: &str, runs: usize, mut f: F) -> f64 {
    let mut wall_ms: Vec<f64> = Vec::with_capacity(runs);
    for r in 0..runs {
        let t0 = Instant::now();
        f();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        wall_ms.push(ms);
        println!("    run {}: {:>7.2} ms", r + 1, ms);
    }
    let measured = &wall_ms[1..];
    let best = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    let mean = measured.iter().sum::<f64>() / measured.len() as f64;
    println!("  {:24} best {:>7.2} ms  mean {:>7.2} ms", label, best, mean);
    best
}

fn main() {
    let path = if cfg!(feature = "faer-fallback") {
        "v1.1 path (faer-fallback)"
    } else {
        "v1.2 fused path"
    };

    println!("M5/M7 perf probe: search + range_search — {}", path);
    println!("------------------------------------------------------");
    println!("n={N}, d={D}, nq={NQ}, k={K}, range threshold={RANGE_THRESHOLD}, runs={RUNS}\n");

    let data = make_data(N, D, 42);
    let queries = data.slice(ndarray::s![0..NQ, ..]).to_owned();

    let mut idx = IndexFlatIP::new(D);
    idx.add(data.view()).unwrap();
    let opts = SearchOpts::default();

    println!("=== search top-{K} (nq={NQ}) ===");
    let _ = time_op("search top-10", RUNS, || {
        let _ = idx.search(queries.view(), K, opts).unwrap();
    });
    println!();

    println!("=== range_search threshold={RANGE_THRESHOLD} (nq={NQ}) ===");
    let _ = time_op("range_search 0.5", RUNS, || {
        let _ = idx.range_search(queries.view(), RANGE_THRESHOLD, opts).unwrap();
    });
}
