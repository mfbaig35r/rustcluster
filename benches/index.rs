//! Criterion benchmarks for `rustcluster::index`.
//!
//! Run with `cargo bench --bench index --no-default-features`. The
//! `--no-default-features` flag disables the `python` feature so this
//! compiles as a plain Rust crate.
//!
//! Shapes match the v1 walkthrough table (`docs/index-performance-plan.md`)
//! so we can compare results across versions:
//! - `flat_search_top10`  — `IndexFlatIP::search` at multiple (n, d, nq=1000, k=10)
//! - `flat_range_search`  — `IndexFlatIP::range_search` at the same shapes
//! - `similarity_graph`   — fused all-pairs above-threshold kernel
//!
//! These are the operations whose performance we're trying to lock in.
//! They use random unit-norm f32 vectors so the numbers are workload-shape
//! representative without being workload-specific.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::{Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rustcluster::index::{IndexFlatIP, SearchOpts, VectorIndex};

const SHAPES: &[(usize, usize)] = &[
    (10_000, 128),
    (50_000, 128),
    (100_000, 128),
    (50_000, 384),
    (20_000, 1536),
];

const NQ: usize = 1000;
const K: usize = 10;
const RANGE_THRESHOLD: f32 = 0.5;
const SIM_GRAPH_THRESHOLD: f32 = 0.5;

fn make_data(n: usize, d: usize, seed: u64) -> Array2<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data: Vec<f32> = (0..n * d).map(|_| rng.gen_range(-1.0_f32..1.0_f32)).collect();
    // L2-normalize each row.
    for i in 0..n {
        let row = &mut data[i * d..(i + 1) * d];
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-30);
        for v in row.iter_mut() {
            *v /= norm;
        }
    }
    Array2::from_shape_vec((n, d), data).unwrap()
}

fn build_index(data: ArrayView2<f32>) -> IndexFlatIP {
    let (_, d) = data.dim();
    let mut idx = IndexFlatIP::new(d);
    idx.add(data).unwrap();
    idx
}

fn bench_search_top10(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_ip_search_top10");
    group.sample_size(10);
    for &(n, d) in SHAPES {
        let data = make_data(n, d, 42);
        let queries = data.slice(ndarray::s![0..NQ, ..]).to_owned();
        let idx = build_index(data.view());
        let opts = SearchOpts::default();
        group.bench_with_input(BenchmarkId::from_parameter(format!("n{}_d{}", n, d)), &n, |b, _| {
            b.iter(|| idx.search(queries.view(), K, opts).unwrap());
        });
    }
    group.finish();
}

fn bench_range_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_ip_range_search");
    group.sample_size(10);
    for &(n, d) in SHAPES {
        let data = make_data(n, d, 43);
        let queries = data.slice(ndarray::s![0..NQ, ..]).to_owned();
        let idx = build_index(data.view());
        let opts = SearchOpts::default();
        group.bench_with_input(BenchmarkId::from_parameter(format!("n{}_d{}", n, d)), &n, |b, _| {
            b.iter(|| idx.range_search(queries.view(), RANGE_THRESHOLD, opts).unwrap());
        });
    }
    group.finish();
}

fn bench_similarity_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_ip_similarity_graph");
    group.sample_size(10);
    // Smaller n for similarity_graph — it's O(n²) so the n=100k case takes
    // long enough that Criterion's default sample size becomes painful.
    for &(n, d) in &[(10_000, 128), (50_000, 128), (50_000, 384), (20_000, 1536)] {
        let data = make_data(n, d, 44);
        let idx = build_index(data.view());
        group.bench_with_input(BenchmarkId::from_parameter(format!("n{}_d{}", n, d)), &n, |b, _| {
            b.iter(|| idx.similarity_graph(SIM_GRAPH_THRESHOLD, true));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search_top10, bench_range_search, bench_similarity_graph);
criterion_main!(benches);
