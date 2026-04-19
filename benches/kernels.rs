use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustcluster::_bench_api::*;

fn random_vec(n: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..n).map(|_| rng.gen_range(-100.0..100.0)).collect()
}

fn random_array2(n: usize, d: usize, rng: &mut StdRng) -> Array2<f64> {
    let data: Vec<f64> = (0..n * d).map(|_| rng.gen_range(-100.0..100.0)).collect();
    Array2::from_shape_vec((n, d), data).unwrap()
}

// ---------------------------------------------------------------------------
// 1. squared_euclidean across dimensions
// ---------------------------------------------------------------------------

fn bench_squared_euclidean(c: &mut Criterion) {
    let mut group = c.benchmark_group("squared_euclidean");
    let mut rng = StdRng::seed_from_u64(42);

    for d in [2, 4, 8, 16, 32, 64, 128, 256, 512] {
        let a = random_vec(d, &mut rng);
        let b = random_vec(d, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(d), &d, |bench, _| {
            bench.iter(|| squared_euclidean(black_box(&a), black_box(&b)));
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. assign_nearest: single point → k centroids
// ---------------------------------------------------------------------------

fn bench_assign_nearest(c: &mut Criterion) {
    let mut group = c.benchmark_group("assign_nearest");
    let mut rng = StdRng::seed_from_u64(42);

    let configs: Vec<(usize, usize)> = vec![
        (8, 8),    // low d, low k
        (8, 32),   // low d, moderate k
        (16, 32),  // moderate d and k
        (32, 32),  // higher d
        (32, 128), // higher k
        (128, 64), // high d
    ];

    for (d, k) in configs {
        let point = random_vec(d, &mut rng);
        let centroids = random_vec(k * d, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("single", format!("d{}_k{}", d, k)),
            &(d, k),
            |bench, _| {
                bench.iter(|| assign_nearest(black_box(&point), black_box(&centroids), k, d));
            },
        );
    }

    // Multi-point: 1000 points through the assignment step
    for (d, k) in [(8, 8), (32, 32)] {
        let n = 1000;
        let data = random_vec(n * d, &mut rng);
        let centroids = random_vec(k * d, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("batch_1000", format!("d{}_k{}", d, k)),
            &(d, k),
            |bench, _| {
                bench.iter(|| {
                    for i in 0..n {
                        let point = &data[i * d..(i + 1) * d];
                        black_box(assign_nearest(point, &centroids, k, d));
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. kmeans++ initialization
// ---------------------------------------------------------------------------

fn bench_kmeans_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_plus_plus_init");

    let configs: Vec<(usize, usize, usize)> = vec![
        (1_000, 8, 8),
        (1_000, 32, 32),
        (10_000, 8, 8),
        (10_000, 32, 32),
    ];

    for (n, d, k) in configs {
        let mut rng = StdRng::seed_from_u64(42);
        let data = random_array2(n, d, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("init", format!("n{}_d{}_k{}", n, d, k)),
            &(n, d, k),
            |bench, _| {
                bench.iter(|| {
                    let mut rng = StdRng::seed_from_u64(0);
                    black_box(kmeans_plus_plus_init(&data.view(), k, &mut rng));
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Full fit: Lloyd vs Hamerly (single n_init, few iterations)
// ---------------------------------------------------------------------------

fn bench_full_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_fit");
    group.sample_size(20); // larger workloads, fewer samples

    let configs: Vec<(usize, usize, usize)> = vec![
        (10_000, 8, 8),
        (10_000, 32, 32),
        (100_000, 8, 8),
        (100_000, 32, 32),
    ];

    for (n, d, k) in configs {
        let mut rng = StdRng::seed_from_u64(42);
        let data = random_array2(n, d, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("lloyd", format!("n{}_d{}_k{}", n, d, k)),
            &(n, d, k),
            |bench, _| {
                bench.iter(|| {
                    black_box(
                        run_kmeans_n_init(&data.view(), k, 20, 1e-4, 42, 1, Algorithm::Lloyd)
                            .unwrap(),
                    );
                });
            },
        );

        if k >= 2 {
            group.bench_with_input(
                BenchmarkId::new("hamerly", format!("n{}_d{}_k{}", n, d, k)),
                &(n, d, k),
                |bench, _| {
                    bench.iter(|| {
                        black_box(
                            run_kmeans_n_init(&data.view(), k, 20, 1e-4, 42, 1, Algorithm::Hamerly)
                                .unwrap(),
                        );
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_squared_euclidean,
    bench_assign_nearest,
    bench_kmeans_init,
    bench_full_fit
);
criterion_main!(benches);
