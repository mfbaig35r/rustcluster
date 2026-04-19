"""Benchmark rustcluster vs scikit-learn KMeans.

Run:
    python benches/benchmark.py

Requires scikit-learn: pip install scikit-learn
"""

import os
import sys
import time

import numpy as np

# Force single-threaded for fair comparison
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

from rustcluster import KMeans as RustKMeans

try:
    from sklearn.cluster import KMeans as SklearnKMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("scikit-learn not installed — running rustcluster benchmarks only.\n")


def make_blobs(n_samples, n_features, n_clusters, seed=0):
    """Generate synthetic clustered data."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-50, 50, size=(n_clusters, n_features))
    labels = rng.integers(0, n_clusters, size=n_samples)
    X = np.empty((n_samples, n_features), dtype=np.float64)
    for i in range(n_samples):
        X[i] = centers[labels[i]] + rng.normal(0, 1, size=n_features)
    return np.ascontiguousarray(X, dtype=np.float64)


def benchmark_one(X, n_clusters, n_runs=5):
    """Benchmark a single configuration. Returns (rust_time, sklearn_time) in seconds."""
    # Warmup
    RustKMeans(n_clusters=n_clusters, n_init=1, random_state=0, max_iter=100).fit(X)

    # Rust benchmark
    rust_times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        RustKMeans(n_clusters=n_clusters, n_init=1, random_state=r, max_iter=100).fit(X)
        rust_times.append(time.perf_counter() - t0)
    rust_median = sorted(rust_times)[n_runs // 2]

    sklearn_median = None
    if HAS_SKLEARN:
        # Warmup
        SklearnKMeans(
            n_clusters=n_clusters, n_init=1, random_state=0, max_iter=100, algorithm="lloyd"
        ).fit(X)

        sklearn_times = []
        for r in range(n_runs):
            t0 = time.perf_counter()
            SklearnKMeans(
                n_clusters=n_clusters, n_init=1, random_state=r, max_iter=100, algorithm="lloyd"
            ).fit(X)
            sklearn_times.append(time.perf_counter() - t0)
        sklearn_median = sorted(sklearn_times)[n_runs // 2]

    return rust_median, sklearn_median


def main():
    configs = [
        # (n_samples, n_features, n_clusters)
        (1_000, 2, 8),
        (1_000, 8, 8),
        (1_000, 16, 8),
        (1_000, 32, 8),
        (10_000, 2, 8),
        (10_000, 8, 8),
        (10_000, 16, 32),
        (10_000, 32, 32),
        (100_000, 2, 8),
        (100_000, 8, 32),
        (100_000, 16, 32),
        (100_000, 32, 32),
    ]

    print("rustcluster vs scikit-learn KMeans Benchmark")
    print("=" * 75)
    print(f"{'n':>10} {'d':>5} {'k':>5} {'rust (ms)':>12} {'sklearn (ms)':>14} {'speedup':>10}")
    print("-" * 75)

    for n_samples, n_features, n_clusters in configs:
        X = make_blobs(n_samples, n_features, n_clusters)
        rust_t, sklearn_t = benchmark_one(X, n_clusters)

        rust_ms = rust_t * 1000
        if sklearn_t is not None:
            sklearn_ms = sklearn_t * 1000
            speedup = sklearn_t / rust_t if rust_t > 0 else float("inf")
            print(
                f"{n_samples:>10,} {n_features:>5} {n_clusters:>5} "
                f"{rust_ms:>12.2f} {sklearn_ms:>14.2f} {speedup:>9.2f}x"
            )
        else:
            print(f"{n_samples:>10,} {n_features:>5} {n_clusters:>5} {rust_ms:>12.2f} {'N/A':>14} {'N/A':>10}")

    print("-" * 75)
    print("\nNotes:")
    print("  - Single-threaded comparison (OMP/MKL/RAYON_NUM_THREADS=1)")
    print("  - n_init=1, max_iter=100, algorithm=lloyd")
    print("  - Median of 5 runs after warmup")
    print("  - speedup = sklearn_time / rust_time")


if __name__ == "__main__":
    main()
