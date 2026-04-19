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


def time_fit(make_model, X, n_runs=5):
    """Time a model fit, return median of n_runs after warmup.

    Uses a different random_state per run to avoid measuring a single
    lucky/unlucky initialization. make_model receives the run index.
    """
    make_model(0).fit(X)  # warmup
    times = []
    for r in range(n_runs):
        t0 = time.perf_counter()
        make_model(r).fit(X)
        times.append(time.perf_counter() - t0)
    return sorted(times)[n_runs // 2]


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

    header = f"{'n':>10} {'d':>5} {'k':>5} {'lloyd (ms)':>12} {'hamerly (ms)':>14} {'sklearn (ms)':>14} {'lloyd sp':>10} {'ham sp':>10}"

    print("rustcluster (Lloyd + Hamerly) vs scikit-learn KMeans Benchmark")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for n_samples, n_features, n_clusters in configs:
        X = make_blobs(n_samples, n_features, n_clusters)

        lloyd_t = time_fit(
            lambda r: RustKMeans(n_clusters=n_clusters, n_init=1, random_state=r, max_iter=100, algorithm="lloyd"),
            X,
        )
        hamerly_t = time_fit(
            lambda r: RustKMeans(n_clusters=n_clusters, n_init=1, random_state=r, max_iter=100, algorithm="hamerly"),
            X,
        )

        lloyd_ms = lloyd_t * 1000
        hamerly_ms = hamerly_t * 1000

        if HAS_SKLEARN:
            sklearn_t = time_fit(
                lambda r: SklearnKMeans(n_clusters=n_clusters, n_init=1, random_state=r, max_iter=100, algorithm="lloyd"),
                X,
            )
            sklearn_ms = sklearn_t * 1000
            lloyd_sp = sklearn_t / lloyd_t if lloyd_t > 0 else float("inf")
            hamerly_sp = sklearn_t / hamerly_t if hamerly_t > 0 else float("inf")
            print(
                f"{n_samples:>10,} {n_features:>5} {n_clusters:>5} "
                f"{lloyd_ms:>12.2f} {hamerly_ms:>14.2f} {sklearn_ms:>14.2f} "
                f"{lloyd_sp:>9.2f}x {hamerly_sp:>9.2f}x"
            )
        else:
            print(
                f"{n_samples:>10,} {n_features:>5} {n_clusters:>5} "
                f"{lloyd_ms:>12.2f} {hamerly_ms:>14.2f} {'N/A':>14} {'N/A':>10} {'N/A':>10}"
            )

    print("-" * len(header))
    print("\nNotes:")
    print("  - Single-threaded comparison (OMP/MKL/RAYON_NUM_THREADS=1)")
    print("  - n_init=1, max_iter=100")
    print("  - Median of 5 runs after warmup")
    print("  - lloyd sp / ham sp = sklearn_time / rust_time")


if __name__ == "__main__":
    main()
