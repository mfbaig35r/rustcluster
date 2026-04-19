"""Multi-threaded scaling benchmark: rustcluster vs scikit-learn.

Runs each configuration in a subprocess with controlled thread count
(RAYON_NUM_THREADS for rustcluster, OMP_NUM_THREADS for sklearn).

Run:
    python benches/benchmark_threads.py
"""

import json
import os
import subprocess
import sys

# Configs to benchmark
CONFIGS = [
    (100_000, 8, 32),
    (100_000, 32, 32),
]
THREAD_COUNTS = [1, 2, 4, 8]
N_RUNS = 5


WORKER_SCRIPT = '''
import os, sys, time, json
import numpy as np

n, d, k = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
n_runs = int(sys.argv[4])
lib = sys.argv[5]  # "rust" or "sklearn"

rng = np.random.default_rng(0)
centers = rng.uniform(-50, 50, size=(k, d))
labels = rng.integers(0, k, size=n)
X = np.empty((n, d), dtype=np.float64)
for i in range(n):
    X[i] = centers[labels[i]] + rng.normal(0, 1, size=d)
X = np.ascontiguousarray(X)

if lib == "rust":
    from rustcluster import KMeans
    KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=100).fit(X)  # warmup
    times = []
    for r in range(n_runs):
        m = KMeans(n_clusters=k, n_init=1, random_state=r, max_iter=100)
        t0 = time.perf_counter()
        m.fit(X)
        times.append(time.perf_counter() - t0)
else:
    from sklearn.cluster import KMeans
    KMeans(n_clusters=k, n_init=1, random_state=0, max_iter=100, algorithm="lloyd").fit(X)
    times = []
    for r in range(n_runs):
        m = KMeans(n_clusters=k, n_init=1, random_state=r, max_iter=100, algorithm="lloyd")
        t0 = time.perf_counter()
        m.fit(X)
        times.append(time.perf_counter() - t0)

times.sort()
median = times[n_runs // 2]
print(json.dumps({"median_ms": median * 1000}))
'''


def run_worker(n, d, k, threads, lib):
    """Run benchmark in subprocess with controlled thread count."""
    env = os.environ.copy()
    if lib == "rust":
        env["RAYON_NUM_THREADS"] = str(threads)
    else:
        env["OMP_NUM_THREADS"] = str(threads)
        env["MKL_NUM_THREADS"] = str(threads)
        env["OPENBLAS_NUM_THREADS"] = str(threads)

    result = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, str(n), str(d), str(k), str(N_RUNS), lib],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        print(f"  ERROR ({lib}, {threads}t): {result.stderr.strip()}", file=sys.stderr)
        return None
    return json.loads(result.stdout.strip())["median_ms"]


def main():
    try:
        import sklearn
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("scikit-learn not installed — running rustcluster only.\n")

    print("Multi-threaded Scaling Benchmark")
    print("=" * 80)

    for n, d, k in CONFIGS:
        print(f"\nn={n:,}  d={d}  k={k}")
        print(f"{'threads':>8} {'rust (ms)':>12} {'sklearn (ms)':>14} {'speedup':>10}")
        print("-" * 50)

        rust_1t = None
        sklearn_1t = None

        for threads in THREAD_COUNTS:
            rust_ms = run_worker(n, d, k, threads, "rust")
            sklearn_ms = run_worker(n, d, k, threads, "sklearn") if has_sklearn else None

            if threads == 1:
                rust_1t = rust_ms
                sklearn_1t = sklearn_ms

            rust_str = f"{rust_ms:.2f}" if rust_ms else "ERR"
            if sklearn_ms:
                speedup = sklearn_ms / rust_ms if rust_ms else 0
                print(f"{threads:>8} {rust_str:>12} {sklearn_ms:>14.2f} {speedup:>9.2f}x")
            else:
                print(f"{threads:>8} {rust_str:>12} {'N/A':>14} {'N/A':>10}")

        # Print scaling summary
        if rust_1t:
            print(f"\nRust scaling from 1 thread:")
            for threads in THREAD_COUNTS:
                ms = run_worker(n, d, k, threads, "rust")
                if ms:
                    print(f"  {threads}t: {ms:.2f}ms ({rust_1t/ms:.2f}x vs 1t)")

    print("\n" + "=" * 80)
    print("Notes:")
    print("  - Each config run in isolated subprocess with env-controlled thread count")
    print(f"  - Median of {N_RUNS} runs after warmup")
    print("  - n_init=1, max_iter=100")


if __name__ == "__main__":
    main()
