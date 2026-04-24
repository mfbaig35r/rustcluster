"""Practical benchmark: cluster slotting vs full refit.

Simulates a production embedding pipeline:
1. Fit on initial batch of embeddings
2. Snapshot the cluster state
3. Slot new batches incrementally
4. Measure: accuracy vs full refit, timing, confidence quality, drift detection
"""

import time
import tempfile
import os
import numpy as np
from rustcluster import KMeans, MiniBatchKMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster


def generate_clustered_embeddings(n_clusters, n_per_cluster, dim, noise_scale=0.3, seed=42):
    """Generate synthetic embeddings with known cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)  # unit norm

    data = []
    labels_true = []
    for i in range(n_clusters):
        noise = rng.standard_normal((n_per_cluster, dim)) * noise_scale
        points = centers[i] + noise
        data.append(points)
        labels_true.extend([i] * n_per_cluster)

    return np.vstack(data).astype(np.float64), np.array(labels_true), centers


def cluster_agreement(labels_a, labels_b):
    """Fraction of points assigned to the same cluster (permutation-invariant)."""
    from itertools import permutations

    unique_a = set(labels_a)
    unique_b = set(labels_b)
    k = max(len(unique_a), len(unique_b))

    if k > 10:
        # For large k, use Hungarian-style greedy matching
        return _greedy_agreement(labels_a, labels_b)

    best = 0.0
    for perm in permutations(range(k)):
        mapping = {old: new for old, new in zip(range(k), perm)}
        mapped = np.array([mapping.get(l, -1) for l in labels_a])
        agreement = np.mean(mapped == labels_b)
        best = max(best, agreement)
    return best


def _greedy_agreement(labels_a, labels_b):
    """Greedy cluster matching for large k."""
    from collections import Counter

    k_a = max(labels_a) + 1
    k_b = max(labels_b) + 1

    # Build confusion matrix
    confusion = np.zeros((k_a, k_b), dtype=int)
    for a, b in zip(labels_a, labels_b):
        if a >= 0 and b >= 0:
            confusion[a, b] += 1

    # Greedy match: pick best pair, remove, repeat
    matched = 0
    used_a = set()
    used_b = set()
    flat = [(confusion[i, j], i, j) for i in range(k_a) for j in range(k_b)]
    flat.sort(reverse=True)

    for count, i, j in flat:
        if i not in used_a and j not in used_b:
            matched += count
            used_a.add(i)
            used_b.add(j)

    return matched / len(labels_a)


def benchmark_kmeans(n_clusters=10, n_train=5000, n_test=1000, dim=32):
    """Benchmark KMeans snapshot slotting vs full refit."""
    print(f"\n{'='*60}")
    print(f"KMeans: k={n_clusters}, n_train={n_train}, n_test={n_test}, d={dim}")
    print(f"{'='*60}")

    data_all, labels_true, centers = generate_clustered_embeddings(
        n_clusters, (n_train + n_test) // n_clusters, dim, noise_scale=0.3
    )
    X_train = data_all[:n_train]
    X_test = data_all[n_train:n_train + n_test]

    # --- Fit ---
    t0 = time.perf_counter()
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(X_train)
    t_fit = time.perf_counter() - t0
    print(f"\nFit time:           {t_fit*1000:.1f} ms")

    # --- Snapshot ---
    t0 = time.perf_counter()
    snap = model.snapshot()
    t_snapshot = time.perf_counter() - t0
    print(f"Snapshot time:      {t_snapshot*1000:.3f} ms")

    # --- Slot new points ---
    t0 = time.perf_counter()
    labels_slotted = snap.assign(X_test)
    t_slot = time.perf_counter() - t0
    print(f"Slot {n_test} points:   {t_slot*1000:.2f} ms  ({n_test/t_slot:.0f} pts/sec)")

    # --- Full refit on combined data ---
    X_combined = np.vstack([X_train, X_test])
    t0 = time.perf_counter()
    model_refit = KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(X_combined)
    t_refit = time.perf_counter() - t0
    labels_refit_test = model_refit.predict(X_test)
    print(f"Full refit time:    {t_refit*1000:.1f} ms")
    print(f"Speedup (slot/refit): {t_refit/t_slot:.0f}x")

    # --- Agreement ---
    agreement = cluster_agreement(labels_slotted, labels_refit_test)
    print(f"\nSlot vs refit agreement: {agreement*100:.1f}%")

    # --- Confidence analysis ---
    result = snap.assign_with_scores(X_test)
    confs = result.confidences_
    print(f"\nConfidence scores:")
    print(f"  Mean:   {np.mean(confs):.3f}")
    print(f"  Median: {np.median(confs):.3f}")
    print(f"  Min:    {np.min(confs):.3f}")
    print(f"  P10:    {np.percentile(confs, 10):.3f}")
    print(f"  P90:    {np.percentile(confs, 90):.3f}")

    # --- Rejection test ---
    result_strict = snap.assign_with_scores(X_test, confidence_threshold=0.3)
    n_rejected = sum(result_strict.rejected_)
    print(f"\nRejection (confidence < 0.3): {n_rejected}/{n_test} ({n_rejected/n_test*100:.1f}%)")

    # --- Save/load roundtrip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "kmeans_snap")
        t0 = time.perf_counter()
        snap.save(path)
        t_save = time.perf_counter() - t0

        size_st = os.path.getsize(os.path.join(path, "centroids.safetensors"))
        size_json = os.path.getsize(os.path.join(path, "metadata.json"))

        t0 = time.perf_counter()
        loaded = ClusterSnapshot.load(path)
        t_load = time.perf_counter() - t0

        labels_loaded = loaded.assign(X_test)
        assert np.array_equal(labels_slotted, labels_loaded), "Load roundtrip mismatch!"

        print(f"\nPersistence:")
        print(f"  Save:  {t_save*1000:.2f} ms")
        print(f"  Load:  {t_load*1000:.2f} ms")
        print(f"  Size:  {size_st + size_json:,} bytes ({size_st:,} tensors + {size_json:,} metadata)")

    # --- Drift detection ---
    report_same = snap.drift_report(X_test)
    X_shifted = X_test + 5.0  # shift all points
    report_drift = snap.drift_report(X_shifted)

    print(f"\nDrift detection:")
    print(f"  Same distribution:    mean_dist={report_same.global_mean_distance_:.3f}")
    print(f"  Shifted (+5.0):       mean_dist={report_drift.global_mean_distance_:.3f}")
    print(f"  Drift ratio:          {report_drift.global_mean_distance_ / max(report_same.global_mean_distance_, 1e-10):.1f}x")


def benchmark_embedding_cluster(n_clusters=20, n_train=10000, n_test=2000, dim=256, pca_dim=32):
    """Benchmark EmbeddingCluster snapshot (the realistic embedding pipeline)."""
    print(f"\n{'='*60}")
    print(f"EmbeddingCluster: k={n_clusters}, n_train={n_train}, n_test={n_test}")
    print(f"  dim={dim} → PCA → {pca_dim}")
    print(f"{'='*60}")

    data_all, labels_true, _ = generate_clustered_embeddings(
        n_clusters, (n_train + n_test) // n_clusters, dim, noise_scale=0.5, seed=123
    )
    X_train = data_all[:n_train].astype(np.float32)
    X_test = data_all[n_train:n_train + n_test].astype(np.float32)

    # --- Fit ---
    t0 = time.perf_counter()
    model = EmbeddingCluster(
        n_clusters=n_clusters, reduction_dim=pca_dim, random_state=42, n_init=3
    ).fit(X_train)
    t_fit = time.perf_counter() - t0
    print(f"\nFit time:           {t_fit*1000:.1f} ms (includes PCA + spherical k-means)")

    # --- Snapshot ---
    t0 = time.perf_counter()
    snap = model.snapshot()
    t_snapshot = time.perf_counter() - t0
    print(f"Snapshot time:      {t_snapshot*1000:.3f} ms")
    print(f"Snapshot shape:     input_dim={snap.input_dim} → d={snap.d} (PCA baked in)")

    # --- Slot ---
    t0 = time.perf_counter()
    labels_slotted = snap.assign(X_test)
    t_slot = time.perf_counter() - t0
    print(f"Slot {n_test} points:  {t_slot*1000:.2f} ms  ({n_test/t_slot:.0f} pts/sec)")
    print(f"  (includes L2 norm + PCA transform + spherical assignment)")

    # --- Full refit ---
    X_combined = np.vstack([X_train, X_test])
    t0 = time.perf_counter()
    model_refit = EmbeddingCluster(
        n_clusters=n_clusters, reduction_dim=pca_dim, random_state=42, n_init=3
    ).fit(X_combined)
    t_refit = time.perf_counter() - t0
    print(f"Full refit time:    {t_refit*1000:.1f} ms")
    print(f"Speedup (slot/refit): {t_refit/t_slot:.0f}x")

    # --- Confidence ---
    result = snap.assign_with_scores(X_test)
    confs = result.confidences_
    print(f"\nConfidence scores:")
    print(f"  Mean:   {np.mean(confs):.3f}")
    print(f"  Median: {np.median(confs):.3f}")
    print(f"  P10:    {np.percentile(confs, 10):.3f}")
    print(f"  P90:    {np.percentile(confs, 90):.3f}")

    # --- Persistence ---
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "emb_snap")
        snap.save(path)

        size_st = os.path.getsize(os.path.join(path, "centroids.safetensors"))
        size_json = os.path.getsize(os.path.join(path, "metadata.json"))

        loaded = ClusterSnapshot.load(path)
        labels_loaded = loaded.assign(X_test)
        assert np.array_equal(labels_slotted, labels_loaded), "Load roundtrip mismatch!"

        print(f"\nPersistence:")
        print(f"  Size: {(size_st + size_json) / 1024:.1f} KB")
        print(f"    centroids:  {size_st / 1024:.1f} KB")
        print(f"    metadata:   {size_json / 1024:.1f} KB")
        print(f"  (vs training data: {X_train.nbytes / 1024 / 1024:.1f} MB)")


def benchmark_scaling():
    """How does slotting scale with batch size?"""
    print(f"\n{'='*60}")
    print(f"Scaling: slot time vs batch size (k=20, d=128)")
    print(f"{'='*60}")

    dim = 128
    n_clusters = 20
    n_train = 5000

    data_train, _, _ = generate_clustered_embeddings(
        n_clusters, n_train // n_clusters, dim, seed=42
    )
    model = KMeans(n_clusters=n_clusters, random_state=42).fit(data_train)
    snap = model.snapshot()

    print(f"\n{'Batch size':>12}  {'Slot time':>12}  {'Throughput':>15}")
    print(f"{'-'*12}  {'-'*12}  {'-'*15}")

    for n_test in [100, 1_000, 10_000, 50_000, 100_000]:
        rng = np.random.default_rng(99)
        X_test = rng.standard_normal((n_test, dim))

        # Warmup
        snap.assign(X_test[:100])

        t0 = time.perf_counter()
        snap.assign(X_test)
        t = time.perf_counter() - t0

        print(f"{n_test:>12,}  {t*1000:>10.2f} ms  {n_test/t:>12,.0f} pts/sec")


if __name__ == "__main__":
    benchmark_kmeans(n_clusters=10, n_train=5000, n_test=1000, dim=32)
    benchmark_embedding_cluster(n_clusters=20, n_train=10000, n_test=2000, dim=256, pca_dim=32)
    benchmark_scaling()
    print(f"\n{'='*60}")
    print("All benchmarks complete.")
