"""Benchmark: how cluster separation affects slotting quality.

Varies noise_scale from well-separated to heavily overlapping.
Shows: agreement, confidence distribution, rejection rates.
"""

import time
import numpy as np
from rustcluster import KMeans
from rustcluster.experimental import EmbeddingCluster


def generate_clustered_embeddings(n_clusters, n_per_cluster, dim, noise_scale, seed=42):
    rng = np.random.default_rng(seed)
    # Space centers further apart for cleaner separation control
    centers = rng.standard_normal((n_clusters, dim)) * 3.0
    data = []
    labels_true = []
    for i in range(n_clusters):
        noise = rng.standard_normal((n_per_cluster, dim)) * noise_scale
        points = centers[i] + noise
        data.append(points)
        labels_true.extend([i] * n_per_cluster)
    return np.vstack(data).astype(np.float64), np.array(labels_true), centers


def label_accuracy(predicted, true_labels, k):
    """What % of slotted labels match the ground truth (best permutation)?"""
    # Build confusion matrix, greedy match
    confusion = np.zeros((k, k), dtype=int)
    for p, t in zip(predicted, true_labels):
        if 0 <= p < k and 0 <= t < k:
            confusion[p, t] += 1

    matched = 0
    used_p, used_t = set(), set()
    flat = sorted(
        [(confusion[i, j], i, j) for i in range(k) for j in range(k)],
        reverse=True,
    )
    for count, i, j in flat:
        if i not in used_p and j not in used_t:
            matched += count
            used_p.add(i)
            used_t.add(j)
    return matched / len(predicted)


def run_separation_sweep():
    k = 8
    n_per = 500
    n_train = k * n_per
    n_test = k * 100
    dim = 32

    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

    print(f"KMeans separation sweep: k={k}, d={dim}, n_train={n_train}, n_test={n_test}")
    print(f"Centers spread: 3.0 * N(0,1) in {dim}d")
    print()
    print(f"{'Noise':>6}  {'Accuracy':>8}  {'Conf Mean':>9}  {'Conf P10':>8}  "
          f"{'Conf P50':>8}  {'Conf P90':>8}  "
          f"{'Rej@0.1':>7}  {'Rej@0.3':>7}  {'Rej@0.5':>7}  {'Slot ms':>8}")
    print(f"{'-'*6}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")

    for noise in noise_levels:
        data_all, labels_true, centers = generate_clustered_embeddings(
            k, n_per + 100, dim, noise_scale=noise, seed=42
        )
        X_train = data_all[:n_train]
        y_train = labels_true[:n_train]
        X_test = data_all[n_train:n_train + n_test]
        y_test = labels_true[n_train:n_train + n_test]

        model = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_train)
        snap = model.snapshot()

        t0 = time.perf_counter()
        result = snap.assign_with_scores(X_test)
        t_slot = (time.perf_counter() - t0) * 1000

        labels = result.labels_
        confs = result.confidences_

        # Accuracy against ground truth
        acc = label_accuracy(labels, y_test, k)

        # Rejection rates at various thresholds
        rej_01 = np.mean(confs < 0.1)
        rej_03 = np.mean(confs < 0.3)
        rej_05 = np.mean(confs < 0.5)

        print(f"{noise:>6.1f}  {acc:>7.1%}  {np.mean(confs):>9.3f}  "
              f"{np.percentile(confs, 10):>8.3f}  {np.median(confs):>8.3f}  "
              f"{np.percentile(confs, 90):>8.3f}  "
              f"{rej_01:>6.1%}  {rej_03:>6.1%}  {rej_05:>6.1%}  "
              f"{t_slot:>7.2f}")


def run_embedding_separation_sweep():
    k = 10
    n_per = 300
    n_train = k * n_per
    n_test = k * 50
    dim = 256
    pca_dim = 32

    noise_levels = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]

    print(f"\n\nEmbeddingCluster separation sweep: k={k}, d={dim}→{pca_dim}, "
          f"n_train={n_train}, n_test={n_test}")
    print()
    print(f"{'Noise':>6}  {'Conf Mean':>9}  {'Conf P10':>8}  "
          f"{'Conf P50':>8}  {'Conf P90':>8}  "
          f"{'Rej@0.1':>7}  {'Rej@0.3':>7}  {'Slot ms':>8}")
    print(f"{'-'*6}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*8}  "
          f"{'-'*7}  {'-'*7}  {'-'*8}")

    for noise in noise_levels:
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((k, dim)) * 2.0
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)

        data = []
        for i in range(k):
            pts = centers[i] + rng.standard_normal((n_per + 50, dim)) * noise
            data.append(pts)
        data_all = np.vstack(data).astype(np.float32)

        X_train = data_all[:n_train]
        X_test = data_all[n_train:n_train + n_test]

        model = EmbeddingCluster(
            n_clusters=k, reduction_dim=pca_dim, random_state=42, n_init=3
        ).fit(X_train)
        snap = model.snapshot()

        t0 = time.perf_counter()
        result = snap.assign_with_scores(X_test)
        t_slot = (time.perf_counter() - t0) * 1000

        confs = result.confidences_
        rej_01 = np.mean(confs < 0.1)
        rej_03 = np.mean(confs < 0.3)

        print(f"{noise:>6.1f}  {np.mean(confs):>9.3f}  "
              f"{np.percentile(confs, 10):>8.3f}  {np.median(confs):>8.3f}  "
              f"{np.percentile(confs, 90):>8.3f}  "
              f"{rej_01:>6.1%}  {rej_03:>6.1%}  {t_slot:>7.2f}")


def show_tiebreaker_behavior():
    """Demonstrate what happens at cluster boundaries."""
    print("\n\nTiebreaker behavior at cluster boundaries")
    print("=" * 60)

    # Two clusters clearly separated on x-axis
    X_train = np.array([
        [0, 0], [0.1, 0.1], [-0.1, 0.1], [0.1, -0.1], [-0.1, -0.1],
        [10, 0], [10.1, 0.1], [9.9, 0.1], [10.1, -0.1], [9.9, -0.1],
    ], dtype=np.float64)

    model = KMeans(n_clusters=2, random_state=42).fit(X_train)
    snap = model.snapshot()

    # Walk from cluster 0 to cluster 1 along x-axis
    steps = np.linspace(0, 10, 21)
    test_points = np.column_stack([steps, np.zeros(len(steps))])

    result = snap.assign_with_scores(test_points)

    print(f"\nWalking from centroid 0 (x=0) to centroid 1 (x=10):")
    print(f"{'x':>6}  {'Label':>5}  {'Confidence':>10}  {'Dist 1st':>10}  {'Dist 2nd':>10}  {'Status':>10}")
    print(f"{'-'*6}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for i, x in enumerate(steps):
        label = result.labels_[i]
        conf = result.confidences_[i]
        d1 = result.distances_[i]
        d2 = result.second_distances_[i]

        if conf < 0.05:
            status = "BOUNDARY"
        elif conf < 0.2:
            status = "weak"
        elif conf < 0.5:
            status = "moderate"
        else:
            status = "strong"

        print(f"{x:>6.1f}  {label:>5}  {conf:>10.4f}  {d1:>10.2f}  {d2:>10.2f}  {status:>10}")


if __name__ == "__main__":
    run_separation_sweep()
    run_embedding_separation_sweep()
    show_tiebreaker_behavior()
    print("\nDone.")
