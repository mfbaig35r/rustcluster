"""Snapshot slotting on real CROSS ruling embeddings (323K x 1536).

Tests the full production workflow:
1. Fit EmbeddingCluster on 80% of data
2. Snapshot the cluster state
3. Slot the held-out 20% incrementally
4. Measure: purity, NMI, confidence, timing
5. Save/load roundtrip
6. Drift detection (in-distribution vs shifted)

Requires: ~/Projects/hts-api/data/cross_embeddings.duckdb
"""

import os
import time
import tempfile
import json
import numpy as np
import duckdb

from rustcluster.experimental import EmbeddingCluster
from rustcluster import ClusterSnapshot


DB_PATH = os.path.expanduser("~/Projects/hts-api/data/cross_embeddings.duckdb")


def load_data(limit=None):
    db = duckdb.connect(DB_PATH, read_only=True)
    query = """
        SELECT embedding_a, hts_chapter, hts_heading
        FROM cross_embeddings
        WHERE embedding_a IS NOT NULL AND hts_chapter IS NOT NULL
    """
    if limit:
        query += f" LIMIT {limit}"
    rows = db.execute(query).fetchall()
    db.close()

    embeddings = np.array([r[0] for r in rows], dtype=np.float32)
    chapters = np.array([r[1] for r in rows])
    headings = np.array([r[2] for r in rows])
    return embeddings, chapters, headings


def compute_purity(true_labels, pred_labels):
    n = len(true_labels)
    purity_sum = 0
    for cid in np.unique(pred_labels[pred_labels >= 0]):
        mask = pred_labels == cid
        if mask.sum() == 0:
            continue
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    # Don't count rejected points in denominator
    valid = pred_labels >= 0
    return purity_sum / valid.sum() if valid.sum() > 0 else 0.0


def compute_nmi(true_labels, pred_labels):
    valid = pred_labels >= 0
    true_labels = true_labels[valid]
    pred_labels = pred_labels[valid]
    n = len(true_labels)
    if n == 0:
        return 0.0

    pred_unique = np.unique(pred_labels)
    true_unique = np.unique(true_labels)

    pred_counts = np.array([np.sum(pred_labels == p) for p in pred_unique])
    h_pred = -np.sum((pred_counts / n) * np.log(pred_counts / n + 1e-30))

    true_counts = np.array([np.sum(true_labels == t) for t in true_unique])
    h_true = -np.sum((true_counts / n) * np.log(true_counts / n + 1e-30))

    mi = 0.0
    for p in pred_unique:
        for t in true_unique:
            joint = np.sum((pred_labels == p) & (true_labels == t))
            if joint == 0:
                continue
            p_joint = joint / n
            mi += p_joint * np.log(p_joint / ((np.sum(pred_labels == p) / n) * (np.sum(true_labels == t) / n)) + 1e-30)

    return 2 * mi / (h_pred + h_true + 1e-30)


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def experiment_slot_vs_refit(embeddings, chapters, headings):
    """Core experiment: fit on 80%, slot 20%, compare to full refit."""
    print_header("Experiment: Slot vs Refit on CROSS Rulings")

    n = len(embeddings)
    n_train = int(n * 0.8)
    n_test = n - n_train
    n_chapters = len(np.unique(chapters))

    # Shuffle deterministically
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train = chapters[train_idx]
    y_test = chapters[test_idx]

    print(f"Data: {n:,} total → {n_train:,} train / {n_test:,} test")
    print(f"Chapters: {n_chapters}, Embedding dim: {embeddings.shape[1]}")

    # --- Fit on training set ---
    print(f"\n--- Fit EmbeddingCluster (K={n_chapters}, PCA→128) ---")
    t0 = time.perf_counter()
    model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model.fit(X_train)
    t_fit = time.perf_counter() - t0

    train_labels = np.array(model.labels_)
    train_purity = compute_purity(y_train, train_labels)
    train_nmi = compute_nmi(y_train, train_labels)
    print(f"  Time: {t_fit:.1f}s")
    print(f"  Train purity: {train_purity:.4f}")
    print(f"  Train NMI:    {train_nmi:.4f}")

    # --- Snapshot ---
    print(f"\n--- Snapshot ---")
    t0 = time.perf_counter()
    snap = model.snapshot()
    t_snap = time.perf_counter() - t0
    print(f"  {snap}")
    print(f"  Time: {t_snap*1000:.3f} ms")

    # --- Slot test set ---
    print(f"\n--- Slot {n_test:,} new points ---")
    t0 = time.perf_counter()
    result = snap.assign_with_scores(X_test)
    t_slot = time.perf_counter() - t0

    slot_labels = result.labels_
    slot_purity = compute_purity(y_test, slot_labels)
    slot_nmi = compute_nmi(y_test, slot_labels)

    print(f"  Time: {t_slot*1000:.1f} ms ({n_test/t_slot:,.0f} pts/sec)")
    print(f"  Purity: {slot_purity:.4f}")
    print(f"  NMI:    {slot_nmi:.4f}")

    # --- Confidence analysis ---
    confs = result.confidences_
    print(f"\n--- Confidence Distribution ---")
    print(f"  Mean:   {np.mean(confs):.4f}")
    print(f"  Median: {np.median(confs):.4f}")
    print(f"  P5:     {np.percentile(confs, 5):.4f}")
    print(f"  P25:    {np.percentile(confs, 25):.4f}")
    print(f"  P75:    {np.percentile(confs, 75):.4f}")
    print(f"  P95:    {np.percentile(confs, 95):.4f}")

    # --- Rejection sweep ---
    print(f"\n--- Rejection Sweep (confidence threshold) ---")
    print(f"  {'Threshold':>10}  {'Rejected':>10}  {'Rate':>8}  {'Purity (kept)':>14}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*14}")
    for thresh in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]:
        r = snap.assign_with_scores(X_test, confidence_threshold=thresh)
        n_rej = sum(r.rejected_)
        kept_labels = r.labels_
        pur = compute_purity(y_test, kept_labels)
        print(f"  {thresh:>10.2f}  {n_rej:>10,}  {n_rej/n_test:>7.1%}  {pur:>14.4f}")

    # --- Full refit comparison ---
    print(f"\n--- Full Refit on All Data (for comparison) ---")
    t0 = time.perf_counter()
    model_full = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model_full.fit(embeddings)
    t_refit = time.perf_counter() - t0

    full_labels = np.array(model_full.labels_)
    full_purity = compute_purity(chapters, full_labels)
    full_nmi = compute_nmi(chapters, full_labels)
    print(f"  Time: {t_refit:.1f}s")
    print(f"  Purity: {full_purity:.4f}")
    print(f"  NMI:    {full_nmi:.4f}")
    print(f"\n  Speedup (slot vs refit): {t_refit/t_slot:.0f}x")

    # --- Save/load ---
    print(f"\n--- Save/Load Roundtrip ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cross_snap")
        t0 = time.perf_counter()
        snap.save(path)
        t_save = time.perf_counter() - t0

        sizes = {f: os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)}
        total_size = sum(sizes.values())

        t0 = time.perf_counter()
        loaded = ClusterSnapshot.load(path)
        t_load = time.perf_counter() - t0

        loaded_labels = loaded.assign(X_test)
        match = np.array_equal(slot_labels, loaded_labels)

        print(f"  Save: {t_save*1000:.2f} ms")
        print(f"  Load: {t_load*1000:.2f} ms")
        print(f"  Size: {total_size/1024:.1f} KB")
        for fname, sz in sizes.items():
            print(f"    {fname}: {sz/1024:.1f} KB")
        print(f"  vs training data: {X_train.nbytes/1024/1024:.1f} MB")
        print(f"  Roundtrip match: {match}")

    # --- Drift detection ---
    print(f"\n--- Drift Detection ---")
    report_same = snap.drift_report(X_test)
    print(f"  In-distribution (test set):")
    print(f"    Global mean distance: {report_same.global_mean_distance_:.4f}")
    print(f"    Rejection rate:       {report_same.rejection_rate_:.1%}")

    # Simulate drift: random embeddings
    X_random = rng.standard_normal(X_test.shape).astype(np.float32)
    report_random = snap.drift_report(X_random)
    print(f"  Random embeddings:")
    print(f"    Global mean distance: {report_random.global_mean_distance_:.4f}")
    print(f"    Rejection rate:       {report_random.rejection_rate_:.1%}")

    # --- Summary ---
    print_header("Summary")
    print(f"  {'Metric':<25} {'Train':>10} {'Slot':>10} {'Full Refit':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
    print(f"  {'Purity':<25} {train_purity:>10.4f} {slot_purity:>10.4f} {full_purity:>12.4f}")
    print(f"  {'NMI':<25} {train_nmi:>10.4f} {slot_nmi:>10.4f} {full_nmi:>12.4f}")
    print(f"  {'Time':<25} {t_fit:>9.1f}s {t_slot*1000:>9.1f}ms {t_refit:>11.1f}s")
    print(f"  {'Throughput':<25} {'—':>10} {n_test/t_slot:>8,.0f}/s {'—':>12}")


if __name__ == "__main__":
    print("Loading CROSS rulings embeddings...")
    t0 = time.perf_counter()
    embeddings, chapters, headings = load_data()
    t_load = time.perf_counter() - t0
    print(f"  Loaded {len(embeddings):,} x {embeddings.shape[1]} in {t_load:.1f}s")

    experiment_slot_vs_refit(embeddings, chapters, headings)
    print("\nDone.")
