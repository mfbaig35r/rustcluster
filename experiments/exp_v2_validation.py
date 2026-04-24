"""Validate v2 features on real CROSS rulings data (323K x 1536).

Tests:
1. Adaptive thresholds: does rejection rate drop from ~85-98% to something usable?
2. vMF drift: does calibrated drift distinguish in-dist from random?
3. Mahalanobis: does it change anything on real embedding data?
4. Hierarchical: does two-level slotting improve sub-label purity?
"""

import os
import time
import numpy as np
import duckdb

from rustcluster import KMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster, HierarchicalSnapshot


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
    valid = pred_labels >= 0
    true_labels = np.asarray(true_labels)[valid]
    pred_labels = np.asarray(pred_labels)[valid]
    n = len(true_labels)
    if n == 0:
        return 0.0, 0
    purity_sum = 0
    for cid in np.unique(pred_labels):
        mask = pred_labels == cid
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / n, n


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def test_adaptive_thresholds(X_train, X_test, y_train, y_test):
    """Does adaptive rejection fix the 98.5% rejection rate?"""
    print_header("Test 1: Adaptive Thresholds vs Global Threshold")

    n_chapters = len(np.unique(y_train))
    model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model.fit(X_train)
    snap = model.snapshot()

    # Calibrate
    t0 = time.perf_counter()
    snap.calibrate(X_train)
    t_cal = time.perf_counter() - t0
    print(f"Calibration time: {t_cal:.1f}s")
    print()

    # Global threshold sweep
    print(f"{'Method':<25}  {'Threshold':<12}  {'Rejected':>10}  {'Rate':>8}  {'Purity':>8}  {'Kept':>8}")
    print(f"{'-'*25}  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    for thresh in [0.0, 0.05, 0.1, 0.2, 0.3]:
        r = snap.assign_with_scores(X_test, confidence_threshold=thresh)
        n_rej = sum(r.rejected_)
        pur, kept = compute_purity(y_test, r.labels_)
        print(f"{'Global':<25}  {thresh:<12.2f}  {n_rej:>10,}  {n_rej/len(y_test):>7.1%}  {pur:>8.4f}  {kept:>8,}")

    print()

    # Adaptive threshold sweep
    for pct in ["p5", "p10", "p25", "p50"]:
        r = snap.assign_with_scores(X_test, adaptive_threshold=True, adaptive_percentile=pct)
        n_rej = sum(r.rejected_)
        pur, kept = compute_purity(y_test, r.labels_)
        print(f"{'Adaptive':<25}  {pct:<12}  {n_rej:>10,}  {n_rej/len(y_test):>7.1%}  {pur:>8.4f}  {kept:>8,}")


def test_vmf_drift(X_train, X_test):
    """Does calibrated vMF drift distinguish in-dist from random?"""
    print_header("Test 2: vMF Drift Detection (Calibrated)")

    n_chapters = 98
    model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model.fit(X_train)
    snap = model.snapshot()
    snap.calibrate(X_train)

    # In-distribution
    report_in = snap.drift_report(X_test)
    kd_in = report_in.kappa_drift_
    dd_in = report_in.direction_drift_

    print(f"In-distribution (test set, {len(X_test):,} points):")
    print(f"  Kappa drift:     mean={np.mean(kd_in):.4f}, max={np.max(np.abs(kd_in)):.4f}")
    print(f"  Direction drift: mean={np.mean(dd_in):.4f}, max={np.max(dd_in):.4f}")
    print(f"  Global mean distance: {report_in.global_mean_distance_:.4f}")

    # Random embeddings
    rng = np.random.default_rng(42)
    X_rand = rng.standard_normal(X_test.shape).astype(np.float32)
    report_rand = snap.drift_report(X_rand)
    kd_rand = report_rand.kappa_drift_
    dd_rand = report_rand.direction_drift_

    print(f"\nRandom embeddings ({len(X_rand):,} points):")
    print(f"  Kappa drift:     mean={np.mean(kd_rand):.4f}, max={np.max(np.abs(kd_rand)):.4f}")
    print(f"  Direction drift: mean={np.mean(dd_rand):.4f}, max={np.max(dd_rand):.4f}")
    print(f"  Global mean distance: {report_rand.global_mean_distance_:.4f}")

    print(f"\nDrift discrimination:")
    print(f"  Kappa drift ratio (random/in-dist): {np.mean(np.abs(kd_rand)) / max(np.mean(np.abs(kd_in)), 1e-10):.1f}x")
    print(f"  Direction drift ratio: {np.mean(dd_rand) / max(np.mean(dd_in), 1e-10):.1f}x")


def test_mahalanobis(X_train, X_test, y_test):
    """Does Mahalanobis change anything on real embedding data?"""
    print_header("Test 3: Mahalanobis vs Voronoi on Real Data")

    n_chapters = 98
    model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model.fit(X_train)
    snap = model.snapshot()
    snap.calibrate(X_train)

    # Voronoi
    t0 = time.perf_counter()
    r_vor = snap.assign_with_scores(X_test)
    t_vor = time.perf_counter() - t0
    pur_vor, _ = compute_purity(y_test, r_vor.labels_)

    # Mahalanobis
    t0 = time.perf_counter()
    r_mah = snap.assign_with_scores(X_test, boundary_mode="mahalanobis")
    t_mah = time.perf_counter() - t0
    pur_mah, _ = compute_purity(y_test, r_mah.labels_)

    # Agreement
    agree = np.mean(np.array(r_vor.labels_) == np.array(r_mah.labels_))

    print(f"{'Mode':<15}  {'Purity':>8}  {'Time (ms)':>10}  {'Mean Conf':>10}")
    print(f"{'-'*15}  {'-'*8}  {'-'*10}  {'-'*10}")
    print(f"{'Voronoi':<15}  {pur_vor:>8.4f}  {t_vor*1000:>9.1f}  {np.mean(r_vor.confidences_):>10.4f}")
    print(f"{'Mahalanobis':<15}  {pur_mah:>8.4f}  {t_mah*1000:>9.1f}  {np.mean(r_mah.confidences_):>10.4f}")
    print(f"\nAgreement: {agree:.1%} of labels match")

    # Where do they disagree?
    disagree = np.array(r_vor.labels_) != np.array(r_mah.labels_)
    if disagree.sum() > 0:
        vor_conf_disagree = np.array(r_vor.confidences_)[disagree]
        mah_conf_disagree = np.array(r_mah.confidences_)[disagree]
        print(f"Disagreements: {disagree.sum():,} points")
        print(f"  Voronoi conf on disagreed:     mean={np.mean(vor_conf_disagree):.4f}")
        print(f"  Mahalanobis conf on disagreed:  mean={np.mean(mah_conf_disagree):.4f}")


def test_hierarchical(X_train, X_test, y_train_ch, y_test_ch, y_train_hd, y_test_hd):
    """Does hierarchical slotting improve heading-level purity?"""
    print_header("Test 4: Hierarchical Slotting (Chapter → Heading)")

    # Root: chapter-level clusters
    n_chapters = len(np.unique(y_train_ch))
    root_model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    root_model.fit(X_train)

    # Flat slotting (single level)
    flat_snap = root_model.snapshot()
    flat_labels = flat_snap.assign(X_test)
    flat_pur_ch, _ = compute_purity(y_test_ch, flat_labels)
    flat_pur_hd, _ = compute_purity(y_test_hd, flat_labels)

    print(f"Flat (k={n_chapters}):")
    print(f"  Chapter purity:  {flat_pur_ch:.4f}")
    print(f"  Heading purity:  {flat_pur_hd:.4f}")

    # Hierarchical: chapter → sub-clusters
    print(f"\nBuilding hierarchical snapshot (k_root={n_chapters}, k_sub=10)...")
    t0 = time.perf_counter()
    hier = HierarchicalSnapshot.build(
        X_train, root_model, n_sub_clusters=10, random_state=42, n_init=2, max_iter=30,
    )
    t_build = time.perf_counter() - t0
    print(f"  Build time: {t_build:.1f}s")
    print(f"  {hier}")

    # Hierarchical slotting
    t0 = time.perf_counter()
    root_labels, child_labels = hier.assign(X_test)
    t_slot = time.perf_counter() - t0

    # Composite label for purity: (root * 1000 + child) as unique cluster ID
    composite = root_labels * 1000 + np.where(child_labels >= 0, child_labels, 0)
    hier_pur_ch, _ = compute_purity(y_test_ch, composite)
    hier_pur_hd, _ = compute_purity(y_test_hd, composite)

    print(f"\nHierarchical (k_root={n_chapters}, k_sub=10):")
    print(f"  Chapter purity:  {hier_pur_ch:.4f}")
    print(f"  Heading purity:  {hier_pur_hd:.4f}")
    print(f"  Slot time:       {t_slot*1000:.1f} ms")

    print(f"\n{'Metric':<20}  {'Flat':>8}  {'Hierarchical':>14}  {'Delta':>8}")
    print(f"{'-'*20}  {'-'*8}  {'-'*14}  {'-'*8}")
    print(f"{'Chapter purity':<20}  {flat_pur_ch:>8.4f}  {hier_pur_ch:>14.4f}  {hier_pur_ch - flat_pur_ch:>+8.4f}")
    print(f"{'Heading purity':<20}  {flat_pur_hd:>8.4f}  {hier_pur_hd:>14.4f}  {hier_pur_hd - flat_pur_hd:>+8.4f}")


if __name__ == "__main__":
    print("Loading CROSS rulings embeddings...")
    t0 = time.perf_counter()
    embeddings, chapters, headings = load_data()
    t_load = time.perf_counter() - t0
    print(f"  Loaded {len(embeddings):,} x {embeddings.shape[1]} in {t_load:.1f}s")

    # 80/20 split
    rng = np.random.default_rng(42)
    n = len(embeddings)
    idx = rng.permutation(n)
    n_train = int(n * 0.8)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]

    X_train = embeddings[train_idx]
    X_test = embeddings[test_idx]
    y_train_ch = chapters[train_idx]
    y_test_ch = chapters[test_idx]
    y_train_hd = headings[train_idx]
    y_test_hd = headings[test_idx]

    print(f"  Split: {n_train:,} train / {n - n_train:,} test")

    test_adaptive_thresholds(X_train, X_test, y_train_ch, y_test_ch)
    test_vmf_drift(X_train, X_test)
    test_mahalanobis(X_train, X_test, y_test_ch)
    test_hierarchical(X_train, X_test, y_train_ch, y_test_ch, y_train_hd, y_test_hd)

    print(f"\n{'='*70}")
    print("  All v2 validation tests complete.")
    print(f"{'='*70}")
