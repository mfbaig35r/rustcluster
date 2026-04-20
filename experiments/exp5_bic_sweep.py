#!/usr/bin/env python3
"""Experiment 5: Does BIC find the right number of clusters?

Sweep K from 10 to 200 on a 50K sample, fit spherical K-means + vMF refinement,
record BIC at each K. Does the BIC minimum land near 98 (true chapter count)?
"""
import sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

print("Loading dependencies...", flush=True)
import numpy as np
import duckdb
sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"

def compute_purity(true_labels, pred_labels):
    n = len(true_labels)
    pred_labels = np.asarray(pred_labels)
    purity_sum = 0
    for cid in np.unique(pred_labels):
        mask = pred_labels == cid
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / n

print("Loading 50K sample...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("""
    SELECT embedding_a, hts_chapter FROM cross_embeddings
    WHERE embedding_a IS NOT NULL AND hts_chapter IS NOT NULL
    LIMIT 50000
""").fetchall()
db.close()

embeddings = np.array([r[0] for r in rows], dtype=np.float32)
chapters = np.array([r[1] for r in rows])
true_k = len(np.unique(chapters))
print(f"Loaded: {len(rows):,} rows, true K={true_k} in {time.perf_counter()-t0:.1f}s", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"{'K':>6} {'BIC':>14} {'Purity':>8} {'Objective':>12} {'Time':>8}", flush=True)
print(f"{'-'*70}", flush=True)

results = []
for k in [10, 20, 30, 50, 75, 98, 120, 150, 200]:
    print(f"  K={k}...", end=" ", flush=True)
    t0 = time.perf_counter()

    model = EmbeddingCluster(
        n_clusters=k, reduction_dim=128,
        random_state=42, n_init=2, max_iter=30,
    )
    model.fit(embeddings)
    fit_time = time.perf_counter() - t0

    model.refine_vmf()
    total_time = time.perf_counter() - t0

    labels = np.array(model.labels_)
    purity = compute_purity(chapters, labels)

    print(f"{k:>6} {model.bic_:>14.0f} {purity:>8.4f} {model.objective_:>12.1f} {total_time:>7.1f}s", flush=True)

    results.append({
        "k": k, "bic": model.bic_, "purity": purity,
        "objective": model.objective_, "time_s": total_time,
    })

best_bic = min(results, key=lambda r: r["bic"])
best_purity = max(results, key=lambda r: r["purity"])

print(f"{'='*70}", flush=True)
print(f"\n  BIC-optimal K = {best_bic['k']}  (true K = {true_k})", flush=True)
print(f"  Best purity K = {best_purity['k']}  (purity = {best_purity['purity']:.4f})", flush=True)
