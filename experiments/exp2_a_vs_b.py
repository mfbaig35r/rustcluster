#!/usr/bin/env python3
"""Experiment 2: Does adding materials + product_use improve clustering?

Compares embedding A (description only) vs B (description + materials + use)
at K=98 (chapter level) with PCA→128.
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

def compute_purity_nmi(true_labels, pred_labels):
    n = len(true_labels)
    pred_labels = np.asarray(pred_labels)
    # Purity
    purity_sum = 0
    for cid in np.unique(pred_labels):
        mask = pred_labels == cid
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    purity = purity_sum / n
    # NMI
    pu = np.unique(pred_labels)
    tu = np.unique(true_labels)
    pc = np.array([np.sum(pred_labels == p) for p in pu])
    tc = np.array([np.sum(true_labels == t) for t in tu])
    hp = -np.sum((pc/n) * np.log(pc/n + 1e-30))
    ht = -np.sum((tc/n) * np.log(tc/n + 1e-30))
    mi = 0.0
    for p in pu:
        for t in tu:
            j = np.sum((pred_labels == p) & (true_labels == t))
            if j == 0: continue
            mi += (j/n) * np.log((j/n) / ((np.sum(pred_labels==p)/n) * (np.sum(true_labels==t)/n)) + 1e-30)
    return purity, 2*mi/(hp+ht+1e-30)

# Load both embeddings
print("Loading embeddings...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("""
    SELECT embedding_a, embedding_b, hts_chapter
    FROM cross_embeddings
    WHERE embedding_a IS NOT NULL AND embedding_b IS NOT NULL AND hts_chapter IS NOT NULL
""").fetchall()
db.close()

emb_a = np.array([r[0] for r in rows], dtype=np.float32)
emb_b = np.array([r[1] for r in rows], dtype=np.float32)
chapters = np.array([r[2] for r in rows])
k = len(np.unique(chapters))
print(f"Loaded: {len(rows):,} rows, {k} chapters in {time.perf_counter()-t0:.1f}s", flush=True)

# Run both
print(f"\n{'='*60}", flush=True)
print(f"{'Embedding':<30} {'Purity':>8} {'NMI':>8} {'Time':>8}", flush=True)
print(f"{'-'*60}", flush=True)

for name, emb in [("A (description only)", emb_a), ("B (desc+materials+use)", emb_b)]:
    print(f"  Running {name}...", flush=True)
    t0 = time.perf_counter()
    model = EmbeddingCluster(n_clusters=k, reduction_dim=128, random_state=42, n_init=3, max_iter=50)
    model.fit(emb)
    elapsed = time.perf_counter() - t0
    labels = np.array(model.labels_)
    purity, nmi = compute_purity_nmi(chapters, labels)
    print(f"  {name:<30} {purity:>8.4f} {nmi:>8.4f} {elapsed:>7.1f}s", flush=True)

print(f"{'='*60}", flush=True)
