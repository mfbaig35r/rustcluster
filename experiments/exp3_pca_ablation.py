#!/usr/bin/env python3
"""Experiment 3: What PCA target dimension gives the best quality/speed tradeoff?

Tests: None (raw 1536), 256, 128, 64, 32 on embedding A at K=98.
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
    purity_sum = 0
    for cid in np.unique(pred_labels):
        mask = pred_labels == cid
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    purity = purity_sum / n
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

print("Loading embeddings...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("""
    SELECT embedding_a, hts_chapter FROM cross_embeddings
    WHERE embedding_a IS NOT NULL AND hts_chapter IS NOT NULL
""").fetchall()
db.close()

embeddings = np.array([r[0] for r in rows], dtype=np.float32)
chapters = np.array([r[1] for r in rows])
k = len(np.unique(chapters))
print(f"Loaded: {len(rows):,} rows, {k} chapters in {time.perf_counter()-t0:.1f}s", flush=True)

print(f"\n{'='*65}", flush=True)
print(f"{'PCA dim':>10} {'Purity':>8} {'NMI':>8} {'Time':>10} {'Iters':>6}", flush=True)
print(f"{'-'*65}", flush=True)

for dim in [32, 64, 128, 256]:
    label = str(dim)
    print(f"  Running PCA→{dim}...", flush=True)
    t0 = time.perf_counter()
    model = EmbeddingCluster(n_clusters=k, reduction_dim=dim, random_state=42, n_init=3, max_iter=50)
    model.fit(embeddings)
    elapsed = time.perf_counter() - t0
    labels = np.array(model.labels_)
    purity, nmi = compute_purity_nmi(chapters, labels)
    print(f"  {label:>10} {purity:>8.4f} {nmi:>8.4f} {elapsed:>9.1f}s {model.n_iter_:>6}", flush=True)

print(f"{'='*65}", flush=True)
