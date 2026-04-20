#!/usr/bin/env python3
"""Experiment 4: EmbeddingCluster vs sklearn KMeans.

Uses 50K sample (sklearn may be slow/OOM on full 325K at 1536d).
Compares: EmbeddingCluster (PCA→128) vs sklearn on raw vs sklearn on normalized.
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
k = len(np.unique(chapters))
print(f"Loaded: {len(rows):,} rows, d={embeddings.shape[1]}, {k} chapters in {time.perf_counter()-t0:.1f}s", flush=True)

print(f"\n{'='*70}", flush=True)
print(f"{'Method':<35} {'Purity':>8} {'NMI':>8} {'Time':>10}", flush=True)
print(f"{'-'*70}", flush=True)

# 1. EmbeddingCluster (PCA→128)
print("  Running EmbeddingCluster (PCA→128)...", flush=True)
t0 = time.perf_counter()
model = EmbeddingCluster(n_clusters=k, reduction_dim=128, random_state=42, n_init=3, max_iter=50)
model.fit(embeddings)
elapsed = time.perf_counter() - t0
purity, nmi = compute_purity_nmi(chapters, np.array(model.labels_))
print(f"  {'EmbeddingCluster (PCA→128)':<35} {purity:>8.4f} {nmi:>8.4f} {elapsed:>9.1f}s", flush=True)

# 2. EmbeddingCluster (PCA→32, fast mode)
print("  Running EmbeddingCluster (PCA→32)...", flush=True)
t0 = time.perf_counter()
model2 = EmbeddingCluster(n_clusters=k, reduction_dim=32, random_state=42, n_init=3, max_iter=50)
model2.fit(embeddings)
elapsed = time.perf_counter() - t0
purity, nmi = compute_purity_nmi(chapters, np.array(model2.labels_))
print(f"  {'EmbeddingCluster (PCA→32)':<35} {purity:>8.4f} {nmi:>8.4f} {elapsed:>9.1f}s", flush=True)

# 3. sklearn KMeans on raw embeddings
try:
    from sklearn.cluster import KMeans as SklearnKMeans

    print("  Running sklearn KMeans (raw 1536d)...", flush=True)
    t0 = time.perf_counter()
    sk = SklearnKMeans(n_clusters=k, n_init=3, random_state=42, max_iter=50)
    sk.fit(embeddings)
    elapsed = time.perf_counter() - t0
    purity, nmi = compute_purity_nmi(chapters, sk.labels_)
    print(f"  {'sklearn KMeans (raw 1536d)':<35} {purity:>8.4f} {nmi:>8.4f} {elapsed:>9.1f}s", flush=True)

    # 4. sklearn KMeans on normalized embeddings
    print("  Running sklearn KMeans (normalized)...", flush=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    t0 = time.perf_counter()
    sk2 = SklearnKMeans(n_clusters=k, n_init=3, random_state=42, max_iter=50)
    sk2.fit(normalized)
    elapsed = time.perf_counter() - t0
    purity, nmi = compute_purity_nmi(chapters, sk2.labels_)
    print(f"  {'sklearn KMeans (normalized)':<35} {purity:>8.4f} {nmi:>8.4f} {elapsed:>9.1f}s", flush=True)

except ImportError:
    print("  sklearn not installed, skipping", flush=True)

print(f"{'='*70}", flush=True)
