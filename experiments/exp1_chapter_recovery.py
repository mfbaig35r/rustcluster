#!/usr/bin/env python3
"""Experiment 1: Does EmbeddingCluster recover HTS chapter groupings?

Single experiment, immediate output, no buffering.
Tests embedding A at K=98 (true chapter count) with PCA→128.
"""

import sys
import time
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

print("Loading dependencies...", flush=True)
import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"

# ---- Load data ----
print(f"Loading from {DB_PATH}...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("""
    SELECT embedding_a, hts_chapter
    FROM cross_embeddings
    WHERE embedding_a IS NOT NULL AND hts_chapter IS NOT NULL
""").fetchall()
db.close()

embeddings = np.array([r[0] for r in rows], dtype=np.float32)
chapters = np.array([r[1] for r in rows])
load_time = time.perf_counter() - t0

n = len(embeddings)
d = embeddings.shape[1]
k = len(np.unique(chapters))
print(f"Loaded: {n:,} embeddings, d={d}, {k} chapters in {load_time:.1f}s", flush=True)

# ---- Cluster ----
print(f"\nClustering with EmbeddingCluster(K={k}, PCA→128)...", flush=True)
t0 = time.perf_counter()
model = EmbeddingCluster(
    n_clusters=k,
    reduction_dim=128,
    random_state=42,
    n_init=3,
    max_iter=50,
)
model.fit(embeddings)
cluster_time = time.perf_counter() - t0

labels = np.array(model.labels_)
print(f"Done in {cluster_time:.1f}s | {model.n_iter_} iterations | objective={model.objective_:.1f}", flush=True)

# ---- Metrics ----
print("\nComputing metrics...", flush=True)

# Purity
purity_sum = 0
for cid in np.unique(labels):
    mask = labels == cid
    true_in_cluster = chapters[mask]
    _, counts = np.unique(true_in_cluster, return_counts=True)
    purity_sum += counts.max()
purity = purity_sum / n

# NMI
pred_unique = np.unique(labels)
true_unique = np.unique(chapters)
pred_counts = np.array([np.sum(labels == p) for p in pred_unique])
true_counts = np.array([np.sum(chapters == t) for t in true_unique])
h_pred = -np.sum((pred_counts / n) * np.log(pred_counts / n + 1e-30))
h_true = -np.sum((true_counts / n) * np.log(true_counts / n + 1e-30))
mi = 0.0
for p in pred_unique:
    for t in true_unique:
        joint = np.sum((labels == p) & (chapters == t))
        if joint == 0:
            continue
        mi += (joint / n) * np.log((joint / n) / ((np.sum(labels == p) / n) * (np.sum(chapters == t) / n)) + 1e-30)
nmi = 2 * mi / (h_pred + h_true + 1e-30)

print(f"\n{'='*50}", flush=True)
print(f"RESULTS: Chapter-Level Recovery (K={k})", flush=True)
print(f"{'='*50}", flush=True)
print(f"  Embeddings:       {n:,} x {d}", flush=True)
print(f"  PCA:              1536 → 128", flush=True)
print(f"  Clusters:         {k}", flush=True)
print(f"  Purity:           {purity:.4f}", flush=True)
print(f"  NMI:              {nmi:.4f}", flush=True)
print(f"  Cluster time:     {cluster_time:.1f}s", flush=True)
print(f"  Resultant (mean): {np.mean(model.resultant_lengths_):.3f}", flush=True)
print(f"  Resultant (min):  {min(model.resultant_lengths_):.3f}", flush=True)
print(f"  Resultant (max):  {max(model.resultant_lengths_):.3f}", flush=True)
print(f"  Representatives:  {model.representatives_[:5]}...", flush=True)

# Top 5 largest clusters
cluster_sizes = np.bincount(labels)
top5 = np.argsort(cluster_sizes)[::-1][:5]
print(f"\n  Top 5 clusters by size:", flush=True)
for rank, cid in enumerate(top5):
    mask = labels == cid
    true_in = chapters[mask]
    vals, cnts = np.unique(true_in, return_counts=True)
    dominant = vals[cnts.argmax()]
    dom_pct = cnts.max() / mask.sum()
    print(f"    #{rank+1}: {mask.sum():>6,} points | dominant chapter={dominant} ({dom_pct:.1%})", flush=True)
