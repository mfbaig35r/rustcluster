#!/usr/bin/env python3
"""Experiment 6b: Per-iteration cost on full 323K dataset."""
import sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

print("Loading...", flush=True)
import numpy as np
import duckdb
sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"

print("Loading full dataset...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("SELECT embedding_a FROM cross_embeddings WHERE embedding_a IS NOT NULL LIMIT 323308").fetchall()
db.close()
embeddings = np.array([r[0] for r in rows], dtype=np.float32)
n = len(embeddings)
print(f"Loaded: {n:,} x {embeddings.shape[1]} in {time.perf_counter()-t0:.1f}s", flush=True)

# 1 iteration to measure PCA cost
print("\nPCA + 1 iter...", flush=True)
t0 = time.perf_counter()
m1 = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=1)
m1.fit(embeddings)
t_pca = time.perf_counter() - t0
print(f"  PCA + 1 iter: {t_pca:.1f}s", flush=True)

# 10 iterations
print("10 iterations...", flush=True)
t0 = time.perf_counter()
m10 = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=10)
m10.fit(embeddings)
t_10 = time.perf_counter() - t0
per_iter = (t_10 - t_pca) / 9
print(f"  10 iters: {t_10:.1f}s | per-iter: {per_iter:.2f}s", flush=True)

# 50 iterations
print("50 iterations...", flush=True)
t0 = time.perf_counter()
m50 = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=50)
m50.fit(embeddings)
t_50 = time.perf_counter() - t0
per_iter_50 = (t_50 - t_pca) / 49
print(f"  50 iters: {t_50:.1f}s | per-iter: {per_iter_50:.2f}s | obj: {m50.objective_:.1f}", flush=True)

print(f"\n{'='*55}", flush=True)
print(f"SUMMARY: {n:,} embeddings, K=98, PCA→128", flush=True)
print(f"{'='*55}", flush=True)
print(f"PCA overhead:       {t_pca:.1f}s", flush=True)
print(f"Per-iteration:      {per_iter_50:.2f}s", flush=True)
print(f"K-means total (50): {t_50 - t_pca:.1f}s", flush=True)
print(f"", flush=True)
print(f"Hamerly projected speedup:", flush=True)
print(f"  2x: PCA + {(t_50-t_pca)/2:.0f}s = {t_pca + (t_50-t_pca)/2:.0f}s total", flush=True)
print(f"  4x: PCA + {(t_50-t_pca)/4:.0f}s = {t_pca + (t_50-t_pca)/4:.0f}s total", flush=True)
print(f"  8x: PCA + {(t_50-t_pca)/8:.0f}s = {t_pca + (t_50-t_pca)/8:.0f}s total", flush=True)
