#!/usr/bin/env python3
"""Experiment 8: faer PCA backend benchmark.

Measures PCA time with the new faer-backed reduction.rs vs exp6's baseline.
"""
import sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

print("Loading dependencies...", flush=True)
import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster, EmbeddingReducer

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"

# ---- Load data ----
print("Loading embeddings...", flush=True)
t0 = time.perf_counter()
db = duckdb.connect(str(DB_PATH), read_only=True)
rows = db.execute("""
    SELECT embedding_a FROM cross_embeddings
    WHERE embedding_a IS NOT NULL
    LIMIT 323308
""").fetchall()
db.close()
embeddings = np.array([r[0] for r in rows], dtype=np.float32)
n, d = embeddings.shape
load_time = time.perf_counter() - t0
print(f"Loaded: {n:,} x {d} in {load_time:.1f}s\n", flush=True)

# ---- PCA via EmbeddingReducer (faer backend) ----
print("=" * 60, flush=True)
print("PCA: EmbeddingReducer (faer backend)", flush=True)
print("=" * 60, flush=True)

t0 = time.perf_counter()
reducer = EmbeddingReducer(target_dim=128, method="pca", random_state=42)
X_reduced = reducer.fit_transform(embeddings)
reducer_time = time.perf_counter() - t0
print(f"  fit_transform: {reducer_time:.1f}s → {X_reduced.shape}", flush=True)
print(f"  norms check: {np.linalg.norm(X_reduced[:3], axis=1)}", flush=True)

# ---- Full pipeline: EmbeddingCluster with PCA ----
print(f"\n{'=' * 60}", flush=True)
print("Full pipeline: EmbeddingCluster (PCA + cluster)", flush=True)
print("=" * 60, flush=True)

# PCA + 1 iter to isolate PCA time
t0 = time.perf_counter()
m1 = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=1)
m1.fit(embeddings)
pca_plus_1 = time.perf_counter() - t0
print(f"  PCA + 1 iter: {pca_plus_1:.1f}s", flush=True)

# Full run
t0 = time.perf_counter()
m50 = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=50)
m50.fit(embeddings)
full_time = time.perf_counter() - t0
kmeans_time = full_time - pca_plus_1
print(f"  Full (PCA + 50 iters): {full_time:.1f}s", flush=True)
print(f"  K-means only: {kmeans_time:.1f}s", flush=True)
print(f"  Obj: {m50.objective_:.1f} | Iters: {m50.n_iter_}", flush=True)

# ---- Transform-only benchmark ----
print(f"\n{'=' * 60}", flush=True)
print("Transform only (projection, no eigendecomp)", flush=True)
print("=" * 60, flush=True)

t0 = time.perf_counter()
X_again = reducer.transform(embeddings)
transform_time = time.perf_counter() - t0
print(f"  transform: {transform_time:.1f}s", flush=True)

# ---- Summary ----
exp6_baseline = 615  # from exp6/exp7 results (pre-faer)

print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Exp6 baseline (pre-faer):    {exp6_baseline}s", flush=True)
print(f"  faer PCA (fit_transform):    {reducer_time:.0f}s", flush=True)
print(f"  faer full pipeline:          {full_time:.0f}s", flush=True)
print(f"  faer transform-only:         {transform_time:.1f}s", flush=True)
print(f"  K-means overhead:            {kmeans_time:.1f}s", flush=True)
print(f"", flush=True)
print(f"  PCA speedup vs baseline:     {exp6_baseline / reducer_time:.1f}x", flush=True)
print(f"  Pipeline speedup vs baseline: {exp6_baseline / full_time:.1f}x", flush=True)
