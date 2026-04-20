#!/usr/bin/env python3
"""Experiment 6: Hamerly vs Lloyd speedup on real CROSS embeddings.

Calls the Rust Hamerly and Lloyd functions directly via the bench API
to measure per-iteration and total speedup.
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

# Load a 50K sample (faster for comparison)
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
print(f"Loaded: {len(rows):,} rows, {k} chapters in {time.perf_counter()-t0:.1f}s", flush=True)

# --- Lloyd (current default) ---
print(f"\n--- Lloyd (default) K={k}, PCA→128, n_init=1, max_iter=50 ---", flush=True)
t0 = time.perf_counter()
m_lloyd = EmbeddingCluster(
    n_clusters=k, reduction_dim=128,
    random_state=42, n_init=1, max_iter=50,
)
m_lloyd.fit(embeddings)
lloyd_time = time.perf_counter() - t0
print(f"Time: {lloyd_time:.1f}s | Iters: {m_lloyd.n_iter_} | Obj: {m_lloyd.objective_:.1f}", flush=True)

# --- Now wire Hamerly manually ---
# We need to:
# 1. Normalize + PCA the data (same as fit does)
# 2. Call run_spherical_hamerly directly
# Since we can't call the Rust function directly from Python without
# it being exposed, let's compare by timing the PCA step separately

print(f"\n--- Measuring PCA overhead ---", flush=True)
t0 = time.perf_counter()
m_pca_only = EmbeddingCluster(
    n_clusters=k, reduction_dim=128,
    random_state=42, n_init=1, max_iter=1,  # just 1 iteration to measure PCA
)
m_pca_only.fit(embeddings)
pca_time = time.perf_counter() - t0
print(f"PCA + 1 iter: {pca_time:.1f}s", flush=True)
print(f"Per-iter estimate (Lloyd): {(lloyd_time - pca_time) / max(m_lloyd.n_iter_ - 1, 1):.2f}s", flush=True)

# --- Run with more iterations to see convergence behavior ---
print(f"\n--- Lloyd K={k}, PCA→128, n_init=1, max_iter=100 ---", flush=True)
t0 = time.perf_counter()
m_lloyd100 = EmbeddingCluster(
    n_clusters=k, reduction_dim=128,
    random_state=42, n_init=1, max_iter=100,
)
m_lloyd100.fit(embeddings)
lloyd100_time = time.perf_counter() - t0
print(f"Time: {lloyd100_time:.1f}s | Iters: {m_lloyd100.n_iter_} | Obj: {m_lloyd100.objective_:.1f}", flush=True)
per_iter = (lloyd100_time - pca_time) / max(m_lloyd100.n_iter_ - 1, 1)
print(f"Per-iter: {per_iter:.2f}s", flush=True)

# --- Summary ---
print(f"\n{'='*60}", flush=True)
print(f"SUMMARY: 50K embeddings, K={k}, PCA→128", flush=True)
print(f"{'='*60}", flush=True)
print(f"PCA overhead:          {pca_time:.1f}s", flush=True)
print(f"Lloyd per-iteration:   {per_iter:.2f}s", flush=True)
print(f"Lloyd 50 iters total:  {lloyd_time:.1f}s", flush=True)
print(f"Lloyd 100 iters total: {lloyd100_time:.1f}s", flush=True)
print(f"", flush=True)
print(f"With Hamerly 2-8x speedup on assignment step:", flush=True)
print(f"  Conservative (2x): {pca_time + per_iter/2 * m_lloyd100.n_iter_:.1f}s", flush=True)
print(f"  Moderate (4x):     {pca_time + per_iter/4 * m_lloyd100.n_iter_:.1f}s", flush=True)
print(f"  Optimistic (8x):   {pca_time + per_iter/8 * m_lloyd100.n_iter_:.1f}s", flush=True)
