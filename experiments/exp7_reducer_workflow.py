#!/usr/bin/env python3
"""Experiment 7: EmbeddingReducer workflow — pay PCA once, cluster for free.

Measures:
  A. Baseline: EmbeddingCluster end-to-end (PCA + cluster every time)
  B. EmbeddingReducer.fit_transform + save (first-time cost)
  C. EmbeddingReducer.load + transform (cached PCA, new data)
  D. reduced_data_ property → re-cluster (zero PCA cost)
  E. Multiple clustering configs on cached reduced data
"""
import sys, time, tempfile, os
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

print("Loading dependencies...", flush=True)
import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster, EmbeddingReducer
from rustcluster import KMeans, DBSCAN

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"
TMPDIR = tempfile.mkdtemp(prefix="exp7_")

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

# ---- A. Baseline: EmbeddingCluster end-to-end ----
print("=" * 60, flush=True)
print("A. BASELINE: EmbeddingCluster end-to-end (PCA + cluster)", flush=True)
print("=" * 60, flush=True)
t0 = time.perf_counter()
baseline = EmbeddingCluster(n_clusters=98, reduction_dim=128, random_state=42, n_init=1, max_iter=50)
baseline.fit(embeddings)
baseline_time = time.perf_counter() - t0
print(f"  Time: {baseline_time:.1f}s | Iters: {baseline.n_iter_} | Obj: {baseline.objective_:.1f}", flush=True)

# Check reduced_data_ is available
rd = baseline.reduced_data_
print(f"  reduced_data_ shape: {rd.shape}, dtype: {rd.dtype}", flush=True)
print(f"  reduced_data_ norms [0:3]: {np.linalg.norm(rd[:3], axis=1)}", flush=True)

# ---- B. EmbeddingReducer.fit_transform + save ----
print(f"\n{'=' * 60}", flush=True)
print("B. EmbeddingReducer: fit_transform + save", flush=True)
print("=" * 60, flush=True)
reducer_path = os.path.join(TMPDIR, "cross_pca_128.bin")
npy_path = os.path.join(TMPDIR, "cross_reduced_128.npy")

t0 = time.perf_counter()
reducer = EmbeddingReducer(target_dim=128, method="pca", random_state=42)
X_reduced = reducer.fit_transform(embeddings)
fit_transform_time = time.perf_counter() - t0
print(f"  fit_transform: {fit_transform_time:.1f}s → {X_reduced.shape}", flush=True)

t0 = time.perf_counter()
reducer.save(reducer_path)
save_time = time.perf_counter() - t0
file_size = os.path.getsize(reducer_path)
print(f"  save: {save_time:.3f}s ({file_size / 1024:.0f} KB)", flush=True)

t0 = time.perf_counter()
np.save(npy_path, X_reduced)
npy_save_time = time.perf_counter() - t0
npy_size = os.path.getsize(npy_path)
print(f"  np.save reduced data: {npy_save_time:.2f}s ({npy_size / 1024 / 1024:.0f} MB)", flush=True)

# ---- C. EmbeddingReducer.load + transform (simulates cold start) ----
print(f"\n{'=' * 60}", flush=True)
print("C. EmbeddingReducer: load + transform (cached PCA model)", flush=True)
print("=" * 60, flush=True)

t0 = time.perf_counter()
loaded_reducer = EmbeddingReducer.load(reducer_path)
load_reducer_time = time.perf_counter() - t0
print(f"  load model: {load_reducer_time:.3f}s", flush=True)

t0 = time.perf_counter()
X_reduced_2 = loaded_reducer.transform(embeddings)
transform_time = time.perf_counter() - t0
print(f"  transform: {transform_time:.1f}s → {X_reduced_2.shape}", flush=True)
print(f"  (this is the projection step only — no eigendecomp)", flush=True)

# Verify identical output
diff = np.max(np.abs(X_reduced - X_reduced_2))
print(f"  max diff vs original: {diff:.2e}", flush=True)

# ---- D. Load pre-reduced data + cluster (zero PCA) ----
print(f"\n{'=' * 60}", flush=True)
print("D. Load pre-reduced numpy + cluster (ZERO PCA cost)", flush=True)
print("=" * 60, flush=True)

t0 = time.perf_counter()
X_cached = np.load(npy_path)
npy_load_time = time.perf_counter() - t0
print(f"  np.load: {npy_load_time:.2f}s → {X_cached.shape}", flush=True)

t0 = time.perf_counter()
fast_model = EmbeddingCluster(n_clusters=98, reduction_dim=None, random_state=42, n_init=1, max_iter=50)
fast_model.fit(X_cached)
cluster_time = time.perf_counter() - t0
print(f"  cluster: {cluster_time:.1f}s | Iters: {fast_model.n_iter_} | Obj: {fast_model.objective_:.1f}", flush=True)
print(f"  TOTAL (load + cluster): {npy_load_time + cluster_time:.1f}s", flush=True)

# ---- E. Multiple configs on cached data ----
print(f"\n{'=' * 60}", flush=True)
print("E. Iterate on clustering configs (all on cached reduced data)", flush=True)
print("=" * 60, flush=True)

configs = [
    ("EmbeddingCluster K=50",  lambda: EmbeddingCluster(n_clusters=50, reduction_dim=None, random_state=42, n_init=1, max_iter=50).fit(X_cached)),
    ("EmbeddingCluster K=98",  lambda: EmbeddingCluster(n_clusters=98, reduction_dim=None, random_state=42, n_init=1, max_iter=50).fit(X_cached)),
    ("EmbeddingCluster K=200", lambda: EmbeddingCluster(n_clusters=200, reduction_dim=None, random_state=42, n_init=1, max_iter=50).fit(X_cached)),
    ("KMeans K=98 cosine",     lambda: KMeans(n_clusters=98, metric="cosine", random_state=42, n_init=1, max_iter=50).fit(X_cached)),
    ("EmbeddingCluster K=98 n_init=5", lambda: EmbeddingCluster(n_clusters=98, reduction_dim=None, random_state=42, n_init=5, max_iter=50).fit(X_cached)),
]

total_configs_time = 0
for name, fn in configs:
    t0 = time.perf_counter()
    m = fn()
    elapsed = time.perf_counter() - t0
    total_configs_time += elapsed
    obj = m.objective_ if hasattr(m, 'objective_') else m.inertia_
    iters = m.n_iter_ if hasattr(m, 'n_iter_') else "?"
    print(f"  {name:40s} {elapsed:5.1f}s | iters={iters} | obj={obj:.1f}", flush=True)

print(f"  {'TOTAL (5 configs)':40s} {total_configs_time:5.1f}s", flush=True)

# ---- Summary ----
print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"  Baseline (PCA + cluster every run):   {baseline_time:.0f}s", flush=True)
print(f"  First run (fit_transform + save):     {fit_transform_time:.0f}s", flush=True)
print(f"  Subsequent run (load npy + cluster):  {npy_load_time + cluster_time:.1f}s", flush=True)
print(f"  5 config iterations (all cached):     {total_configs_time:.1f}s", flush=True)
print(f"", flush=True)
print(f"  Speedup (2nd run vs baseline):        {baseline_time / (npy_load_time + cluster_time):.0f}x", flush=True)
print(f"  Speedup (5 configs vs 5x baseline):   {baseline_time * 5 / total_configs_time:.0f}x", flush=True)
print(f"", flush=True)
print(f"  Reducer model file: {file_size / 1024:.0f} KB", flush=True)
print(f"  Reduced data file:  {npy_size / 1024 / 1024:.0f} MB", flush=True)
print(f"  Temp dir: {TMPDIR}", flush=True)
