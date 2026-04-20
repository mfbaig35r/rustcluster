#!/usr/bin/env python3
"""Experiment 9: faer PCA quality comparison.

Compares clustering quality between faer-backed PCA and historical baselines.
Measures: purity, NMI, ARI, silhouette, vMF BIC, cluster stability across seeds.
"""
import sys, time
from pathlib import Path
sys.stdout.reconfigure(line_buffering=True)

print("Loading dependencies...", flush=True)
import numpy as np
import duckdb

sys.path.insert(0, str(Path(__file__).parent.parent))
from rustcluster.experimental import EmbeddingCluster, EmbeddingReducer
from rustcluster import KMeans

DB_PATH = Path.home() / "Projects/hts-api/data/cross_embeddings.duckdb"

# ---- Helpers ----

def compute_purity(labels, chapters, n):
    purity_sum = 0
    for cid in np.unique(labels):
        mask = labels == cid
        _, counts = np.unique(chapters[mask], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / n


def compute_nmi(labels, chapters, n):
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
            mi += (joint / n) * np.log(
                (joint / n) / ((np.sum(labels == p) / n) * (np.sum(chapters == t) / n)) + 1e-30
            )
    return 2 * mi / (h_pred + h_true + 1e-30)


def compute_ari(labels_a, labels_b, n):
    """Adjusted Rand Index between two clusterings."""
    from collections import Counter
    # Contingency table
    pairs_a = Counter()
    pairs_b = Counter()
    pairs_ab = Counter()
    for i in range(n):
        pairs_a[labels_a[i]] += 1
        pairs_b[labels_b[i]] += 1
        pairs_ab[(labels_a[i], labels_b[i])] += 1

    # Compute ARI from contingency
    sum_comb_ab = sum(c * (c - 1) // 2 for c in pairs_ab.values())
    sum_comb_a = sum(c * (c - 1) // 2 for c in pairs_a.values())
    sum_comb_b = sum(c * (c - 1) // 2 for c in pairs_b.values())
    total_comb = n * (n - 1) // 2

    expected = sum_comb_a * sum_comb_b / total_comb if total_comb > 0 else 0
    max_val = (sum_comb_a + sum_comb_b) / 2
    if max_val == expected:
        return 1.0
    return (sum_comb_ab - expected) / (max_val - expected)


def top_clusters(labels, chapters, n_top=5):
    cluster_sizes = np.bincount(labels)
    top = np.argsort(cluster_sizes)[::-1][:n_top]
    results = []
    for cid in top:
        mask = labels == cid
        vals, cnts = np.unique(chapters[mask], return_counts=True)
        dominant = vals[cnts.argmax()]
        purity = cnts.max() / mask.sum()
        results.append((cid, mask.sum(), dominant, purity))
    return results


# ---- Load data ----
print("Loading embeddings with chapter labels...", flush=True)
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
n, d = embeddings.shape
k = len(np.unique(chapters))
load_time = time.perf_counter() - t0
print(f"Loaded: {n:,} x {d}, {k} chapters in {load_time:.1f}s\n", flush=True)

# ============================================================
# A. Main comparison: faer PCA (current) vs historical baseline
# ============================================================
print("=" * 60, flush=True)
print("A. CHAPTER RECOVERY: faer PCA backend", flush=True)
print(f"   Config: K={k}, PCA→128, n_init=3, max_iter=50, seed=42", flush=True)
print("=" * 60, flush=True)

t0 = time.perf_counter()
model = EmbeddingCluster(
    n_clusters=k, reduction_dim=128, random_state=42, n_init=3, max_iter=50
)
model.fit(embeddings)
fit_time = time.perf_counter() - t0

labels = np.array(model.labels_)
purity = compute_purity(labels, chapters, n)
nmi = compute_nmi(labels, chapters, n)

print(f"  Time:             {fit_time:.1f}s", flush=True)
print(f"  Purity:           {purity:.4f}", flush=True)
print(f"  NMI:              {nmi:.4f}", flush=True)
print(f"  Objective:        {model.objective_:.1f}", flush=True)
print(f"  Iterations:       {model.n_iter_}", flush=True)
print(f"  Resultant (mean): {np.mean(model.resultant_lengths_):.3f}", flush=True)
print(f"  Resultant (min):  {min(model.resultant_lengths_):.3f}", flush=True)
print(f"  Resultant (max):  {max(model.resultant_lengths_):.3f}", flush=True)

print(f"\n  Top 5 clusters:", flush=True)
for rank, (cid, size, dom, pur) in enumerate(top_clusters(labels, chapters)):
    print(f"    #{rank+1}: {size:>6,} pts | chapter={dom} ({pur:.1%})", flush=True)

# Historical baseline from exp1
print(f"\n  vs Exp1 baseline (pre-faer, same config):", flush=True)
print(f"    Purity:    0.5864 → {purity:.4f} (Δ={purity - 0.5864:+.4f})", flush=True)
print(f"    NMI:       0.5384 → {nmi:.4f} (Δ={nmi - 0.5384:+.4f})", flush=True)
print(f"    Objective: 214308 → {model.objective_:.0f} (Δ={model.objective_ - 214308:+.0f})", flush=True)
print(f"    Time:      633s   → {fit_time:.0f}s ({633/fit_time:.1f}x faster)", flush=True)

# ============================================================
# B. vMF refinement quality
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print("B. vMF REFINEMENT", flush=True)
print("=" * 60, flush=True)

model.refine_vmf()
print(f"  BIC:              {model.bic_:.1f}", flush=True)
print(f"  κ (mean):         {np.mean(model.concentrations_):.1f}", flush=True)
print(f"  κ (min):          {min(model.concentrations_):.1f}", flush=True)
print(f"  κ (max):          {max(model.concentrations_):.1f}", flush=True)

# Check probability quality
probs = model.probabilities_
max_probs = probs.max(axis=1)
print(f"  P(max) mean:      {max_probs.mean():.3f}", flush=True)
print(f"  P(max) > 0.9:     {(max_probs > 0.9).sum():,} ({(max_probs > 0.9).mean():.1%})", flush=True)
print(f"  P(max) < 0.5:     {(max_probs < 0.5).sum():,} ({(max_probs < 0.5).mean():.1%})", flush=True)

# ============================================================
# C. Seed stability: how much do results vary across seeds?
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print("C. SEED STABILITY (n_init=1, 5 seeds)", flush=True)
print("=" * 60, flush=True)

# First, reduce once (faer PCA), then cluster with different seeds
reducer = EmbeddingReducer(target_dim=128, method="pca", random_state=42)
X_reduced = reducer.fit_transform(embeddings)

seed_results = []
seed_labels = []
for seed in [0, 42, 123, 456, 789]:
    t0 = time.perf_counter()
    m = EmbeddingCluster(
        n_clusters=k, reduction_dim=None, random_state=seed, n_init=1, max_iter=50
    )
    m.fit(X_reduced)
    elapsed = time.perf_counter() - t0
    labs = np.array(m.labels_)
    pur = compute_purity(labs, chapters, n)
    nm = compute_nmi(labs, chapters, n)
    seed_results.append((seed, pur, nm, m.objective_, elapsed))
    seed_labels.append(labs)
    print(f"  seed={seed:3d}: purity={pur:.4f}  NMI={nm:.4f}  obj={m.objective_:.0f}  time={elapsed:.1f}s", flush=True)

purities = [r[1] for r in seed_results]
nmis = [r[2] for r in seed_results]
objs = [r[3] for r in seed_results]
print(f"\n  Purity: {np.mean(purities):.4f} ± {np.std(purities):.4f} (range: {min(purities):.4f} - {max(purities):.4f})", flush=True)
print(f"  NMI:    {np.mean(nmis):.4f} ± {np.std(nmis):.4f} (range: {min(nmis):.4f} - {max(nmis):.4f})", flush=True)
print(f"  Obj:    {np.mean(objs):.0f} ± {np.std(objs):.0f}", flush=True)

# ARI between seed pairs
print(f"\n  Pairwise ARI (label agreement between seeds):", flush=True)
ari_values = []
for i in range(len(seed_labels)):
    for j in range(i + 1, len(seed_labels)):
        ari = compute_ari(seed_labels[i], seed_labels[j], n)
        ari_values.append(ari)
        si, sj = seed_results[i][0], seed_results[j][0]
        print(f"    seed {si:3d} vs {sj:3d}: ARI={ari:.4f}", flush=True)
print(f"  Mean pairwise ARI: {np.mean(ari_values):.4f}", flush=True)

# ============================================================
# D. Matryoshka vs PCA quality
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print("D. MATRYOSHKA vs PCA QUALITY", flush=True)
print("=" * 60, flush=True)

reducer_mat = EmbeddingReducer(target_dim=128, method="matryoshka")
X_mat = reducer_mat.fit_transform(embeddings)

m_mat = EmbeddingCluster(
    n_clusters=k, reduction_dim=None, random_state=42, n_init=3, max_iter=50
)
m_mat.fit(X_mat)
labs_mat = np.array(m_mat.labels_)
pur_mat = compute_purity(labs_mat, chapters, n)
nmi_mat = compute_nmi(labs_mat, chapters, n)

print(f"  PCA→128:        purity={purity:.4f}  NMI={nmi:.4f}  obj={model.objective_:.0f}", flush=True)
print(f"  Matryoshka→128: purity={pur_mat:.4f}  NMI={nmi_mat:.4f}  obj={m_mat.objective_:.0f}", flush=True)
ari_pca_mat = compute_ari(labels, labs_mat, n)
print(f"  ARI(PCA vs Mat): {ari_pca_mat:.4f}", flush=True)

# ============================================================
# E. Summary
# ============================================================
print(f"\n{'=' * 60}", flush=True)
print("SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"", flush=True)
print(f"  Chapter recovery (K={k}, n_init=3):", flush=True)
print(f"    {'Metric':<20s} {'Pre-faer':>10s} {'faer':>10s} {'Delta':>10s}", flush=True)
print(f"    {'Purity':<20s} {'0.5864':>10s} {purity:>10.4f} {purity-0.5864:>+10.4f}", flush=True)
print(f"    {'NMI':<20s} {'0.5384':>10s} {nmi:>10.4f} {nmi-0.5384:>+10.4f}", flush=True)
print(f"    {'Objective':<20s} {'214308':>10s} {model.objective_:>10.0f} {model.objective_-214308:>+10.0f}", flush=True)
print(f"    {'Time':<20s} {'633s':>10s} {fit_time:>9.0f}s {f'{633/fit_time:.0f}x faster':>10s}", flush=True)
print(f"", flush=True)
print(f"  Seed stability (n_init=1, 5 seeds):", flush=True)
print(f"    Purity: {np.mean(purities):.4f} ± {np.std(purities):.4f}", flush=True)
print(f"    NMI:    {np.mean(nmis):.4f} ± {np.std(nmis):.4f}", flush=True)
print(f"    Mean pairwise ARI: {np.mean(ari_values):.4f}", flush=True)
print(f"", flush=True)
print(f"  PCA vs Matryoshka:", flush=True)
print(f"    PCA purity={purity:.4f}, Matryoshka purity={pur_mat:.4f}", flush=True)
print(f"    ARI between them: {ari_pca_mat:.4f}", flush=True)
