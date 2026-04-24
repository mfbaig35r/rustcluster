# Databricks notebook source
# MAGIC %md
# MAGIC # Cluster Slotting Experiment — Supplier Embeddings
# MAGIC
# MAGIC Tests the snapshot/slotting feature on 312K real supplier embeddings.
# MAGIC
# MAGIC **Workflow:**
# MAGIC 1. Load embeddings from Delta
# MAGIC 2. 80/20 train/test split
# MAGIC 3. Fit EmbeddingCluster on train
# MAGIC 4. Snapshot → slot test set
# MAGIC 5. Measure: purity, confidence, timing, save/load, drift
# MAGIC 6. Rejection sweep: confidence threshold vs purity tradeoff
# MAGIC 7. Compare to full refit

# COMMAND ----------

# MAGIC %pip install rustcluster --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

import time
PIPELINE_START = time.perf_counter()

config = {
    "input_table": "global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_selection_enrichment_embeddings",
    "metadata_columns": ["parent_supplier_id", "supplier_name", "industry", "commodity", "sub_commodity"],
    "reduction_method": "matryoshka",
    "reduction_dim": 128,
    "n_clusters": 20,
    "n_init": 3,
    "max_iter": 100,
    "random_state": 42,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Embeddings

# COMMAND ----------

import numpy as np
import pandas as pd
import gc
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("snapshot_exp")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

t0 = time.perf_counter()

df = spark.read.table(config["input_table"]).filter("embedding IS NOT NULL")
pdf = df.select(*config["metadata_columns"]).toPandas()
n_total = len(pdf)
log.info(f"Metadata: {n_total:,} rows")

# Stream embeddings
log.info("Streaming embeddings from Delta...")
emb_list = []
for row in df.select("embedding").toLocalIterator():
    emb_list.append(np.array(row.embedding, dtype=np.float32))

embeddings = np.vstack(emb_list)
del emb_list, df; gc.collect()

n, d = embeddings.shape
load_time = time.perf_counter() - t0
log.info(f"Loaded: {n:,} x {d} in {load_time:.1f}s ({embeddings.nbytes / 1e9:.1f} GB)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/Test Split + Reduce

# COMMAND ----------

rng = np.random.default_rng(config["random_state"])
idx = rng.permutation(n)
n_train = int(n * 0.8)
n_test = n - n_train

train_idx = idx[:n_train]
test_idx = idx[n_train:]

target = config["reduction_dim"]

# Matryoshka truncation (instant for text-embedding-3-small)
X_all = embeddings[:, :target].copy().astype(np.float64)
del embeddings; gc.collect()

X_train = X_all[train_idx]
X_test = X_all[test_idx]
y_train_commodity = pdf["commodity"].values[train_idx]
y_test_commodity = pdf["commodity"].values[test_idx]
y_train_sub = pdf["sub_commodity"].values[train_idx]
y_test_sub = pdf["sub_commodity"].values[test_idx]

log.info(f"Split: {n_train:,} train / {n_test:,} test, reduced to {target}d")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fit EmbeddingCluster

# COMMAND ----------

from rustcluster.experimental import EmbeddingCluster

t0 = time.perf_counter()
model = EmbeddingCluster(
    n_clusters=config["n_clusters"],
    reduction_dim=None,  # already reduced via matryoshka
    random_state=config["random_state"],
    n_init=config["n_init"],
    max_iter=config["max_iter"],
)
model.fit(X_train)
t_fit = time.perf_counter() - t0

train_labels = np.array(model.labels_)
log.info(f"Fit: K={config['n_clusters']}, {t_fit:.1f}s, objective={model.objective_:.1f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Snapshot + Slot Test Set

# COMMAND ----------

t0 = time.perf_counter()
snap = model.snapshot()
t_snap = time.perf_counter() - t0

print(f"Snapshot: {snap}")
print(f"  Created in {t_snap*1000:.3f} ms")

t0 = time.perf_counter()
result = snap.assign_with_scores(X_test)
t_slot = time.perf_counter() - t0

slot_labels = result.labels_
confs = result.confidences_

print(f"\nSlotted {n_test:,} points in {t_slot*1000:.1f} ms ({n_test/t_slot:,.0f} pts/sec)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Purity (Commodity + Sub-Commodity)

# COMMAND ----------

def compute_purity(true_labels, pred_labels):
    """Within each predicted cluster, what fraction belongs to the dominant true label?"""
    valid = pred_labels >= 0
    true_labels = np.asarray(true_labels)[valid]
    pred_labels = np.asarray(pred_labels)[valid]
    n = len(true_labels)
    if n == 0:
        return 0.0
    purity_sum = 0
    for cid in np.unique(pred_labels):
        mask = pred_labels == cid
        _, counts = np.unique(true_labels[mask], return_counts=True)
        purity_sum += counts.max()
    return purity_sum / n


def compute_nmi(true_labels, pred_labels):
    valid = pred_labels >= 0
    true_labels = np.asarray(true_labels)[valid]
    pred_labels = np.asarray(pred_labels)[valid]
    n = len(true_labels)
    if n == 0:
        return 0.0
    pred_unique = np.unique(pred_labels)
    true_unique = np.unique(true_labels)
    pred_counts = np.array([np.sum(pred_labels == p) for p in pred_unique])
    h_pred = -np.sum((pred_counts / n) * np.log(pred_counts / n + 1e-30))
    true_counts = np.array([np.sum(true_labels == t) for t in true_unique])
    h_true = -np.sum((true_counts / n) * np.log(true_counts / n + 1e-30))
    mi = 0.0
    for p in pred_unique:
        for t in true_unique:
            joint = np.sum((pred_labels == p) & (true_labels == t))
            if joint == 0:
                continue
            p_joint = joint / n
            mi += p_joint * np.log(p_joint / ((np.sum(pred_labels == p) / n) * (np.sum(true_labels == t) / n)) + 1e-30)
    return 2 * mi / (h_pred + h_true + 1e-30)


# Train metrics
train_pur_comm = compute_purity(y_train_commodity, train_labels)
train_nmi_comm = compute_nmi(y_train_commodity, train_labels)

# Slot metrics
slot_pur_comm = compute_purity(y_test_commodity, slot_labels)
slot_nmi_comm = compute_nmi(y_test_commodity, slot_labels)
slot_pur_sub = compute_purity(y_test_sub, slot_labels)

print(f"{'Metric':<30} {'Train':>10} {'Slotted':>10}")
print(f"{'-'*30} {'-'*10} {'-'*10}")
print(f"{'Commodity purity':<30} {train_pur_comm:>10.4f} {slot_pur_comm:>10.4f}")
print(f"{'Commodity NMI':<30} {train_nmi_comm:>10.4f} {slot_nmi_comm:>10.4f}")
print(f"{'Sub-commodity purity':<30} {'—':>10} {slot_pur_sub:>10.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Confidence Distribution

# COMMAND ----------

print(f"Confidence scores ({n_test:,} slotted points):")
print(f"  Mean:   {np.mean(confs):.4f}")
print(f"  Median: {np.median(confs):.4f}")
print(f"  P5:     {np.percentile(confs, 5):.4f}")
print(f"  P25:    {np.percentile(confs, 25):.4f}")
print(f"  P75:    {np.percentile(confs, 75):.4f}")
print(f"  P95:    {np.percentile(confs, 95):.4f}")

# Histogram
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(confs, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution — Slotted Supplier Embeddings")
    ax.axvline(x=0.1, color="orange", linestyle="--", label="threshold=0.1")
    ax.axvline(x=0.3, color="red", linestyle="--", label="threshold=0.3")
    ax.legend()
    display(fig)
except Exception:
    print("  (matplotlib not available for histogram)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Rejection Sweep

# COMMAND ----------

print(f"{'Threshold':>10}  {'Rejected':>10}  {'Rate':>8}  {'Commodity Purity':>16}  {'Sub-Commodity Purity':>20}")
print(f"{'-'*10}  {'-'*10}  {'-'*8}  {'-'*16}  {'-'*20}")

for thresh in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    r = snap.assign_with_scores(X_test, confidence_threshold=thresh)
    n_rej = sum(r.rejected_)
    pur_comm = compute_purity(y_test_commodity, r.labels_)
    pur_sub = compute_purity(y_test_sub, r.labels_)
    print(f"{thresh:>10.2f}  {n_rej:>10,}  {n_rej/n_test:>7.1%}  {pur_comm:>16.4f}  {pur_sub:>20.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Slot Training Data Back (Fidelity Check)

# COMMAND ----------

t0 = time.perf_counter()
train_slotted = snap.assign(X_train)
t_train_slot = time.perf_counter() - t0

match = np.sum(train_slotted == train_labels)
print(f"Training fidelity: {match:,} / {n_train:,} ({match/n_train:.2%})")
print(f"Time: {t_train_slot*1000:.1f} ms")

mismatches = np.where(train_slotted != train_labels)[0]
if len(mismatches) > 0:
    mis_confs = snap.assign_with_scores(X_train[mismatches]).confidences_
    print(f"Mismatches: {len(mismatches)} (max confidence: {np.max(mis_confs):.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Full Refit Comparison

# COMMAND ----------

t0 = time.perf_counter()
model_full = EmbeddingCluster(
    n_clusters=config["n_clusters"],
    reduction_dim=None,
    random_state=config["random_state"],
    n_init=config["n_init"],
    max_iter=config["max_iter"],
)
model_full.fit(X_all)
t_refit = time.perf_counter() - t0

full_labels = np.array(model_full.labels_)
full_pur_comm = compute_purity(pdf["commodity"].values, full_labels)
full_nmi_comm = compute_nmi(pdf["commodity"].values, full_labels)

print(f"Full refit: {t_refit:.1f}s")
print(f"  Commodity purity: {full_pur_comm:.4f}")
print(f"  Commodity NMI:    {full_nmi_comm:.4f}")
print(f"\nSpeedup (slot vs refit): {t_refit/t_slot:.0f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save/Load Roundtrip

# COMMAND ----------

import tempfile, os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "supplier_snap")

    t0 = time.perf_counter()
    snap.save(path)
    t_save = time.perf_counter() - t0

    sizes = {f: os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)}
    total_size = sum(sizes.values())

    t0 = time.perf_counter()
    from rustcluster import ClusterSnapshot
    loaded = ClusterSnapshot.load(path)
    t_load = time.perf_counter() - t0

    loaded_labels = loaded.assign(X_test)
    match = np.array_equal(slot_labels, loaded_labels)

    print(f"Save: {t_save*1000:.2f} ms")
    print(f"Load: {t_load*1000:.2f} ms")
    print(f"Size: {total_size/1024:.1f} KB")
    for fname, sz in sorted(sizes.items()):
        print(f"  {fname}: {sz/1024:.1f} KB")
    print(f"vs training data: {X_train.nbytes/1024/1024:.1f} MB")
    print(f"Roundtrip match: {match}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Drift Detection

# COMMAND ----------

report_test = snap.drift_report(X_test)
print(f"In-distribution (test set):")
print(f"  Global mean distance: {report_test.global_mean_distance_:.4f}")
print(f"  Cluster sizes:        {report_test.new_cluster_sizes_}")

# Simulate drift: random embeddings
X_random = rng.standard_normal(X_test.shape)
report_random = snap.drift_report(X_random)
print(f"\nRandom embeddings (drift):")
print(f"  Global mean distance: {report_random.global_mean_distance_:.4f}")

print(f"\nDrift ratio: {report_random.global_mean_distance_ / max(report_test.global_mean_distance_, 1e-10):.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Cluster Profile (Top Commodities per Cluster)

# COMMAND ----------

slot_df = pd.DataFrame({
    "commodity": y_test_commodity,
    "sub_commodity": y_test_sub,
    "cluster": slot_labels,
    "confidence": confs,
})

for cid in range(config["n_clusters"]):
    mask = slot_df["cluster"] == cid
    cluster_data = slot_df[mask]
    n_in = mask.sum()
    if n_in == 0:
        continue
    top_comm = cluster_data["commodity"].value_counts().head(3)
    mean_conf = cluster_data["confidence"].mean()
    top_str = " | ".join(f"{c} ({cnt})" for c, cnt in top_comm.items())
    print(f"  Cluster {cid:>2}: {n_in:>6,} items, conf={mean_conf:.3f}  →  {top_str}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Summary

# COMMAND ----------

pipeline_time = time.perf_counter() - PIPELINE_START

print(f"{'='*70}")
print(f"  Supplier Snapshot Experiment — Summary")
print(f"{'='*70}")
print(f"  Records:           {n:,}")
print(f"  Train/Test:        {n_train:,} / {n_test:,}")
print(f"  Embedding dim:     {d} → {target} (matryoshka)")
print(f"  Clusters:          {config['n_clusters']}")
print(f"{'='*70}")
print(f"  {'Metric':<25} {'Train':>10} {'Slot':>10} {'Full Refit':>12}")
print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
print(f"  {'Commodity purity':<25} {train_pur_comm:>10.4f} {slot_pur_comm:>10.4f} {full_pur_comm:>12.4f}")
print(f"  {'Commodity NMI':<25} {train_nmi_comm:>10.4f} {slot_nmi_comm:>10.4f} {full_nmi_comm:>12.4f}")
print(f"  {'Time':<25} {t_fit:>9.1f}s {t_slot*1000:>9.1f}ms {t_refit:>11.1f}s")
print(f"  {'Speedup vs refit':<25} {'':>10} {t_refit/t_slot:>9.0f}x {'':>12}")
print(f"  {'Snapshot size':<25} {'':>10} {total_size/1024:>8.1f} KB {'':>12}")
print(f"  {'Training fidelity':<25} {'':>10} {match and '100%' or '<100%':>10} {'':>12}")
print(f"{'='*70}")
print(f"  Total pipeline:    {pipeline_time:.1f}s")
