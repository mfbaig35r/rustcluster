# Databricks notebook source
# MAGIC %md
# MAGIC # Supplier Clustering — Ultra Performance
# MAGIC
# MAGIC Optimized for speed on 312K supplier embeddings (text-embedding-3-small).
# MAGIC
# MAGIC **Key optimizations:**
# MAGIC - Matryoshka truncation instead of PCA (instant — text-embedding-3-small is Matryoshka-trained)
# MAGIC - Streaming embedding extraction (no full toPandas on embedding column)
# MAGIC - Single-pass commodity + sub-commodity clustering
# MAGIC - Sampled quality metrics only
# MAGIC
# MAGIC **Expected runtime: ~60-90 seconds** (vs ~10+ minutes with PCA + sklearn)

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
    "output_table": "global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster",
    "log_table": "global_supplier_master.supplier_name_standardization.rustcluster_logs",

    # Reduction: "matryoshka" (instant, recommended for text-embedding-3) or "pca"
    "reduction_method": "matryoshka",
    "reduction_dim": 128,

    # Clustering
    "min_clusters": 2,
    "max_commodity_clusters": 20,
    "max_sub_clusters": 25,
    "n_init": 3,
    "max_iter": 100,
    "random_state": 42,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load

# COMMAND ----------

import numpy as np
import pandas as pd
import gc
import logging
import json
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ultra")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

t0 = time.perf_counter()

# Metadata (no embeddings — tiny)
df = spark.read.table(config["input_table"]).filter("embedding IS NOT NULL")
pdf = df.select(*config["metadata_columns"]).toPandas()
n_total = len(pdf)
log.info(f"Metadata: {n_total:,} rows")

# Stream embeddings row-by-row — JVM never holds full array
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
# MAGIC ## 2. Reduce
# MAGIC
# MAGIC **Matryoshka** truncation is instant — text-embedding-3-small is trained with Matryoshka
# MAGIC representation learning, so the first 128 dimensions carry the most information.
# MAGIC No matrix factorization needed.

# COMMAND ----------

from rustcluster.experimental import EmbeddingReducer, EmbeddingCluster

t0 = time.perf_counter()
method = config["reduction_method"]
target = config["reduction_dim"]

if method == "matryoshka":
    # Instant — slice first target_dim columns + L2 normalize
    X = embeddings[:, :target].copy()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = (X / norms).astype(np.float64)
    del embeddings; gc.collect()
    reduce_time = time.perf_counter() - t0
    log.info(f"Matryoshka: {d}d -> {target}d in {reduce_time:.3f}s (instant)")

elif method == "pca":
    sample_n = min(50_000, n)
    sample_idx = np.random.RandomState(config["random_state"]).choice(n, sample_n, replace=False)
    reducer = EmbeddingReducer(target_dim=target, method="pca", random_state=config["random_state"])
    reducer.fit(embeddings[sample_idx])
    del sample_idx; gc.collect()

    chunks = []
    for s in range(0, n, 50_000):
        chunks.append(reducer.transform(embeddings[s:min(s + 50_000, n)]))
    X = np.vstack(chunks)
    del chunks, embeddings; gc.collect()
    reduce_time = time.perf_counter() - t0
    log.info(f"PCA: {d}d -> {target}d in {reduce_time:.1f}s")

print(f"X: {X.shape}, {X.nbytes / 1e6:.0f} MB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Commodity Clustering

# COMMAND ----------

from rustcluster import silhouette_score as rust_silhouette

distinct_commodities = pdf["commodity"].nunique()
k_comm = max(config["min_clusters"], int(distinct_commodities * 0.1))
k_comm = min(k_comm, config["max_commodity_clusters"])

t0 = time.perf_counter()
comm_model = EmbeddingCluster(
    n_clusters=k_comm,
    reduction_dim=None,
    random_state=config["random_state"],
    n_init=config["n_init"],
    max_iter=config["max_iter"],
)
comm_model.fit(X)
comm_time = time.perf_counter() - t0

pdf["commodity_cluster"] = comm_model.labels_.astype(int)

# Sampled silhouette (O(n^2) on full data is prohibitive)
sil_idx = np.random.RandomState(42).choice(n, min(10_000, n), replace=False)
sil = rust_silhouette(X[sil_idx], comm_model.labels_[sil_idx].tolist())

log.info(f"Commodity: K={k_comm}, {comm_time:.1f}s, silhouette={sil:.3f}")

for cid in range(k_comm):
    mask = pdf["commodity_cluster"] == cid
    top = pdf.loc[mask, "commodity"].value_counts().index[0] if mask.sum() > 0 else "?"
    print(f"  {cid:2d}: {mask.sum():>6,} | {top}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sub-Commodity Clustering

# COMMAND ----------

t0 = time.perf_counter()
pdf["sub_commodity_cluster"] = -1
sub_plan = []

for cid in range(k_comm):
    mask = pdf["commodity_cluster"] == cid
    idx = pdf.index[mask]
    gX = X[mask.values]
    ng = len(idx)

    if ng < config["min_clusters"] * 2:
        pdf.loc[idx, "sub_commodity_cluster"] = 0
        sub_plan.append({"cluster": cid, "n": ng, "sub_k": 1})
        continue

    sub_k = max(config["min_clusters"], int(pdf.loc[idx, "sub_commodity"].nunique() * 0.1))
    sub_k = min(sub_k, config["max_sub_clusters"], ng // 2)

    sm = EmbeddingCluster(
        n_clusters=sub_k, reduction_dim=None,
        random_state=config["random_state"], n_init=config["n_init"], max_iter=config["max_iter"],
    )
    sm.fit(gX)
    pdf.loc[idx, "sub_commodity_cluster"] = sm.labels_.astype(int)
    sub_plan.append({"cluster": cid, "n": ng, "sub_k": sub_k, "obj": round(float(sm.objective_), 1)})

sub_time = time.perf_counter() - t0
log.info(f"Sub-clustering: {len(sub_plan)} groups in {sub_time:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary

# COMMAND ----------

total_sub = sum(p["sub_k"] for p in sub_plan)
pipeline_time = time.perf_counter() - PIPELINE_START

print(f"{'='*60}")
print(f"  Method:          {config['reduction_method']} -> {config['reduction_dim']}d")
print(f"  Records:         {n:,}")
print(f"  Commodity K:     {k_comm}")
print(f"  Sub-clusters:    {total_sub}")
print(f"  Silhouette:      {sil:.3f}")
print(f"{'='*60}")
print(f"  Load:            {load_time:.1f}s")
print(f"  Reduce:          {reduce_time:.1f}s")
print(f"  Commodity:       {comm_time:.1f}s")
print(f"  Sub-commodity:   {sub_time:.1f}s")
print(f"  TOTAL PIPELINE:  {pipeline_time:.1f}s")
print(f"{'='*60}")

display(pd.DataFrame(sub_plan))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preview

# COMMAND ----------

out_cols = ["parent_supplier_id", "supplier_name", "industry", "commodity", "sub_commodity",
            "commodity_cluster", "sub_commodity_cluster"]
display(pdf[out_cols].head(30))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save

# COMMAND ----------

run_id = str(uuid.uuid4())

out_pdf = pdf[out_cols].copy()
out_pdf["cluster_run_id"] = run_id
out_pdf["reduction_method"] = config["reduction_method"]
out_pdf["reduction_dim"] = config["reduction_dim"]
out_pdf["clustered_at"] = datetime.now()

spark.createDataFrame(out_pdf) \
    .write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(config["output_table"])

log.info(f"Saved {len(out_pdf):,} rows to {config['output_table']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log

# COMMAND ----------

run_log = {
    "run_id": run_id,
    "input_table": config["input_table"],
    "output_table": config["output_table"],
    "records": int(n),
    "embedding_dim": int(d),
    "reduction_method": config["reduction_method"],
    "reduction_dim": config["reduction_dim"],
    "k_commodity": k_comm,
    "total_sub_clusters": total_sub,
    "silhouette": round(sil, 4),
    "timing": {
        "load_s": round(load_time, 1),
        "reduce_s": round(reduce_time, 1),
        "commodity_s": round(comm_time, 1),
        "sub_commodity_s": round(sub_time, 1),
        "total_s": round(pipeline_time, 1),
    },
    "config": {k: v for k, v in config.items() if k not in ("input_table", "output_table", "log_table", "metadata_columns")},
    "library": "rustcluster",
    "library_version": __import__("rustcluster").__version__,
}

log_json = json.dumps(run_log, indent=2, default=str)
spark.createDataFrame([(run_id, log_json, datetime.now())], ["run_id", "cluster_log", "timestamp"]) \
    .write.format("delta").mode("append").option("mergeSchema", "true") \
    .saveAsTable(config["log_table"])

print(log_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify

# COMMAND ----------

result = spark.read.table(config["output_table"])
print(f"Output: {result.count():,} records")
print(f"Commodity clusters: {result.select('commodity_cluster').distinct().count()}")
print(f"Unique (commodity, sub) pairs: {result.select('commodity_cluster', 'sub_commodity_cluster').distinct().count()}")
display(result.limit(10))
