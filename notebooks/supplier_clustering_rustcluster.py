# Databricks notebook source
# MAGIC %md
# MAGIC # Supplier Commodity Clustering — rustcluster
# MAGIC
# MAGIC Two-level clustering of supplier embeddings:
# MAGIC 1. **Commodity clusters** — group suppliers by commodity similarity
# MAGIC 2. **Sub-commodity clusters** — finer-grained groups within each commodity cluster
# MAGIC
# MAGIC Uses `rustcluster` (Rust-backed, faer-accelerated PCA, spherical K-means).
# MAGIC Replaces the sklearn-based pipeline with ~10x faster PCA and cosine-native clustering.

# COMMAND ----------

# MAGIC %pip install rustcluster --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

config = {
    # Input
    "input_table": "global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_selection_enrichment_embeddings",
    "columns": ["parent_supplier_id", "supplier_name", "industry", "commodity", "sub_commodity", "embedding"],

    # Output
    "output_table": "global_supplier_master.supplier_name_standardization.ai_zeroshot_supplier_enrichment_rustcluster",
    "log_table": "global_supplier_master.supplier_name_standardization.rustcluster_logs",

    # Clustering
    "reduction_dim": 128,          # PCA target dimensions (None to skip)
    "min_clusters": 2,
    "max_commodity_clusters": 20,
    "max_sub_clusters": 25,
    "n_init": 3,                   # K-means restarts (higher = better but slower)
    "random_state": 42,
}

print(f"Input:  {config['input_table']}")
print(f"Output: {config['output_table']}")
print(f"PCA:    {config['reduction_dim']}d")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and Prepare Data

# COMMAND ----------

import numpy as np
import pandas as pd
import logging
import json
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rustcluster_pipeline")

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Load from Delta
logger.info(f"Loading from {config['input_table']}")
df_spark = spark.read.table(config["input_table"]).select(*config["columns"]).filter("embedding IS NOT NULL")
total_count = df_spark.count()
logger.info(f"Loaded {total_count:,} records")

# Convert to Pandas
pdf = df_spark.toPandas()
embeddings = np.array(pdf["embedding"].tolist(), dtype=np.float32)
n, d = embeddings.shape
logger.info(f"Embeddings: {n:,} x {d}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dimensionality Reduction (PCA — One Time Cost)

# COMMAND ----------

import time
from rustcluster.experimental import EmbeddingReducer, EmbeddingCluster

reduction_dim = config["reduction_dim"]

if reduction_dim and reduction_dim < d:
    logger.info(f"Running PCA: {d}d -> {reduction_dim}d")
    t0 = time.perf_counter()
    reducer = EmbeddingReducer(target_dim=reduction_dim, method="pca", random_state=config["random_state"])
    X = reducer.fit_transform(embeddings)
    pca_time = time.perf_counter() - t0
    logger.info(f"PCA complete: {pca_time:.1f}s, output shape {X.shape}")
else:
    logger.info("Skipping PCA (reduction_dim is None or >= input dim)")
    X = embeddings.astype(np.float64)
    # L2-normalize for cosine clustering
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1
    X = X / norms
    pca_time = 0.0

print(f"Working data: {X.shape}, dtype={X.dtype}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Commodity-Level Clustering

# COMMAND ----------

from rustcluster import KMeans as RustKMeans, silhouette_score as rust_silhouette

# Determine number of commodity clusters
distinct_commodities = pdf["commodity"].nunique()
n_commodity_clusters = max(config["min_clusters"], int(distinct_commodities * 0.1))
n_commodity_clusters = min(n_commodity_clusters, config["max_commodity_clusters"])
logger.info(f"Distinct commodities: {distinct_commodities}, using K={n_commodity_clusters}")

# Cluster
t0 = time.perf_counter()
commodity_model = EmbeddingCluster(
    n_clusters=n_commodity_clusters,
    reduction_dim=None,  # already reduced
    random_state=config["random_state"],
    n_init=config["n_init"],
    max_iter=100,
)
commodity_model.fit(X)
commodity_time = time.perf_counter() - t0

pdf["commodity_cluster"] = commodity_model.labels_.astype(int)

sil = rust_silhouette(X, commodity_model.labels_.tolist())
logger.info(f"Commodity clustering: K={n_commodity_clusters}, {commodity_time:.1f}s, silhouette={sil:.3f}")

# Cluster sizes
for cid in range(n_commodity_clusters):
    mask = pdf["commodity_cluster"] == cid
    size = mask.sum()
    top_commodity = pdf.loc[mask, "commodity"].value_counts().index[0] if size > 0 else "?"
    print(f"  Cluster {cid:2d}: {size:>6,} records | top commodity: {top_commodity}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sub-Commodity Clustering

# COMMAND ----------

pdf["sub_commodity_cluster"] = -1
sub_cluster_plan = []

for cluster_id in range(n_commodity_clusters):
    mask = pdf["commodity_cluster"] == cluster_id
    group_idx = pdf.index[mask]
    group_X = X[mask.values]
    n_group = len(group_idx)

    if n_group < config["min_clusters"] * 2:
        pdf.loc[group_idx, "sub_commodity_cluster"] = 0
        sub_cluster_plan.append({"commodity_cluster": cluster_id, "n": n_group, "sub_k": 1, "note": "too small"})
        continue

    # Determine sub-cluster count
    distinct_sub = pdf.loc[group_idx, "sub_commodity"].nunique()
    sub_k = max(config["min_clusters"], int(distinct_sub * 0.1))
    sub_k = min(sub_k, config["max_sub_clusters"], n_group // 2)

    # Cluster
    sub_model = EmbeddingCluster(
        n_clusters=sub_k,
        reduction_dim=None,
        random_state=config["random_state"],
        n_init=config["n_init"],
        max_iter=100,
    )
    sub_model.fit(group_X)
    pdf.loc[group_idx, "sub_commodity_cluster"] = sub_model.labels_.astype(int)

    sub_cluster_plan.append({
        "commodity_cluster": cluster_id,
        "n": n_group,
        "sub_k": sub_k,
        "objective": round(float(sub_model.objective_), 1),
    })
    logger.info(f"  Commodity cluster {cluster_id}: {n_group:,} records -> {sub_k} sub-clusters")

print(f"\nSub-clustering complete: {len(sub_cluster_plan)} commodity clusters processed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Results Summary

# COMMAND ----------

total_sub_clusters = sum(p["sub_k"] for p in sub_cluster_plan)
print(f"{'='*60}")
print(f"CLUSTERING SUMMARY")
print(f"{'='*60}")
print(f"  Records:              {n:,}")
print(f"  Embedding dim:        {d} -> {X.shape[1]} (PCA)")
print(f"  Commodity clusters:   {n_commodity_clusters}")
print(f"  Sub-commodity clusters: {total_sub_clusters} total")
print(f"  PCA time:             {pca_time:.1f}s")
print(f"  Commodity cluster time: {commodity_time:.1f}s")
print(f"  Silhouette (commodity): {sil:.3f}")
print()

# Show cluster plan
plan_df = pd.DataFrame(sub_cluster_plan)
display(plan_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Preview Output

# COMMAND ----------

output_cols = ["parent_supplier_id", "supplier_name", "industry", "commodity", "sub_commodity",
               "commodity_cluster", "sub_commodity_cluster"]
preview = pdf[output_cols].head(20)
display(preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save to Delta

# COMMAND ----------

run_id = str(uuid.uuid4())
logger.info(f"Run ID: {run_id}")

# Build output DataFrame
output_pdf = pdf[output_cols].copy()
output_pdf["cluster_run_id"] = run_id
output_pdf["clustered_at"] = datetime.now()

# Save to Delta
output_df = spark.createDataFrame(output_pdf)
output_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(config["output_table"])
logger.info(f"Saved {len(output_pdf):,} records to {config['output_table']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log Run

# COMMAND ----------

cluster_log = {
    "run_id": run_id,
    "input_table": config["input_table"],
    "output_table": config["output_table"],
    "total_records": int(n),
    "embedding_dim": int(d),
    "reduction_dim": config["reduction_dim"],
    "n_commodity_clusters": n_commodity_clusters,
    "total_sub_clusters": total_sub_clusters,
    "commodity_silhouette": round(sil, 4),
    "pca_time_s": round(pca_time, 1),
    "commodity_cluster_time_s": round(commodity_time, 1),
    "n_init": config["n_init"],
    "random_state": config["random_state"],
    "cluster_plan": sub_cluster_plan,
    "library": "rustcluster",
    "library_version": __import__("rustcluster").__version__,
}

log_json = json.dumps(cluster_log, indent=2, default=str)
log_df = spark.createDataFrame(
    [(run_id, log_json, datetime.now())],
    ["run_id", "cluster_log", "timestamp"]
)
log_df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(config["log_table"])
logger.info(f"Logged run {run_id} to {config['log_table']}")

print(log_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify Output

# COMMAND ----------

result = spark.read.table(config["output_table"])
print(f"Output table: {result.count():,} records")
print(f"Commodity clusters: {result.select('commodity_cluster').distinct().count()}")
print(f"Sub-commodity clusters: {result.select('commodity_cluster', 'sub_commodity_cluster').distinct().count()}")
display(result.limit(10))
