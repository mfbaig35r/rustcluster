# Databricks notebook source
# MAGIC %md
# MAGIC # rustcluster PyPI Package Test
# MAGIC
# MAGIC Validates `rustcluster` installed from PyPI on a Databricks cluster.
# MAGIC Tests all algorithms, embedding clustering, and the reducer workflow.

# COMMAND ----------

# MAGIC %pip install rustcluster --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Version and Import Check

# COMMAND ----------

import rustcluster
import numpy as np

from rustcluster import (
    KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)
from rustcluster.experimental import EmbeddingCluster, EmbeddingReducer

print(f"rustcluster version: {rustcluster.__version__}")
print(f"numpy version: {np.__version__}")
print("All imports successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Test Data

# COMMAND ----------

rng = np.random.default_rng(42)

def make_blobs(n_per=200, d=8, k=5, noise=0.5):
    """Generate k well-separated clusters."""
    centers = rng.standard_normal((k, d)) * 5
    X, y = [], []
    for i in range(k):
        X.append(centers[i] + rng.standard_normal((n_per, d)) * noise)
        y.extend([i] * n_per)
    return np.vstack(X).astype(np.float32), np.array(y)

def make_embeddings(n=500, d=256, k=5):
    """Generate k clusters on the unit sphere (embedding-like)."""
    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X, y = [], []
    for i in range(k):
        cluster = centers[i] + rng.standard_normal((n // k, d)) * 0.15
        X.append(cluster)
        y.extend([i] * (n // k))
    X = np.vstack(X).astype(np.float32)
    return X, np.array(y)

X_blobs, y_blobs = make_blobs()
X_embed, y_embed = make_embeddings()
print(f"Blobs: {X_blobs.shape}, Embeddings: {X_embed.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. KMeans

# COMMAND ----------

model = KMeans(n_clusters=5, random_state=42, n_init=3)
model.fit(X_blobs)

assert model.labels_.shape == (1000,)
assert model.cluster_centers_.shape == (5, 8)
assert model.n_iter_ >= 1
assert model.inertia_ > 0

# Predict on new data
preds = model.predict(X_blobs[:10])
assert preds.shape == (10,)

sil = silhouette_score(X_blobs, model.labels_)
print(f"KMeans: inertia={model.inertia_:.1f}, iters={model.n_iter_}, silhouette={sil:.3f}")
assert sil > 0.3, f"Silhouette too low: {sil}"
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. KMeans with Cosine Metric

# COMMAND ----------

model_cos = KMeans(n_clusters=5, metric="cosine", random_state=42)
model_cos.fit(X_embed)

assert model_cos.labels_.shape == (500,)
print(f"KMeans (cosine): inertia={model_cos.inertia_:.1f}, iters={model_cos.n_iter_}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. MiniBatchKMeans

# COMMAND ----------

model_mb = MiniBatchKMeans(n_clusters=5, batch_size=128, random_state=42)
model_mb.fit(X_blobs)

assert model_mb.labels_.shape == (1000,)
assert model_mb.inertia_ > 0
print(f"MiniBatchKMeans: inertia={model_mb.inertia_:.1f}, iters={model_mb.n_iter_}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. DBSCAN

# COMMAND ----------

model_db = DBSCAN(eps=1.5, min_samples=5)
model_db.fit(X_blobs)

n_clusters = len(set(model_db.labels_)) - (1 if -1 in model_db.labels_ else 0)
n_noise = sum(model_db.labels_ == -1)
assert n_clusters >= 3, f"Expected at least 3 clusters, got {n_clusters}"
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. HDBSCAN

# COMMAND ----------

model_hdb = HDBSCAN(min_cluster_size=20)
model_hdb.fit(X_blobs)

n_clusters = len(set(model_hdb.labels_)) - (1 if -1 in model_hdb.labels_ else 0)
assert n_clusters >= 3
assert model_hdb.probabilities_.shape == (1000,)
assert all(0 <= p <= 1 for p in model_hdb.probabilities_)
assert len(model_hdb.cluster_persistence_) == n_clusters
print(f"HDBSCAN: {n_clusters} clusters, probabilities range=[{model_hdb.probabilities_.min():.2f}, {model_hdb.probabilities_.max():.2f}]")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Agglomerative Clustering

# COMMAND ----------

model_agg = AgglomerativeClustering(n_clusters=5, linkage="ward")
model_agg.fit(X_blobs)

assert model_agg.labels_.shape == (1000,)
n_merges = model_agg.children_.shape[0]
assert model_agg.children_.shape == (n_merges, 2)
assert model_agg.distances_.shape == (n_merges,)
print(f"Agglomerative: labels={len(set(model_agg.labels_))} clusters, {n_merges} merges, last merge distance={model_agg.distances_[-1]:.2f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Evaluation Metrics

# COMMAND ----------

labels = KMeans(n_clusters=5, random_state=42).fit(X_blobs).labels_

sil = silhouette_score(X_blobs, labels)
ch = calinski_harabasz_score(X_blobs, labels)
db = davies_bouldin_score(X_blobs, labels)

assert -1 <= sil <= 1
assert ch > 0
assert db >= 0
print(f"Silhouette: {sil:.3f}")
print(f"Calinski-Harabasz: {ch:.1f}")
print(f"Davies-Bouldin: {db:.3f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Pickle Serialization

# COMMAND ----------

import pickle

model_orig = KMeans(n_clusters=5, random_state=42).fit(X_blobs)
data = pickle.dumps(model_orig)
model_loaded = pickle.loads(data)

preds_orig = model_orig.predict(X_blobs[:50])
preds_loaded = model_loaded.predict(X_blobs[:50])
assert np.array_equal(preds_orig, preds_loaded), "Pickle roundtrip failed"
print(f"Pickle: {len(data)} bytes, predictions match after roundtrip")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. EmbeddingCluster

# COMMAND ----------

model_ec = EmbeddingCluster(n_clusters=5, reduction_dim=32, random_state=42, n_init=3)
model_ec.fit(X_embed)

assert model_ec.labels_.shape == (500,)
assert model_ec.cluster_centers_.shape == (5, 32)
assert model_ec.objective_ > 0
assert model_ec.n_iter_ >= 1
assert len(model_ec.representatives_) == 5
assert len(model_ec.intra_similarity_) == 5
assert len(model_ec.resultant_lengths_) == 5
assert all(0 <= r <= 1 for r in model_ec.resultant_lengths_)

print(f"EmbeddingCluster: obj={model_ec.objective_:.1f}, iters={model_ec.n_iter_}")
print(f"  Resultant lengths: mean={np.mean(model_ec.resultant_lengths_):.3f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. EmbeddingCluster — reduced_data_ Property

# COMMAND ----------

rd = model_ec.reduced_data_
assert rd is not None, "reduced_data_ should not be None when reduction_dim is set"
assert rd.shape == (500, 32), f"Expected (500, 32), got {rd.shape}"

# Should be L2-normalized
norms = np.linalg.norm(rd, axis=1)
assert np.allclose(norms, 1.0, atol=1e-6), f"Rows not unit-norm: {norms[:5]}"

# Should be a copy
rd2 = model_ec.reduced_data_
rd[0, 0] = 999.0
assert rd2[0, 0] != 999.0, "reduced_data_ should return a copy"

print(f"reduced_data_: shape={rd.shape}, norms OK, copy OK")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. EmbeddingCluster — No Reduction

# COMMAND ----------

model_norec = EmbeddingCluster(n_clusters=5, reduction_dim=None, random_state=42)
model_norec.fit(X_embed)

assert model_norec.labels_.shape == (500,)
assert model_norec.cluster_centers_.shape == (5, 256)
assert model_norec.reduced_data_ is None
print(f"EmbeddingCluster (no reduction): obj={model_norec.objective_:.1f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 14. vMF Refinement

# COMMAND ----------

model_vmf = EmbeddingCluster(n_clusters=5, reduction_dim=32, random_state=42)
model_vmf.fit(X_embed)
model_vmf.refine_vmf()

assert model_vmf.probabilities_.shape == (500, 5)
row_sums = model_vmf.probabilities_.sum(axis=1)
assert np.allclose(row_sums, 1.0, atol=1e-5), f"Probabilities don't sum to 1: {row_sums[:5]}"
assert len(model_vmf.concentrations_) == 5
assert all(k > 0 for k in model_vmf.concentrations_)
assert np.isfinite(model_vmf.bic_)

print(f"vMF: BIC={model_vmf.bic_:.0f}, kappa={model_vmf.concentrations_}")
print(f"  P(max) mean={model_vmf.probabilities_.max(axis=1).mean():.3f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 15. EmbeddingReducer — PCA

# COMMAND ----------

reducer = EmbeddingReducer(target_dim=32, method="pca", random_state=42)
X_reduced = reducer.fit_transform(X_embed)

assert X_reduced.shape == (500, 32), f"Expected (500, 32), got {X_reduced.shape}"
assert X_reduced.dtype == np.float64

# L2-normalized
norms = np.linalg.norm(X_reduced, axis=1)
assert np.allclose(norms, 1.0, atol=1e-10), f"Not unit-norm: {norms[:5]}"

# fit then transform should match fit_transform
reducer2 = EmbeddingReducer(target_dim=32, method="pca", random_state=42)
reducer2.fit(X_embed)
X_sep = reducer2.transform(X_embed)
assert np.allclose(X_reduced, X_sep, atol=1e-10), "fit+transform != fit_transform"

print(f"EmbeddingReducer (PCA): {X_embed.shape} -> {X_reduced.shape}, unit-norm OK")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 16. EmbeddingReducer — Save/Load

# COMMAND ----------

import tempfile, os

with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
    path = f.name

try:
    reducer.save(path)
    file_size = os.path.getsize(path)

    loaded = EmbeddingReducer.load(path)
    X_loaded = loaded.transform(X_embed)

    assert np.allclose(X_reduced, X_loaded, atol=1e-10), "Save/load roundtrip failed"
    print(f"Save/load: {file_size:,} bytes, roundtrip match OK")
    print("  PASSED")
finally:
    os.unlink(path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 17. EmbeddingReducer — Matryoshka

# COMMAND ----------

reducer_mat = EmbeddingReducer(target_dim=32, method="matryoshka")
X_mat = reducer_mat.fit_transform(X_embed)

assert X_mat.shape == (500, 32)

# Should match first 32 cols, L2-normalized
expected = X_embed[:, :32].astype(np.float64).copy()
expected /= np.linalg.norm(expected, axis=1, keepdims=True)
assert np.allclose(X_mat, expected, atol=1e-6), "Matryoshka didn't truncate correctly"

print(f"EmbeddingReducer (matryoshka): {X_embed.shape} -> {X_mat.shape}, truncation OK")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 18. Reduced Data with Other Algorithms

# COMMAND ----------

X_reduced = EmbeddingReducer(target_dim=32).fit_transform(X_embed)

results = {}

km = KMeans(n_clusters=5, metric="cosine", random_state=42).fit(X_reduced)
results["KMeans (cosine)"] = len(set(km.labels_))

db = DBSCAN(eps=0.5, metric="cosine").fit(X_reduced)
results["DBSCAN (cosine)"] = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

hdb = HDBSCAN(min_cluster_size=20, metric="cosine").fit(X_reduced)
results["HDBSCAN (cosine)"] = len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0)

agg = AgglomerativeClustering(n_clusters=5, linkage="average", metric="cosine").fit(X_reduced)
results["Agglomerative (cosine)"] = len(set(agg.labels_))

ec = EmbeddingCluster(n_clusters=5, reduction_dim=None, random_state=42).fit(X_reduced)
results["EmbeddingCluster"] = len(set(ec.labels_))

for name, k in results.items():
    print(f"  {name}: {k} clusters")
    assert k >= 1, f"{name} produced 0 clusters"

print("  ALL PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 19. f32 and f64 Compatibility

# COMMAND ----------

X_f32 = X_blobs.astype(np.float32)
X_f64 = X_blobs.astype(np.float64)

m32 = KMeans(n_clusters=5, random_state=42).fit(X_f32)
m64 = KMeans(n_clusters=5, random_state=42).fit(X_f64)

assert m32.labels_.shape == m64.labels_.shape
print(f"f32 inertia: {m32.inertia_:.1f}")
print(f"f64 inertia: {m64.inertia_:.1f}")
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 20. Error Handling

# COMMAND ----------

errors_caught = 0

# Transform before fit
try:
    EmbeddingReducer(target_dim=32).transform(X_embed)
    assert False, "Should have raised"
except RuntimeError:
    errors_caught += 1

# Save before fit
try:
    EmbeddingReducer(target_dim=32).save("/tmp/nope.bin")
    assert False, "Should have raised"
except RuntimeError:
    errors_caught += 1

# Invalid method
try:
    EmbeddingReducer(target_dim=32, method="invalid")
    assert False, "Should have raised"
except ValueError:
    errors_caught += 1

# Labels before fit
try:
    EmbeddingCluster().labels_
    assert False, "Should have raised"
except RuntimeError:
    errors_caught += 1

# Dimension mismatch
try:
    r = EmbeddingReducer(target_dim=32)
    r.fit(X_embed)  # 256d
    r.transform(np.random.randn(10, 128).astype(np.float32))  # 128d
    assert False, "Should have raised"
except ValueError:
    errors_caught += 1

print(f"Error handling: {errors_caught}/5 errors caught correctly")
assert errors_caught == 5
print("  PASSED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 50)
print("ALL TESTS PASSED")
print("=" * 50)
print()
print("Tested:")
print("  - KMeans (euclidean + cosine)")
print("  - MiniBatchKMeans")
print("  - DBSCAN")
print("  - HDBSCAN")
print("  - AgglomerativeClustering")
print("  - EmbeddingCluster (with/without PCA)")
print("  - EmbeddingCluster reduced_data_")
print("  - vMF refinement")
print("  - EmbeddingReducer (PCA + matryoshka)")
print("  - EmbeddingReducer save/load")
print("  - Reduced data with all algorithms")
print("  - Evaluation metrics (silhouette, CH, DB)")
print("  - Pickle serialization")
print("  - f32/f64 compatibility")
print("  - Error handling")
