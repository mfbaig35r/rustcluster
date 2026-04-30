"""End-to-end port of the Databricks "Item Similarity Index Generation" notebook.

Original (FAISS) flow:
    embeddings = ...                # (n, d) f32, possibly with None to drop
    embeddings = embeddings / norms # L2 normalize
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    for i in range(0, n, batch):
        lims, dists, labels = index.range_search(embeddings[i:i+batch], 0.7)
        for j in range(...):
            for k in range(lims[j], lims[j+1]):
                if labels[k] == i + j:
                    continue
                rows.append((gc[i+j], gc[labels[k]], float(dists[k])))
        spark_write_batch(rows, mode="append" if i else "overwrite")

Replacement (rustcluster) flow:
    from rustcluster.index import IndexFlatIP
    idx = IndexFlatIP(dim=d)
    idx.add_with_ids(embeddings, gc_item_numbers)        # external IDs baked in
    src, dst, scores = idx.similarity_graph(threshold=0.7)  # one call, fused kernel
    df = pd.DataFrame({"source_gc_item_number": src,
                       "similar_gc_item_number": dst,
                       "similarity_score": scores})
    spark.createDataFrame(df).write...

The batched-write loop, manual self-match filter, and ID-remap dict all go away.

This script demonstrates the rewrite end-to-end against synthetic data and
verifies the resulting edge list is identical (as a set) to the FAISS-style
loop, save/load, and Spark-shape DataFrame.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Tuple

import numpy as np

try:
    import faiss

    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from rustcluster.index import IndexFlatIP


def synth(n: int, d: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    embeddings = rng.standard_normal((n, d)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    gc_item_numbers = np.arange(10_000, 10_000 + n, dtype=np.uint64)
    return embeddings, gc_item_numbers


def faiss_loop(
    embeddings: np.ndarray, gc_ids: np.ndarray, threshold: float, batch: int = 1000
) -> set[tuple[int, int]]:
    """Reproduces the original FAISS notebook loop, returning an edge set."""
    n, d = embeddings.shape
    idx = faiss.IndexFlatIP(d)
    idx.add(embeddings)

    edges: set[tuple[int, int]] = set()
    for i in range(0, n, batch):
        end = min(i + batch, n)
        lims, dists, labs = idx.range_search(embeddings[i:end], threshold)
        for j in range(end - i):
            qi = i + j
            for k in range(lims[j], lims[j + 1]):
                neighbor = int(labs[k])
                if neighbor == qi:
                    continue
                edges.add((int(gc_ids[qi]), int(gc_ids[neighbor])))
    return edges


def rustcluster_kernel(
    embeddings: np.ndarray, gc_ids: np.ndarray, threshold: float
) -> set[tuple[int, int]]:
    """The new flow: a single similarity_graph call."""
    idx = IndexFlatIP(dim=embeddings.shape[1])
    idx.add_with_ids(embeddings, gc_ids)
    src, dst, _ = idx.similarity_graph(threshold=threshold)
    return set(zip(src.tolist(), dst.tolist()))


def main() -> None:
    n, d = 1500, 128
    threshold = 0.3
    embeddings, gc_ids = synth(n, d)

    print(f"# Workload: n={n}, d={d}, threshold={threshold}")

    # 1. Verify edge sets agree with the FAISS loop (when faiss is available).
    if HAS_FAISS:
        t0 = time.perf_counter()
        faiss_set = faiss_loop(embeddings, gc_ids, threshold)
        t_faiss = time.perf_counter() - t0
        print(f"  FAISS loop: {len(faiss_set):,} edges in {t_faiss:.3f}s")

        t0 = time.perf_counter()
        rc_set = rustcluster_kernel(embeddings, gc_ids, threshold)
        t_rc = time.perf_counter() - t0
        print(f"  rustcluster.similarity_graph: {len(rc_set):,} edges in {t_rc:.3f}s")
        print(f"  speedup vs FAISS loop: {t_faiss / t_rc:.2f}x")
        assert faiss_set == rc_set, "edge sets differ"
        print("  edge sets identical")
    else:
        print("  (faiss-cpu not installed — skipping cross-check)")
        rc_set = rustcluster_kernel(embeddings, gc_ids, threshold)
        print(f"  rustcluster.similarity_graph: {len(rc_set):,} edges")

    # 2. Save / load round-trip — what the Databricks notebook would do
    #    instead of rebuilding the index every run.
    print()
    print("# Persistence round-trip")
    idx = IndexFlatIP(dim=d)
    idx.add_with_ids(embeddings, gc_ids)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "items.rci")
        idx.save(path)
        size_kb = sum(
            os.path.getsize(os.path.join(path, f)) for f in os.listdir(path)
        ) / 1024
        print(f"  saved to {path}: {sorted(os.listdir(path))} ({size_kb:.1f} KB)")
        loaded = IndexFlatIP.load(path)
        s1, d1, sc1 = idx.similarity_graph(threshold=threshold, unique_pairs=True)
        s2, d2, sc2 = loaded.similarity_graph(threshold=threshold, unique_pairs=True)
        assert np.array_equal(s1, s2) and np.array_equal(d1, d2) and np.allclose(sc1, sc2)
        print(f"  loaded → similarity_graph identical ({len(s1):,} unique pairs)")

    # 3. Show the Spark-friendly DataFrame shape (without actually needing Spark).
    print()
    print("# DataFrame shape for spark.createDataFrame")
    src, dst, scores = idx.similarity_graph(threshold=threshold)
    print(f"  src dtype={src.dtype}, dst dtype={dst.dtype}, scores dtype={scores.dtype}")
    print(f"  total directed edges: {len(src):,}")
    print("  first 3 rows:")
    for i in range(min(3, len(src))):
        print(f"    ({src[i]}, {dst[i]}, {scores[i]:.4f})")


if __name__ == "__main__":
    main()
