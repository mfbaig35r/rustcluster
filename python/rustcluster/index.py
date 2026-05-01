"""Vector indexing — flat exact search and similarity-graph extraction.

This module provides a FAISS-flavored API for exact nearest-neighbor search
over dense f32 vectors, plus a fused all-pairs similarity-graph kernel that
streams `(src, dst, score)` edges directly from a tile-blocked GEMM.

It is intentionally a small surface today (no IVF, HNSW, or PQ); v1's goal is
to replace FAISS in self-similarity workloads while reusing the rest of
``rustcluster``'s primitives (``ClusterSnapshot``, ``EmbeddingReducer``).

Quickstart
----------

>>> import numpy as np
>>> from rustcluster.index import IndexFlatIP
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((1000, 128)).astype(np.float32)
>>> X /= np.linalg.norm(X, axis=1, keepdims=True)
>>> idx = IndexFlatIP(dim=128)
>>> idx.add(X)
>>> distances, labels = idx.search(X[:5], k=10)
>>> lims, dists, labs = idx.range_search(X[:5], threshold=0.5)

External IDs
------------

Use ``add_with_ids`` to bind your own ``u64`` keys to vectors. Search and
range_search will return your IDs directly — no manual remapping.

>>> ids = np.arange(10_000, 11_000, dtype=np.uint64)
>>> idx2 = IndexFlatIP(dim=128)
>>> idx2.add_with_ids(X, ids)
>>> _, labels = idx2.search(X[:1], k=3)
>>> labels[0, 0]  # 10000 (the external ID, not 0)

Similarity-graph extraction
---------------------------

For self-similarity workloads (find every pair within a threshold over the
indexed dataset), use ``similarity_graph`` rather than looping
``range_search`` — it iterates the upper triangle once via cache-blocked
tiles, emitting edges directly. Output is three parallel NumPy arrays ready
for Pandas / Spark.

>>> src, dst, scores = idx2.similarity_graph(threshold=0.7)
>>> # Each undirected pair appears as both (a, b) and (b, a) by default.
>>> # Pass unique_pairs=True to emit only (a, b) where a < b.

Persistence
-----------

>>> idx2.save("clusters/items.rci")
>>> reloaded = IndexFlatIP.load("clusters/items.rci")

Format is a directory with ``vectors.safetensors``, optionally
``ids.safetensors``, plus ``metadata.json`` (matching the existing
``ClusterSnapshot`` convention).

Notes
-----

- All inputs must be f32 NumPy arrays. f64 is not accepted; cast first.
- For cosine similarity, L2-normalize on the way in and use ``IndexFlatIP``.
- Squared L2 distances are returned by ``IndexFlatL2`` (matches FAISS).
- ``similarity_graph`` allocates the full edge list. With a loose threshold
  and large ``n`` this is O(n²) — choose your threshold accordingly. A
  streaming variant lands in v2.

Performance vs FAISS
--------------------

This module uses ``faer`` (pure-Rust SIMD GEMM) for matrix multiplications
rather than a BLAS dependency. The tradeoff:

- At ``d ≤ 384``: ``similarity_graph`` matches or beats the FAISS batched
  ``range_search`` loop (2-3x faster at d=128 on Apple Silicon).
- At ``d ≥ 1024`` (raw embedding dimensions like ``text-embedding-3-small``
  at 1536): faer lags hand-tuned BLAS by ~20-30%. The pure-Rust wheel
  trades raw GEMM speed for the "pip install just works" property — no
  BLAS dependency, no per-platform wheel-distribution overhead.

A BLAS-backed build that closes this gap is being scoped (see
``docs/blas-backend-research-prompt.md``).
"""

from rustcluster._rustcluster import IndexFlatL2, IndexFlatIP

__all__ = ["IndexFlatL2", "IndexFlatIP"]
