"""Vector indexing — flat exact search.

This module provides a FAISS-flavored API for exact nearest-neighbor search
over dense f32 vectors. It is intentionally a small surface today; future
versions will add IVF, HNSW, and PQ.

Examples
--------
>>> import numpy as np
>>> from rustcluster.index import IndexFlatIP
>>> rng = np.random.default_rng(0)
>>> X = rng.standard_normal((1000, 128)).astype(np.float32)
>>> X /= np.linalg.norm(X, axis=1, keepdims=True)
>>> idx = IndexFlatIP(dim=128)
>>> idx.add(X)
>>> distances, labels = idx.search(X[:5], k=10)
>>> lims, dists, labs = idx.range_search(X[:5], threshold=0.5)
"""

from rustcluster._rustcluster import IndexFlatL2, IndexFlatIP

__all__ = ["IndexFlatL2", "IndexFlatIP"]
