"""rustcluster.experimental — Experimental clustering algorithms.

These APIs may change between versions.
"""

import numpy as np

from rustcluster._rustcluster import EmbeddingCluster as _RustEmbeddingCluster

__all__ = ["EmbeddingCluster"]


def _prepare(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.dtype == np.float32:
        return np.ascontiguousarray(X, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float64)


class EmbeddingCluster:
    """Purpose-built clustering for dense embedding vectors.

    Pipeline: L2 normalize -> Randomized PCA -> Spherical K-means -> Evaluation

    Optimized for OpenAI, Cohere, Voyage, and similar embedding models.
    Spherical K-means maximizes cosine similarity on the unit hypersphere.

    Parameters
    ----------
    n_clusters : int, default=50
        Number of clusters.
    reduction_dim : int or None, default=128
        Target dimensionality for PCA. None skips reduction.
    max_iter : int, default=100
        Maximum iterations for spherical K-means.
    tol : float, default=1e-6
        Convergence tolerance (angular shift).
    random_state : int, default=0
        Seed for reproducibility.
    n_init : int, default=5
        Number of initializations (best objective wins).
    """

    def __init__(self, n_clusters=50, reduction_dim=128, max_iter=100,
                 tol=1e-6, random_state=0, n_init=5, reduction="pca"):
        self._model = _RustEmbeddingCluster(
            n_clusters=n_clusters,
            reduction_dim=reduction_dim,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=n_init,
            reduction=reduction,
        )
        self._n_clusters = n_clusters
        self._reduction_dim = reduction_dim
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._n_init = n_init
        self._reduction = reduction
        self._X = None  # cached for refine_vmf

    def fit(self, X):
        """Fit the embedding clustering pipeline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Raw embedding vectors (will be L2-normalized internally).

        Returns
        -------
        self
        """
        X = _prepare(X)
        self._X = X
        self._model.fit(X)
        return self

    def refine_vmf(self):
        """Refine clusters with von Mises-Fisher mixture model.

        Produces soft cluster probabilities, per-cluster concentration
        parameters, and BIC for model selection. Requires fit() first.

        Returns
        -------
        self
        """
        if self._X is None:
            raise RuntimeError("Call fit() before refine_vmf()")
        self._model.refine_vmf(self._X)
        return self

    @property
    def labels_(self):
        """Hard cluster assignments."""
        return self._model.labels_

    @property
    def cluster_centers_(self):
        """Unit-norm cluster centroids in reduced space."""
        return self._model.cluster_centers_

    @property
    def objective_(self):
        """Sum of cosine similarities (higher = better)."""
        return self._model.objective_

    @property
    def n_iter_(self):
        """Number of iterations in the best run."""
        return self._model.n_iter_

    @property
    def representatives_(self):
        """Indices of the most representative point per cluster."""
        return self._model.representatives_

    @property
    def intra_similarity_(self):
        """Per-cluster average cosine similarity to centroid."""
        return self._model.intra_similarity_

    @property
    def resultant_lengths_(self):
        """Per-cluster directional concentration [0, 1]."""
        return self._model.resultant_lengths_

    @property
    def probabilities_(self):
        """Soft cluster membership (n, k). Call refine_vmf() first."""
        return self._model.probabilities_

    @property
    def concentrations_(self):
        """Per-cluster vMF concentration κ. Call refine_vmf() first."""
        return self._model.concentrations_

    @property
    def bic_(self):
        """Bayesian Information Criterion. Call refine_vmf() first."""
        return self._model.bic_

    def __repr__(self):
        return (
            f"EmbeddingCluster(n_clusters={self._n_clusters}, "
            f"reduction_dim={self._reduction_dim}, max_iter={self._max_iter}, "
            f"n_init={self._n_init})"
        )
