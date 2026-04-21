"""rustcluster.experimental — Experimental clustering algorithms.

These APIs may change between versions.
"""

import numpy as np

from rustcluster._rustcluster import EmbeddingCluster as _RustEmbeddingCluster
from rustcluster._rustcluster import EmbeddingReducer as _RustEmbeddingReducer
from rustcluster import _prepare_array as _prepare

__all__ = ["EmbeddingCluster", "EmbeddingReducer"]


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
        self._model.fit(X)
        return self

    def refine_vmf(self, X):
        """Refine clusters with von Mises-Fisher mixture model.

        Produces soft cluster probabilities, per-cluster concentration
        parameters, and BIC for model selection. Requires fit() first.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Same data passed to fit() (will be L2-normalized internally).

        Returns
        -------
        self
        """
        X = _prepare(X)
        self._model.refine_vmf(X)
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
    def reduced_data_(self):
        """PCA-reduced, L2-normalized data. None if no reduction was performed."""
        val = self._model.reduced_data_
        if val is None:
            return None
        return np.array(val, copy=True)

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

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "n_clusters": self._n_clusters, "reduction_dim": self._reduction_dim,
            "max_iter": self._max_iter, "tol": self._tol,
            "random_state": self._random_state, "n_init": self._n_init,
            "reduction": self._reduction,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


class EmbeddingReducer:
    """Standalone dimensionality reducer for embedding vectors.

    Wraps randomized PCA (or Matryoshka truncation) as a reusable artifact.
    Fit once, save, then iterate on clustering for free.

    Output is always L2-normalized to preserve cosine geometry.

    Parameters
    ----------
    target_dim : int, default=128
        Target output dimensionality.
    method : str, default="pca"
        Reduction method: "pca" (randomized PCA) or "matryoshka" (prefix truncation).
    random_state : int, default=0
        Seed for reproducibility (PCA only).
    """

    def __init__(self, target_dim=128, method="pca", random_state=0):
        self._model = _RustEmbeddingReducer(
            target_dim=target_dim,
            method=method,
            random_state=random_state,
        )
        self._target_dim = target_dim
        self._method = method

    def fit(self, X):
        """Fit the reducer on embedding data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Embedding vectors.

        Returns
        -------
        self
        """
        X = _prepare(X)
        self._model.fit(X)
        return self

    def transform(self, X):
        """Transform data using the fitted reducer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Embedding vectors (must have same n_features as fit data).

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, target_dim)
            L2-normalized reduced embeddings.
        """
        X = _prepare(X)
        return np.asarray(self._model.transform(X))

    def fit_transform(self, X):
        """Fit and transform in one call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Embedding vectors.

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, target_dim)
            L2-normalized reduced embeddings.
        """
        X = _prepare(X)
        return np.asarray(self._model.fit_transform(X))

    def save(self, path):
        """Save fitted state to a binary file.

        Parameters
        ----------
        path : str
            File path (e.g., "pca_128.bin").
        """
        self._model.save(str(path))

    @classmethod
    def load(cls, path):
        """Load a fitted reducer from a binary file.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        EmbeddingReducer
            Fitted reducer ready for transform().
        """
        rust_model = _RustEmbeddingReducer.load(str(path))
        obj = cls.__new__(cls)
        obj._model = rust_model
        obj._target_dim = rust_model.target_dim
        obj._method = rust_model.method
        return obj

    def __repr__(self):
        return repr(self._model)
