"""rustcluster — Rust-backed Python clustering library."""

import numpy as np

from rustcluster._rustcluster import KMeans as _RustKMeans

__all__ = ["KMeans"]


class KMeans:
    """K-means clustering backed by a Rust implementation.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default=300
        Maximum iterations per run.
    tol : float, default=1e-4
        Convergence tolerance on centroid shift.
    random_state : int, default=0
        Seed for reproducibility.
    n_init : int, default=10
        Number of initializations; the run with lowest inertia wins.
    algorithm : str, default="auto"
        Algorithm to use: "auto", "lloyd", or "hamerly".
        "auto" selects based on data dimensions and cluster count.
    """

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10, algorithm="auto"):
        self._model = _RustKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=n_init,
            algorithm=algorithm,
        )
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._n_init = n_init
        self._algorithm = algorithm

    def _prepare(self, X):
        """Ensure input is C-contiguous float64."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        return np.ascontiguousarray(X, dtype=np.float64)

    def fit(self, X):
        """Fit the K-means model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._prepare(X)
        self._model.fit(X)
        return self

    def predict(self, X):
        """Predict cluster labels for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = self._prepare(X)
        return self._model.predict(X)

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = self._prepare(X)
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        """Cluster labels for the training data."""
        return self._model.labels_

    @property
    def cluster_centers_(self):
        """Centroid coordinates, shape (n_clusters, n_features)."""
        return self._model.cluster_centers_

    @property
    def inertia_(self):
        """Sum of squared distances to nearest centroid."""
        return self._model.inertia_

    @property
    def n_iter_(self):
        """Number of iterations in the best run."""
        return self._model.n_iter_

    def __repr__(self):
        return (
            f"KMeans(n_clusters={self._n_clusters}, max_iter={self._max_iter}, "
            f"tol={self._tol}, random_state={self._random_state}, "
            f"n_init={self._n_init}, algorithm=\"{self._algorithm}\")"
        )
