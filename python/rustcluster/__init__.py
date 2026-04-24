"""rustcluster — Rust-backed Python clustering library."""

from importlib.metadata import version as _get_version

__version__ = _get_version("rustcluster")

import numpy as np

from rustcluster._rustcluster import KMeans as _RustKMeans
from rustcluster._rustcluster import MiniBatchKMeans as _RustMiniBatchKMeans
from rustcluster._rustcluster import Dbscan as _RustDbscan
from rustcluster._rustcluster import Hdbscan as _RustHdbscan
from rustcluster._rustcluster import AgglomerativeClustering as _RustAgglomerative
from rustcluster._rustcluster import (
    silhouette_score as _silhouette_score,
    calinski_harabasz_score as _calinski_harabasz_score,
    davies_bouldin_score as _davies_bouldin_score,
)

from rustcluster.snapshot import ClusterSnapshot

__all__ = [
    "KMeans",
    "MiniBatchKMeans",
    "DBSCAN",
    "HDBSCAN",
    "AgglomerativeClustering",
    "ClusterSnapshot",
    "silhouette_score",
    "calinski_harabasz_score",
    "davies_bouldin_score",
]


def _prepare_array(X):
    """Ensure input is a 2D C-contiguous float32 or float64 array."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.dtype == np.float32:
        return np.ascontiguousarray(X, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float64)


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
        Note: cosine metric forces "lloyd" (Hamerly assumes Euclidean bounds).
    metric : str, default="euclidean"
        Distance metric: "euclidean" or "cosine".
    """

    def __init__(self, n_clusters, max_iter=300, tol=1e-4, random_state=0, n_init=10, algorithm="auto", metric="euclidean"):
        self._model = _RustKMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            n_init=n_init,
            algorithm=algorithm,
            metric=metric,
        )
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._n_init = n_init
        self._algorithm = algorithm
        self._metric = metric

    def fit(self, X):
        """Fit the K-means model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _prepare_array(X)
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
        X = _prepare_array(X)
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
        X = _prepare_array(X)
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
            f"n_init={self._n_init}, algorithm=\"{self._algorithm}\", "
            f"metric=\"{self._metric}\")"
        )

    def snapshot(self):
        """Create a frozen snapshot for incremental assignment."""
        return ClusterSnapshot(self._model.snapshot())

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "n_clusters": self._n_clusters, "max_iter": self._max_iter,
            "tol": self._tol, "random_state": self._random_state,
            "n_init": self._n_init, "algorithm": self._algorithm, "metric": self._metric,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


class MiniBatchKMeans:
    """Mini-batch K-means clustering backed by a Rust implementation.

    Processes random subsets of data per iteration for faster convergence
    on large datasets, at the cost of slightly worse cluster quality.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    batch_size : int, default=1024
        Number of samples per mini-batch.
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=0.0
        Early stopping tolerance on EWA inertia change. 0 disables.
    random_state : int, default=0
        Seed for reproducibility.
    max_no_improvement : int, default=10
        Stop after this many iterations with no improvement.
    metric : str, default="euclidean"
        Distance metric: "euclidean" or "cosine".
    """

    def __init__(self, n_clusters, batch_size=1024, max_iter=100, tol=0.0,
                 random_state=0, max_no_improvement=10, metric="euclidean"):
        self._model = _RustMiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            max_no_improvement=max_no_improvement,
            metric=metric,
        )
        self._n_clusters = n_clusters
        self._batch_size = batch_size
        self._max_iter = max_iter
        self._tol = tol
        self._random_state = random_state
        self._max_no_improvement = max_no_improvement
        self._metric = metric

    def fit(self, X):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _prepare_array(X)
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
        X = _prepare_array(X)
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
        X = _prepare_array(X)
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
        """Number of iterations run."""
        return self._model.n_iter_

    def __repr__(self):
        return (
            f"MiniBatchKMeans(n_clusters={self._n_clusters}, batch_size={self._batch_size}, "
            f"max_iter={self._max_iter}, tol={self._tol}, random_state={self._random_state}, "
            f"max_no_improvement={self._max_no_improvement}, metric=\"{self._metric}\")"
        )

    def snapshot(self):
        """Create a frozen snapshot for incremental assignment."""
        return ClusterSnapshot(self._model.snapshot())

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "n_clusters": self._n_clusters, "batch_size": self._batch_size,
            "max_iter": self._max_iter, "tol": self._tol, "random_state": self._random_state,
            "max_no_improvement": self._max_no_improvement, "metric": self._metric,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


class DBSCAN:
    """DBSCAN clustering backed by a Rust implementation.

    Parameters
    ----------
    eps : float, default=0.5
        Maximum distance between two samples to be considered neighbors.
    min_samples : int, default=5
        Minimum number of points required to form a core point.
    metric : str, default="euclidean"
        Distance metric: "euclidean" or "cosine".
    """

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self._model = _RustDbscan(eps=eps, min_samples=min_samples, metric=metric)
        self._eps = eps
        self._min_samples = min_samples
        self._metric = metric

    def fit(self, X):
        """Fit the DBSCAN model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _prepare_array(X)
        self._model.fit(X)
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), -1 for noise
        """
        X = _prepare_array(X)
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        """Cluster labels for the training data. -1 indicates noise."""
        return self._model.labels_

    @property
    def core_sample_indices_(self):
        """Indices of core samples."""
        return self._model.core_sample_indices_

    @property
    def components_(self):
        """Core sample coordinates, shape (n_core_samples, n_features)."""
        return self._model.components_

    def __repr__(self):
        return (
            f"DBSCAN(eps={self._eps}, min_samples={self._min_samples}, "
            f"metric=\"{self._metric}\")"
        )

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "eps": self._eps, "min_samples": self._min_samples, "metric": self._metric,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

class HDBSCAN:
    """HDBSCAN clustering backed by a Rust implementation.

    Hierarchical density-based clustering that automatically determines
    the number of clusters and provides soft cluster membership probabilities.

    Parameters
    ----------
    min_cluster_size : int, default=5
        Minimum number of points to form a cluster.
    min_samples : int or None, default=None
        Core distance parameter. Defaults to min_cluster_size if None.
    metric : str, default="euclidean"
        Distance metric: "euclidean" or "cosine".
    cluster_selection_method : str, default="eom"
        Method for extracting flat clusters: "eom" (Excess of Mass) or "leaf".
    """

    def __init__(self, min_cluster_size=5, min_samples=None,
                 metric="euclidean", cluster_selection_method="eom"):
        self._model = _RustHdbscan(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=cluster_selection_method,
        )
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples if min_samples is not None else min_cluster_size
        self._metric = metric
        self._cluster_selection_method = cluster_selection_method

    def fit(self, X):
        """Fit the HDBSCAN model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _prepare_array(X)
        self._model.fit(X)
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), -1 for noise
        """
        X = _prepare_array(X)
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        """Cluster labels. -1 indicates noise."""
        return self._model.labels_

    @property
    def probabilities_(self):
        """Soft cluster membership probabilities in [0, 1]."""
        return self._model.probabilities_

    @property
    def cluster_persistence_(self):
        """Per-cluster stability scores."""
        return self._model.cluster_persistence_

    def __repr__(self):
        return (
            f"HDBSCAN(min_cluster_size={self._min_cluster_size}, "
            f"min_samples={self._min_samples}, metric=\"{self._metric}\", "
            f"cluster_selection_method=\"{self._cluster_selection_method}\")"
        )

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "min_cluster_size": self._min_cluster_size, "min_samples": self._min_samples,
            "metric": self._metric, "cluster_selection_method": self._cluster_selection_method,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


class AgglomerativeClustering:
    """Agglomerative (hierarchical) clustering backed by a Rust implementation.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters to find.
    linkage : str, default="ward"
        Linkage criterion: "ward", "complete", "average", or "single".
        Ward requires euclidean metric.
    metric : str, default="euclidean"
        Distance metric: "euclidean", "cosine", or "manhattan".
    """

    def __init__(self, n_clusters=2, linkage="ward", metric="euclidean"):
        self._model = _RustAgglomerative(
            n_clusters=n_clusters, linkage=linkage, metric=metric,
        )
        self._n_clusters = n_clusters
        self._linkage = linkage
        self._metric = metric

    def fit(self, X):
        """Fit the model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = _prepare_array(X)
        self._model.fit(X)
        return self

    def fit_predict(self, X):
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        X = _prepare_array(X)
        return self._model.fit_predict(X)

    @property
    def labels_(self):
        """Cluster labels for the training data."""
        return self._model.labels_

    @property
    def n_clusters_(self):
        """Number of clusters found."""
        return self._model.n_clusters_

    @property
    def children_(self):
        """Merge history, shape (n_merges, 2)."""
        return self._model.children_

    @property
    def distances_(self):
        """Distance at each merge."""
        return self._model.distances_

    def __repr__(self):
        return (
            f"AgglomerativeClustering(n_clusters={self._n_clusters}, "
            f"linkage=\"{self._linkage}\", metric=\"{self._metric}\")"
        )

    def __getstate__(self):
        return {"model_state": self._model.__getstate__(), "params": {
            "n_clusters": self._n_clusters, "linkage": self._linkage, "metric": self._metric,
        }}

    def __setstate__(self, state):
        self.__init__(**state["params"])
        self._model.__setstate__(state["model_state"])


# ---------------------------------------------------------------------------
# Clustering metrics
# ---------------------------------------------------------------------------

def silhouette_score(X, labels):
    """Compute the mean silhouette coefficient for all samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)
        Cluster labels. -1 for noise (excluded from computation).

    Returns
    -------
    score : float in [-1, 1]. Higher is better.
    """
    X = _prepare_array(X)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    return _silhouette_score(X, labels)


def calinski_harabasz_score(X, labels):
    """Compute the Calinski-Harabasz index (Variance Ratio Criterion).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)

    Returns
    -------
    score : float. Higher is better.
    """
    X = _prepare_array(X)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    return _calinski_harabasz_score(X, labels)


def davies_bouldin_score(X, labels):
    """Compute the Davies-Bouldin index.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : array-like of shape (n_samples,)

    Returns
    -------
    score : float. Lower is better.
    """
    X = _prepare_array(X)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    return _davies_bouldin_score(X, labels)
