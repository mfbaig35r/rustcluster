"""Cluster snapshot for incremental assignment without re-clustering."""

import numpy as np

from rustcluster._rustcluster import ClusterSnapshot as _RustClusterSnapshot


def _prepare_array(X):
    """Ensure input is a 2D C-contiguous float32 or float64 array."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.dtype == np.float32:
        return np.ascontiguousarray(X, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float64)

__all__ = ["ClusterSnapshot"]


class ClusterSnapshot:
    """Frozen cluster state for assigning new points without re-clustering.

    Create via ``model.snapshot()`` after ``fit()``, or load from disk
    with ``ClusterSnapshot.load(path)``.

    Parameters
    ----------
    _rust_snapshot : internal Rust object
        Do not construct directly. Use ``model.snapshot()`` or ``ClusterSnapshot.load()``.
    """

    def __init__(self, _rust_snapshot):
        self._snap = _rust_snapshot

    def assign(self, X):
        """Assign new points to the nearest cluster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,), dtype int64
        """
        X = _prepare_array(X)
        return self._snap.assign(X)

    def assign_with_scores(self, X, *, distance_threshold=None, confidence_threshold=None):
        """Assign with confidence scores and optional rejection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        distance_threshold : float, optional
            Reject points farther than this from all centroids.
        confidence_threshold : float, optional
            Reject points with confidence below this (0-1).

        Returns
        -------
        result : AssignmentResult
            Access ``.labels_``, ``.distances_``, ``.confidences_``, ``.rejected_``.
        """
        X = _prepare_array(X)
        return self._snap.assign_with_scores(
            X,
            distance_threshold=distance_threshold,
            confidence_threshold=confidence_threshold,
        )

    def drift_report(self, X):
        """Compute drift statistics against new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        report : DriftReport
            Access ``.global_mean_distance_``, ``.relative_drift_``, etc.
        """
        X = _prepare_array(X)
        return self._snap.drift_report(X)

    def save(self, path):
        """Save snapshot to directory (safetensors + metadata.json)."""
        self._snap.save(str(path))

    @classmethod
    def load(cls, path):
        """Load snapshot from directory."""
        return cls(_RustClusterSnapshot.load(str(path)))

    @property
    def k(self):
        """Number of clusters."""
        return self._snap.k

    @property
    def d(self):
        """Centroid dimensionality (after preprocessing)."""
        return self._snap.d

    @property
    def input_dim(self):
        """Expected input dimensionality."""
        return self._snap.input_dim

    @property
    def algorithm(self):
        """Algorithm that produced this snapshot."""
        return self._snap.algorithm

    @property
    def metric(self):
        """Distance metric."""
        return self._snap.metric

    @property
    def spherical(self):
        """Whether assignment uses spherical (max-dot) logic."""
        return self._snap.spherical

    def __repr__(self):
        return (
            f"ClusterSnapshot(algorithm=\"{self.algorithm}\", k={self.k}, "
            f"d={self.d}, input_dim={self.input_dim})"
        )
