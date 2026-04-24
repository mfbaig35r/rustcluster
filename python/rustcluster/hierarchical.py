"""Hierarchical cluster slotting: cascading snapshots for multi-level assignment."""

import os
import json
import numpy as np

from rustcluster.snapshot import ClusterSnapshot

__all__ = ["HierarchicalSnapshot"]


def _prepare_array(X):
    """Ensure input is a 2D C-contiguous float32 or float64 array."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got {X.ndim}D")
    if X.dtype == np.float32:
        return np.ascontiguousarray(X, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float64)


class HierarchicalSnapshot:
    """Two-level hierarchical slotting: root clusters -> sub-clusters.

    Assigns points first to a root-level snapshot, then routes each point
    to a child snapshot within its assigned root cluster.

    Parameters
    ----------
    root : ClusterSnapshot
        Top-level snapshot (coarse clusters).
    children : dict[int, ClusterSnapshot], optional
        Per-root-cluster child snapshots. Keys are root cluster indices.
        Clusters without a child snapshot get child_label=-1.
    """

    def __init__(self, root, children=None):
        self.root = root
        self.children = children or {}

    def assign(self, X, **kwargs):
        """Cascading assignment: root -> child.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        **kwargs
            Forwarded to root and child ``assign()`` calls (e.g., boundary_mode).

        Returns
        -------
        root_labels : ndarray of shape (n_samples,), dtype int64
        child_labels : ndarray of shape (n_samples,), dtype int64
            -1 if no child snapshot for that root cluster.
        """
        X = _prepare_array(X)
        n = X.shape[0]

        root_labels = self.root.assign(X, **kwargs)
        child_labels = np.full(n, -1, dtype=np.int64)

        for cluster_id, child_snap in self.children.items():
            mask = root_labels == cluster_id
            if not np.any(mask):
                continue
            X_subset = X[mask]
            child_labels[mask] = child_snap.assign(X_subset, **kwargs)

        return root_labels, child_labels

    def assign_with_scores(self, X, **kwargs):
        """Cascading assignment with full scoring at both levels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        **kwargs
            Forwarded to ``assign_with_scores()`` at both levels.

        Returns
        -------
        HierarchicalAssignmentResult
        """
        X = _prepare_array(X)
        n = X.shape[0]

        root_result = self.root.assign_with_scores(X, **kwargs)
        root_labels = np.array(root_result.labels_)
        root_confs = np.array(root_result.confidences_)
        root_rejected = np.array(root_result.rejected_)

        child_labels = np.full(n, -1, dtype=np.int64)
        child_confs = np.zeros(n, dtype=np.float64)
        child_rejected = np.ones(n, dtype=bool)  # default: rejected (no child)

        for cluster_id, child_snap in self.children.items():
            # Only slot non-rejected points into this cluster
            mask = (root_labels == cluster_id) & (~root_rejected)
            if not np.any(mask):
                continue
            X_subset = X[mask]
            child_result = child_snap.assign_with_scores(X_subset, **kwargs)
            child_labels[mask] = child_result.labels_
            child_confs[mask] = child_result.confidences_
            child_rejected[mask] = child_result.rejected_

        return HierarchicalAssignmentResult(
            root_labels=root_labels,
            root_confidences=root_confs,
            root_rejected=root_rejected,
            child_labels=child_labels,
            child_confidences=child_confs,
            child_rejected=child_rejected,
        )

    def save(self, path):
        """Save hierarchical snapshot to directory.

        Creates:
            {path}/root/          — root snapshot
            {path}/children/0/    — child snapshot for root cluster 0
            {path}/children/1/    — child snapshot for root cluster 1
            ...
            {path}/hierarchy.json — metadata (which children exist)
        """
        path = str(path)
        os.makedirs(path, exist_ok=True)

        self.root.save(os.path.join(path, "root"))

        children_dir = os.path.join(path, "children")
        os.makedirs(children_dir, exist_ok=True)

        child_ids = sorted(self.children.keys())
        for cid in child_ids:
            self.children[cid].save(os.path.join(children_dir, str(cid)))

        meta = {"child_ids": child_ids}
        with open(os.path.join(path, "hierarchy.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path):
        """Load hierarchical snapshot from directory."""
        path = str(path)

        root = ClusterSnapshot.load(os.path.join(path, "root"))

        with open(os.path.join(path, "hierarchy.json")) as f:
            meta = json.load(f)

        children = {}
        for cid in meta["child_ids"]:
            children[cid] = ClusterSnapshot.load(
                os.path.join(path, "children", str(cid))
            )

        return cls(root=root, children=children)

    @classmethod
    def build(cls, X, root_model, n_sub_clusters=5, **fit_kwargs):
        """Convenience: fit sub-clusters per root cluster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (same data used to fit root_model).
        root_model : fitted clustering model
            Must have ``.snapshot()`` and ``.labels_`` attributes.
        n_sub_clusters : int or dict, default=5
            Number of sub-clusters per root cluster. If dict, maps
            root cluster ID to sub-cluster count.
        **fit_kwargs
            Forwarded to sub-cluster model constructors (e.g., random_state).

        Returns
        -------
        HierarchicalSnapshot
        """
        from rustcluster.experimental import EmbeddingCluster

        X = _prepare_array(X)
        root_snap = root_model.snapshot()
        root_labels = np.array(root_model.labels_)
        k_root = root_snap.k

        children = {}
        for cid in range(k_root):
            mask = root_labels == cid
            X_sub = X[mask]
            n_sub = X_sub.shape[0]

            if isinstance(n_sub_clusters, dict):
                k_sub = n_sub_clusters.get(cid, 5)
            else:
                k_sub = n_sub_clusters

            # Need at least k_sub * 2 points
            if n_sub < k_sub * 2:
                k_sub = max(1, n_sub // 2)

            if k_sub < 2 or n_sub < 4:
                continue

            sub_model = EmbeddingCluster(
                n_clusters=k_sub,
                reduction_dim=None,
                **fit_kwargs,
            )
            sub_model.fit(X_sub)
            children[cid] = sub_model.snapshot()

        return cls(root=root_snap, children=children)

    @property
    def k_root(self):
        """Number of root-level clusters."""
        return self.root.k

    @property
    def n_children(self):
        """Number of child snapshots."""
        return len(self.children)

    def __repr__(self):
        child_ks = {cid: s.k for cid, s in self.children.items()}
        return (
            f"HierarchicalSnapshot(root_k={self.root.k}, "
            f"children={len(self.children)}, child_ks={child_ks})"
        )


class HierarchicalAssignmentResult:
    """Result of hierarchical assignment at two levels."""

    def __init__(self, root_labels, root_confidences, root_rejected,
                 child_labels, child_confidences, child_rejected):
        self.root_labels = root_labels
        self.root_confidences = root_confidences
        self.root_rejected = root_rejected
        self.child_labels = child_labels
        self.child_confidences = child_confidences
        self.child_rejected = child_rejected

    @property
    def labels_(self):
        """Tuple of (root_labels, child_labels) arrays."""
        return self.root_labels, self.child_labels

    @property
    def confidences_(self):
        """Tuple of (root_confidences, child_confidences) arrays."""
        return self.root_confidences, self.child_confidences

    @property
    def rejected_(self):
        """Boolean array — True if rejected at either level."""
        return self.root_rejected | self.child_rejected

    def __repr__(self):
        n = len(self.root_labels)
        rej = self.rejected_.sum()
        return f"HierarchicalAssignmentResult(n={n}, rejected={rej})"
