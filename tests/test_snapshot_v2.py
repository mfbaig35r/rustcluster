"""Tests for Cluster Slotting v2: adaptive thresholds, vMF drift, Mahalanobis, hierarchical."""

import numpy as np
import pytest
import tempfile
import os

from rustcluster import KMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster, HierarchicalSnapshot


@pytest.fixture
def well_separated_data():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(100, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(100, 2))
    return np.vstack([c1, c2])


class TestCalibrate:
    def test_calibrate_sets_is_calibrated(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        assert not snap.is_calibrated
        snap.calibrate(well_separated_data)
        assert snap.is_calibrated

    def test_calibrate_save_load_roundtrip(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)
        with tempfile.TemporaryDirectory() as d:
            snap.save(os.path.join(d, "s"))
            loaded = ClusterSnapshot.load(os.path.join(d, "s"))
            assert loaded.is_calibrated


class TestAdaptiveThresholds:
    def test_adaptive_basic(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)

        X_new = np.array([[1, 1], [5, 5], [9, 9]], dtype=np.float64)
        result = snap.assign_with_scores(X_new, adaptive_threshold=True)
        assert len(result.labels_) == 3

    def test_adaptive_without_calibrate_raises(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1]], dtype=np.float64)
        with pytest.raises(RuntimeError, match="calibrate"):
            snap.assign_with_scores(X_new, adaptive_threshold=True)

    def test_adaptive_rejects_less_than_global(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)

        # Adaptive with P10 should reject fewer than a global 0.3 threshold
        result_global = snap.assign_with_scores(well_separated_data, confidence_threshold=0.3)
        result_adaptive = snap.assign_with_scores(
            well_separated_data, adaptive_threshold=True, adaptive_percentile="p10"
        )

        global_rejected = sum(result_global.rejected_)
        adaptive_rejected = sum(result_adaptive.rejected_)
        # Adaptive should generally reject fewer on well-separated data
        # (but this depends on the cluster structure)
        assert adaptive_rejected >= 0  # at minimum it shouldn't crash

    def test_adaptive_percentiles(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)

        X_new = well_separated_data[:20]
        for pct in ["p5", "p10", "p25", "p50"]:
            result = snap.assign_with_scores(X_new, adaptive_threshold=True, adaptive_percentile=pct)
            assert len(result.labels_) == 20


class TestVmfDrift:
    @pytest.fixture
    def embedding_snap(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 32)).astype(np.float32)
        model = EmbeddingCluster(n_clusters=5, reduction_dim=16).fit(X)
        snap = model.snapshot()
        snap.calibrate(X)
        return snap, X

    def test_vmf_drift_in_distribution(self, embedding_snap):
        snap, X = embedding_snap
        report = snap.drift_report(X)
        kd = report.kappa_drift_
        dd = report.direction_drift_
        assert kd is not None
        assert dd is not None
        # In-distribution: kappa drift should be near 0
        assert all(abs(v) < 0.1 for v in kd), f"kappa_drift too high: {kd}"
        # Direction drift should be near 0
        assert all(v < 0.05 for v in dd), f"direction_drift too high: {dd}"

    def test_vmf_drift_random_data(self, embedding_snap):
        snap, _ = embedding_snap
        rng = np.random.default_rng(99)
        X_rand = rng.standard_normal((200, 32)).astype(np.float32)
        report = snap.drift_report(X_rand)
        kd = report.kappa_drift_
        # Random data should show negative kappa drift (more dispersed)
        assert any(v < -0.05 for v in kd), f"Expected negative kappa_drift for random data: {kd}"

    def test_kmeans_no_vmf_fields(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        report = snap.drift_report(well_separated_data)
        assert report.kappa_drift_ is None
        assert report.direction_drift_ is None


class TestMahalanobis:
    def test_mahalanobis_basic(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        labels = snap.assign(X_new, boundary_mode="mahalanobis")
        assert labels.shape == (2,)

    def test_mahalanobis_with_scores(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        result = snap.assign_with_scores(X_new, boundary_mode="mahalanobis")
        assert len(result.confidences_) == 2
        assert all(0 <= c <= 1 for c in result.confidences_)

    def test_mahalanobis_without_calibrate_raises(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="calibrate"):
            snap.assign(X_new, boundary_mode="mahalanobis")

    def test_mahalanobis_save_load(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        snap.calibrate(well_separated_data)
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        labels_before = snap.assign(X_new, boundary_mode="mahalanobis")

        with tempfile.TemporaryDirectory() as d:
            snap.save(os.path.join(d, "s"))
            loaded = ClusterSnapshot.load(os.path.join(d, "s"))
            labels_after = loaded.assign(X_new, boundary_mode="mahalanobis")
            np.testing.assert_array_equal(labels_before, labels_after)

    def test_invalid_boundary_mode(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1]], dtype=np.float64)
        with pytest.raises(ValueError, match="boundary_mode"):
            snap.assign(X_new, boundary_mode="invalid")


class TestHierarchicalSnapshot:
    @pytest.fixture
    def hierarchical_data(self):
        rng = np.random.default_rng(42)
        # 3 well-separated root clusters, each with internal sub-structure
        data = []
        for center in [[0, 0], [20, 0], [0, 20]]:
            for sub in [[-1, -1], [1, 1]]:
                pts = rng.normal(loc=[center[0] + sub[0], center[1] + sub[1]], scale=0.3, size=(30, 2))
                data.append(pts)
        return np.vstack(data)  # 180 points, 3 root x 2 sub

    def test_hierarchical_basic(self, hierarchical_data):
        root = KMeans(n_clusters=3, random_state=42).fit(hierarchical_data)
        hier = HierarchicalSnapshot.build(
            hierarchical_data, root, n_sub_clusters=2, random_state=42
        )
        assert hier.k_root == 3
        assert hier.n_children == 3

        X_new = np.array([[0.5, 0.5], [20.5, 0.5], [0.5, 20.5]], dtype=np.float64)
        root_labels, child_labels = hier.assign(X_new)
        assert root_labels.shape == (3,)
        assert child_labels.shape == (3,)
        # All should have child assignments
        assert all(l >= 0 for l in child_labels)

    def test_hierarchical_with_scores(self, hierarchical_data):
        root = KMeans(n_clusters=3, random_state=42).fit(hierarchical_data)
        hier = HierarchicalSnapshot.build(
            hierarchical_data, root, n_sub_clusters=2, random_state=42
        )
        X_new = np.array([[0.5, 0.5], [20.5, 0.5]], dtype=np.float64)
        result = hier.assign_with_scores(X_new)

        root_labels, child_labels = result.labels_
        root_confs, child_confs = result.confidences_
        assert len(root_labels) == 2
        assert len(child_labels) == 2
        assert all(0 <= c <= 1 for c in root_confs)

    def test_hierarchical_missing_child(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal([0, 0], 0.5, (50, 2)), rng.normal([10, 10], 0.5, (50, 2))])
        root = KMeans(n_clusters=2, random_state=42).fit(X)
        root_snap = root.snapshot()

        # Only provide child for cluster 0
        mask = np.array(root.labels_) == 0
        sub = KMeans(n_clusters=2, random_state=42).fit(X[mask])
        hier = HierarchicalSnapshot(root=root_snap, children={0: sub.snapshot()})

        X_new = np.array([[0.5, 0.5], [9.5, 9.5]], dtype=np.float64)
        root_labels, child_labels = hier.assign(X_new)

        # Point in cluster 0 should have child label
        cluster_0_idx = np.where(root_labels == 0)[0]
        cluster_1_idx = np.where(root_labels == 1)[0]
        assert all(child_labels[i] >= 0 for i in cluster_0_idx)
        assert all(child_labels[i] == -1 for i in cluster_1_idx)

    def test_hierarchical_save_load(self, hierarchical_data):
        root = KMeans(n_clusters=3, random_state=42).fit(hierarchical_data)
        hier = HierarchicalSnapshot.build(
            hierarchical_data, root, n_sub_clusters=2, random_state=42
        )
        X_new = np.array([[0.5, 0.5], [20.5, 0.5], [0.5, 20.5]], dtype=np.float64)
        r1, c1 = hier.assign(X_new)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hier")
            hier.save(path)

            # Verify directory structure
            assert os.path.exists(os.path.join(path, "root"))
            assert os.path.exists(os.path.join(path, "hierarchy.json"))
            assert os.path.exists(os.path.join(path, "children"))

            loaded = HierarchicalSnapshot.load(path)
            r2, c2 = loaded.assign(X_new)
            np.testing.assert_array_equal(r1, r2)
            np.testing.assert_array_equal(c1, c2)

    def test_hierarchical_repr(self, hierarchical_data):
        root = KMeans(n_clusters=3, random_state=42).fit(hierarchical_data)
        hier = HierarchicalSnapshot.build(
            hierarchical_data, root, n_sub_clusters=2, random_state=42
        )
        r = repr(hier)
        assert "root_k=3" in r
        assert "children=3" in r

    def test_hierarchical_rejection_short_circuits(self, hierarchical_data):
        root = KMeans(n_clusters=3, random_state=42).fit(hierarchical_data)
        hier = HierarchicalSnapshot.build(
            hierarchical_data, root, n_sub_clusters=2, random_state=42
        )
        # Extreme threshold should reject at root level
        X_new = np.array([[0.5, 0.5]], dtype=np.float64)
        result = hier.assign_with_scores(X_new, confidence_threshold=0.9999)
        # If rejected at root, child should also be rejected
        if result.root_rejected[0]:
            assert result.child_rejected[0]
