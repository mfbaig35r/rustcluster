"""Tests for ClusterSnapshot: snapshot, save/load, assign, drift."""

import numpy as np
import pytest
import tempfile
import os

from rustcluster import KMeans, MiniBatchKMeans, ClusterSnapshot
from rustcluster.experimental import EmbeddingCluster


@pytest.fixture
def well_separated_data():
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(50, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(50, 2))
    return np.vstack([c1, c2])


class TestKMeansSnapshot:
    def test_snapshot_creates(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        assert snap.k == 2
        assert snap.d == 2
        assert snap.input_dim == 2
        assert snap.algorithm == "kmeans"
        assert snap.metric == "euclidean"
        assert not snap.spherical

    def test_assign_matches_predict(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        np.testing.assert_array_equal(snap.assign(X_new), m.predict(X_new))

    def test_assign_with_scores(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        result = snap.assign_with_scores(X_new)
        assert len(result.labels_) == 2
        assert len(result.distances_) == 2
        assert len(result.confidences_) == 2
        assert all(0 <= c <= 1 for c in result.confidences_)
        assert all(not r for r in result.rejected_)

    def test_high_confidence_for_close_points(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_close = np.array([[0.1, 0.1]], dtype=np.float64)
        result = snap.assign_with_scores(X_close)
        assert result.confidences_[0] > 0.9

    def test_rejection_by_distance(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_far = np.array([[1000, 1000]], dtype=np.float64)
        result = snap.assign_with_scores(X_far, distance_threshold=100.0)
        assert result.rejected_[0]
        assert result.labels_[0] == -1

    def test_rejection_by_confidence(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        # Point equidistant between clusters
        c0 = m.cluster_centers_[0]
        c1 = m.cluster_centers_[1]
        midpoint = ((c0 + c1) / 2).reshape(1, -1)
        result = snap.assign_with_scores(midpoint, confidence_threshold=0.5)
        assert result.rejected_[0]

    def test_save_load_roundtrip(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "snap")
            snap.save(path)
            loaded = ClusterSnapshot.load(path)
            assert loaded.k == snap.k
            assert loaded.d == snap.d
            assert loaded.input_dim == snap.input_dim
            assert loaded.algorithm == snap.algorithm
            X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
            np.testing.assert_array_equal(loaded.assign(X_new), snap.assign(X_new))

    def test_cosine_metric(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 8))
        m = KMeans(n_clusters=3, metric="cosine", random_state=42).fit(X)
        snap = m.snapshot()
        assert snap.metric == "cosine"
        labels = snap.assign(X[:5])
        assert labels.shape == (5,)

    def test_f32_input(self, well_separated_data):
        X32 = well_separated_data.astype(np.float32)
        m = KMeans(n_clusters=2, random_state=42).fit(X32)
        snap = m.snapshot()
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float32)
        labels = snap.assign(X_new)
        assert labels.shape == (2,)

    def test_unfitted_raises(self):
        m = KMeans(n_clusters=2)
        with pytest.raises(RuntimeError, match="not been fitted"):
            m.snapshot()

    def test_dimension_mismatch(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_bad = np.array([[1, 2, 3]], dtype=np.float64)
        with pytest.raises(ValueError, match="dimension"):
            snap.assign(X_bad)

    def test_repr(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        r = repr(snap)
        assert "kmeans" in r
        assert "k=2" in r


class TestMiniBatchSnapshot:
    def test_snapshot_and_assign(self, well_separated_data):
        m = MiniBatchKMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        assert snap.algorithm == "minibatch_kmeans"
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        labels = snap.assign(X_new)
        assert labels.shape == (2,)
        assert set(labels.tolist()).issubset({0, 1})

    def test_assign_matches_predict(self, well_separated_data):
        m = MiniBatchKMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
        np.testing.assert_array_equal(snap.assign(X_new), m.predict(X_new))

    def test_save_load(self, well_separated_data):
        m = MiniBatchKMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mb_snap")
            snap.save(path)
            loaded = ClusterSnapshot.load(path)
            X_new = np.array([[1, 1], [9, 9]], dtype=np.float64)
            np.testing.assert_array_equal(loaded.assign(X_new), snap.assign(X_new))


class TestEmbeddingClusterSnapshot:
    @pytest.fixture
    def embedding_data(self):
        rng = np.random.default_rng(42)
        centers = rng.standard_normal((3, 64))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        data = []
        for i in range(3):
            noise = rng.standard_normal((30, 64)) * 0.1
            data.append(centers[i] + noise)
        return np.vstack(data).astype(np.float32)

    def test_snapshot_with_pca(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        snap = m.snapshot()
        assert snap.k == 3
        assert snap.d == 16  # reduced dim
        assert snap.input_dim == 64  # original dim
        assert snap.algorithm == "embedding_cluster"
        assert snap.spherical

    def test_assign_new_embeddings(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        snap = m.snapshot()
        rng = np.random.default_rng(99)
        X_new = rng.standard_normal((5, 64)).astype(np.float32)
        labels = snap.assign(X_new)
        assert labels.shape == (5,)
        assert all(0 <= l < 3 for l in labels)

    def test_assign_with_scores(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        snap = m.snapshot()
        X_new = embedding_data[:5]
        result = snap.assign_with_scores(X_new)
        assert len(result.labels_) == 5
        assert all(0 <= c <= 1 for c in result.confidences_)

    def test_save_load_with_pca(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        snap = m.snapshot()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "emb_snap")
            snap.save(path)
            loaded = ClusterSnapshot.load(path)
            assert loaded.k == 3
            assert loaded.d == 16
            assert loaded.input_dim == 64
            rng = np.random.default_rng(99)
            X_new = rng.standard_normal((5, 64)).astype(np.float32)
            np.testing.assert_array_equal(loaded.assign(X_new), snap.assign(X_new))

    def test_no_reduction(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 8)).astype(np.float32)
        m = EmbeddingCluster(n_clusters=2, reduction_dim=None).fit(X)
        snap = m.snapshot()
        assert snap.d == 8
        assert snap.input_dim == 8
        labels = snap.assign(X[:3])
        assert labels.shape == (3,)

    def test_dimension_mismatch(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        snap = m.snapshot()
        X_bad = np.random.default_rng(42).standard_normal((5, 32)).astype(np.float32)
        with pytest.raises(ValueError, match="dimension"):
            snap.assign(X_bad)


class TestDriftReport:
    def test_same_data_low_drift(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        report = snap.drift_report(well_separated_data)
        assert report.n_samples_ == 100
        assert report.global_mean_distance_ < 1.0  # close to centroids

    def test_shifted_data_high_drift(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        shifted = well_separated_data + 50.0
        report = snap.drift_report(shifted)
        assert report.global_mean_distance_ > 100.0

    def test_cluster_sizes_sum_to_n(self, well_separated_data):
        m = KMeans(n_clusters=2, random_state=42).fit(well_separated_data)
        snap = m.snapshot()
        report = snap.drift_report(well_separated_data)
        assert sum(report.new_cluster_sizes_) == 100

    def test_embedding_drift(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((90, 64)).astype(np.float32)
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(X)
        snap = m.snapshot()
        report = snap.drift_report(X)
        assert report.n_samples_ == 90
