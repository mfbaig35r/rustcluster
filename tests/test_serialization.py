"""Tests for pickle serialization of all clustering models."""

import copy
import pickle

import numpy as np
import pytest

from rustcluster import (
    KMeans, MiniBatchKMeans, DBSCAN, HDBSCAN, AgglomerativeClustering,
)
from rustcluster.experimental import EmbeddingCluster


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(20, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(20, 2))
    return np.vstack([c1, c2])


class TestKMeansPickle:
    def test_roundtrip_fitted(self, data):
        m = KMeans(n_clusters=2, random_state=42).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)
        np.testing.assert_allclose(m.cluster_centers_, m2.cluster_centers_)
        assert abs(m.inertia_ - m2.inertia_) < 1e-10

    def test_roundtrip_unfitted(self):
        m = KMeans(n_clusters=3, max_iter=50)
        m2 = pickle.loads(pickle.dumps(m))
        assert repr(m2) == repr(m)

    def test_predict_after_unpickle(self, data):
        m = KMeans(n_clusters=2, random_state=42).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.predict(data), m2.predict(data))

    def test_deepcopy(self, data):
        m = KMeans(n_clusters=2, random_state=42).fit(data)
        m2 = copy.deepcopy(m)
        np.testing.assert_array_equal(m.labels_, m2.labels_)

    def test_f32_preserves_dtype(self, data):
        X = data.astype(np.float32)
        m = KMeans(n_clusters=2, random_state=42).fit(X)
        m2 = pickle.loads(pickle.dumps(m))
        assert m2.cluster_centers_.dtype == np.float32


class TestMiniBatchPickle:
    def test_roundtrip(self, data):
        m = MiniBatchKMeans(n_clusters=2, random_state=42).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)
        np.testing.assert_allclose(m.cluster_centers_, m2.cluster_centers_)


class TestDBSCANPickle:
    def test_roundtrip(self, data):
        m = DBSCAN(eps=1.0, min_samples=3).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)

    def test_core_indices_preserved(self, data):
        m = DBSCAN(eps=1.0, min_samples=3).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.core_sample_indices_, m2.core_sample_indices_)


class TestHDBSCANPickle:
    def test_roundtrip(self, data):
        m = HDBSCAN(min_cluster_size=5).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)
        np.testing.assert_array_equal(m.probabilities_, m2.probabilities_)

    def test_persistence_preserved(self, data):
        m = HDBSCAN(min_cluster_size=5).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.cluster_persistence_, m2.cluster_persistence_)


class TestAgglomerativePickle:
    def test_roundtrip(self, data):
        m = AgglomerativeClustering(n_clusters=2).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)

    def test_children_preserved(self, data):
        m = AgglomerativeClustering(n_clusters=2).fit(data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.children_, m2.children_)
        np.testing.assert_allclose(m.distances_, m2.distances_)


class TestEmbeddingClusterPickle:
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

    def test_roundtrip_fitted(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(embedding_data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)
        np.testing.assert_allclose(m.cluster_centers_, m2.cluster_centers_)
        assert abs(m.objective_ - m2.objective_) < 1e-10

    def test_roundtrip_unfitted(self):
        m = EmbeddingCluster(n_clusters=5, reduction_dim=32)
        m2 = pickle.loads(pickle.dumps(m))
        assert repr(m2) == repr(m)

    def test_roundtrip_with_pca(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=16).fit(embedding_data)
        m2 = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(m.labels_, m2.labels_)
        np.testing.assert_allclose(m.cluster_centers_, m2.cluster_centers_)

    def test_deepcopy(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(embedding_data)
        m2 = copy.deepcopy(m)
        np.testing.assert_array_equal(m.labels_, m2.labels_)

    def test_no_training_data_in_pickle(self, embedding_data):
        m = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(embedding_data)
        blob = pickle.dumps(m)
        # Pickle should be much smaller than the training data
        assert len(blob) < embedding_data.nbytes
