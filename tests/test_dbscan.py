"""Integration tests for rustcluster DBSCAN."""

import numpy as np
import pytest

from rustcluster import DBSCAN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_blobs():
    """Two well-separated dense clusters in 2D."""
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    return np.vstack([c1, c2])


@pytest.fixture
def blobs_with_outlier():
    """Two clusters plus one far-away outlier."""
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(20, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(20, 2))
    outlier = np.array([[50.0, 50.0]])
    return np.vstack([c1, c2, outlier])


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------

class TestBasic:
    def test_fit_returns_self(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        result = model.fit(two_blobs)
        assert result is model

    def test_labels_length(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_fit_predict_returns_labels(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        labels = model.fit_predict(two_blobs)
        assert len(labels) == len(two_blobs)

    def test_fit_predict_matches_labels_(self, two_blobs):
        m1 = DBSCAN(eps=1.0, min_samples=3)
        m1.fit(two_blobs)

        m2 = DBSCAN(eps=1.0, min_samples=3)
        labels = m2.fit_predict(two_blobs)

        np.testing.assert_array_equal(labels, m1.labels_)

    def test_core_sample_indices_exist(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(two_blobs)
        assert len(model.core_sample_indices_) > 0

    def test_components_shape(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(two_blobs)
        n_core = len(model.core_sample_indices_)
        assert model.components_.shape == (n_core, 2)

    def test_repr(self):
        model = DBSCAN(eps=0.3, min_samples=10)
        assert "DBSCAN" in repr(model)
        assert "0.3" in repr(model)
        assert "10" in repr(model)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

class TestCorrectness:
    def test_two_clusters_found(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(two_blobs)
        unique = set(model.labels_)
        unique.discard(-1)
        assert len(unique) == 2

    def test_clusters_partition_correctly(self, two_blobs):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(two_blobs)
        labels = model.labels_
        # First 30 should share one cluster, last 30 another
        c1 = set(labels[:30])
        c2 = set(labels[30:])
        c1.discard(-1)
        c2.discard(-1)
        assert len(c1) == 1
        assert len(c2) == 1
        assert c1 != c2

    def test_noise_detection(self, blobs_with_outlier):
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(blobs_with_outlier)
        # Last point is the outlier
        assert model.labels_[-1] == -1

    def test_chain_clustering(self):
        """Points in a chain should form one cluster."""
        X = np.array([[i * 0.4, 0.0] for i in range(10)])
        model = DBSCAN(eps=0.5, min_samples=2)
        model.fit(X)
        non_noise = [l for l in model.labels_ if l >= 0]
        assert len(set(non_noise)) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_noise(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        model = DBSCAN(eps=0.1, min_samples=2)
        model.fit(X)
        assert all(l == -1 for l in model.labels_)

    def test_all_one_cluster(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])
        model = DBSCAN(eps=1.0, min_samples=2)
        model.fit(X)
        assert all(l == 0 for l in model.labels_)

    def test_min_samples_one(self):
        """Every point is core when min_samples=1."""
        X = np.array([[0.0, 0.0], [100.0, 100.0]])
        model = DBSCAN(eps=0.1, min_samples=1)
        model.fit(X)
        # Each point is its own cluster
        assert all(l >= 0 for l in model.labels_)
        assert len(model.core_sample_indices_) == 2

    def test_large_eps(self):
        """Very large eps → all in one cluster."""
        X = np.array([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])
        model = DBSCAN(eps=1000.0, min_samples=2)
        model.fit(X)
        assert all(l == 0 for l in model.labels_)

    def test_single_point_noise(self):
        X = np.array([[1.0, 2.0]])
        model = DBSCAN(eps=1.0, min_samples=2)
        model.fit(X)
        assert model.labels_[0] == -1

    def test_identical_points(self):
        X = np.full((5, 2), 3.0)
        model = DBSCAN(eps=0.1, min_samples=2)
        model.fit(X)
        assert all(l == 0 for l in model.labels_)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_eps_zero(self):
        with pytest.raises(ValueError, match="eps"):
            DBSCAN(eps=0.0)

    def test_invalid_eps_negative(self):
        with pytest.raises(ValueError, match="eps"):
            DBSCAN(eps=-1.0)

    def test_invalid_min_samples_zero(self):
        with pytest.raises(ValueError, match="min_samples"):
            DBSCAN(min_samples=0)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            DBSCAN(metric="hamming")

    def test_empty_input(self):
        model = DBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))

    def test_nan_input(self):
        model = DBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

    def test_1d_input(self):
        model = DBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Properties before fit
# ---------------------------------------------------------------------------

class TestNotFitted:
    def test_labels_before_fit(self):
        model = DBSCAN()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.labels_

    def test_core_sample_indices_before_fit(self):
        model = DBSCAN()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.core_sample_indices_

    def test_components_before_fit(self):
        model = DBSCAN()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.components_


# ---------------------------------------------------------------------------
# f32 support
# ---------------------------------------------------------------------------

class TestFloat32:
    def test_f32_fit(self, two_blobs):
        X = two_blobs.astype(np.float32)
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(X)
        assert len(model.labels_) == len(X)

    def test_f32_components_dtype(self, two_blobs):
        X = two_blobs.astype(np.float32)
        model = DBSCAN(eps=1.0, min_samples=3)
        model.fit(X)
        assert model.components_.dtype == np.float32

    def test_f32_matches_f64(self, two_blobs):
        m64 = DBSCAN(eps=1.0, min_samples=3)
        m64.fit(two_blobs)

        m32 = DBSCAN(eps=1.0, min_samples=3)
        m32.fit(two_blobs.astype(np.float32))

        np.testing.assert_array_equal(m64.labels_, m32.labels_)

    def test_integer_input(self):
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
        model = DBSCAN(eps=2.0, min_samples=2)
        model.fit(X)
        assert len(model.labels_) == 6


# ---------------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------------

class TestCosineDBSCAN:
    def test_cosine_fit(self, two_blobs):
        model = DBSCAN(eps=0.1, min_samples=3, metric="cosine")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_cosine_separates_directions(self):
        """Cosine DBSCAN should cluster by direction."""
        rng = np.random.default_rng(42)
        dir1 = rng.normal(loc=[1, 0], scale=0.05, size=(20, 2))
        dir2 = rng.normal(loc=[0, 1], scale=0.05, size=(20, 2))
        X = np.vstack([dir1, dir2])

        model = DBSCAN(eps=0.1, min_samples=3, metric="cosine")
        model.fit(X)
        unique = set(model.labels_)
        unique.discard(-1)
        assert len(unique) == 2

    def test_cosine_metric_in_repr(self):
        model = DBSCAN(metric="cosine")
        assert 'metric="cosine"' in repr(model)
