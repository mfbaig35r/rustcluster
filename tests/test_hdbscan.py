"""Integration tests for rustcluster HDBSCAN."""

import numpy as np
import pytest

from rustcluster import HDBSCAN


@pytest.fixture
def two_blobs():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    return np.vstack([c1, c2])


@pytest.fixture
def blobs_with_outlier():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    outlier = np.array([[50.0, 50.0]])
    return np.vstack([c1, c2, outlier])


class TestFit:
    def test_fit_returns_self(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        assert model.fit(two_blobs) is model

    def test_labels_length(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_fit_predict_returns_labels(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        labels = model.fit_predict(two_blobs)
        assert len(labels) == len(two_blobs)

    def test_fit_predict_matches_fit(self, two_blobs):
        m1 = HDBSCAN(min_cluster_size=5)
        m1.fit(two_blobs)

        m2 = HDBSCAN(min_cluster_size=5)
        labels = m2.fit_predict(two_blobs)
        np.testing.assert_array_equal(labels, m1.labels_)


class TestClustering:
    def test_finds_two_clusters(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        unique = set(model.labels_)
        unique.discard(-1)
        assert len(unique) == 2

    def test_noise_for_outlier(self, blobs_with_outlier):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(blobs_with_outlier)
        assert model.labels_[-1] == -1

    def test_all_noise_when_too_few(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0]])
        model = HDBSCAN(min_cluster_size=5)
        model.fit(X)
        assert all(l == -1 for l in model.labels_)


class TestProbabilities:
    def test_probabilities_shape(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        assert len(model.probabilities_) == len(two_blobs)

    def test_probabilities_range(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        assert all(0.0 <= p <= 1.0 for p in model.probabilities_)

    def test_noise_probability_zero(self, blobs_with_outlier):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(blobs_with_outlier)
        noise_mask = model.labels_ == -1
        if noise_mask.any():
            assert all(model.probabilities_[noise_mask] == 0.0)


class TestClusterPersistence:
    def test_persistence_shape(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        n_clusters = len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)
        assert len(model.cluster_persistence_) == n_clusters

    def test_persistence_non_negative(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5)
        model.fit(two_blobs)
        assert all(p >= 0 for p in model.cluster_persistence_)


class TestParameters:
    def test_min_cluster_size_1_raises(self):
        with pytest.raises(ValueError, match="min_cluster_size"):
            HDBSCAN(min_cluster_size=1)

    def test_default_min_samples(self):
        model = HDBSCAN(min_cluster_size=10)
        assert "min_samples=10" in repr(model)

    def test_custom_min_samples(self):
        model = HDBSCAN(min_cluster_size=5, min_samples=3)
        assert "min_samples=3" in repr(model)

    def test_eom_method(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5, cluster_selection_method="eom")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_leaf_method(self, two_blobs):
        model = HDBSCAN(min_cluster_size=5, cluster_selection_method="leaf")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="cluster_selection_method"):
            HDBSCAN(cluster_selection_method="bogus")

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            HDBSCAN(metric="hamming")


class TestValidation:
    def test_empty_input(self):
        model = HDBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))

    def test_1d_input(self):
        model = HDBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0]))

    def test_nan_input(self):
        model = HDBSCAN()
        with pytest.raises(ValueError):
            model.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))


class TestNotFitted:
    def test_labels_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            HDBSCAN().labels_

    def test_probabilities_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            HDBSCAN().probabilities_

    def test_persistence_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            HDBSCAN().cluster_persistence_


class TestDtypes:
    def test_float32(self, two_blobs):
        X = two_blobs.astype(np.float32)
        model = HDBSCAN(min_cluster_size=5)
        model.fit(X)
        assert len(model.labels_) == len(X)

    def test_integer_input(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],
                       [10, 10], [10, 11], [11, 10], [11, 11]])
        model = HDBSCAN(min_cluster_size=2)
        model.fit(X)
        assert len(model.labels_) == 8


class TestDeterminism:
    def test_same_input_same_output(self, two_blobs):
        m1 = HDBSCAN(min_cluster_size=5)
        m1.fit(two_blobs)
        m2 = HDBSCAN(min_cluster_size=5)
        m2.fit(two_blobs)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)
        np.testing.assert_array_equal(m1.probabilities_, m2.probabilities_)


class TestCosine:
    def test_cosine_metric(self):
        rng = np.random.default_rng(42)
        dir1 = rng.normal(loc=[1, 0], scale=0.05, size=(20, 2))
        dir2 = rng.normal(loc=[0, 1], scale=0.05, size=(20, 2))
        X = np.vstack([dir1, dir2])
        model = HDBSCAN(min_cluster_size=5, metric="cosine")
        model.fit(X)
        assert len(model.labels_) == 40


class TestRepr:
    def test_repr(self):
        model = HDBSCAN(min_cluster_size=10, min_samples=3)
        r = repr(model)
        assert "HDBSCAN" in r
        assert "10" in r
        assert "3" in r
