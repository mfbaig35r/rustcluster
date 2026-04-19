"""Integration tests for rustcluster MiniBatchKMeans."""

import numpy as np
import pytest

from rustcluster import MiniBatchKMeans


@pytest.fixture
def blob_data():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    c2 = rng.normal(loc=[20, 20], scale=0.5, size=(50, 2))
    return np.vstack([c1, c2])


@pytest.fixture
def large_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10_000, 8))


class TestFit:
    def test_fit_returns_self(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        assert model.fit(blob_data) is model

    def test_labels_length(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)

    def test_centers_shape(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert model.cluster_centers_.shape == (2, 2)

    def test_inertia_non_negative(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert model.inertia_ >= 0

    def test_n_iter_positive(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42, max_iter=50)
        model.fit(blob_data)
        assert 1 <= model.n_iter_ <= 50

    def test_separates_clusters(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42, max_iter=50)
        model.fit(blob_data)
        assert len(set(model.labels_[:50])) == 1
        assert len(set(model.labels_[50:])) == 1
        assert model.labels_[0] != model.labels_[50]


class TestPredict:
    def test_predict_after_fit(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        preds = model.predict(blob_data)
        np.testing.assert_array_equal(preds, model.labels_)

    def test_predict_before_fit_raises(self):
        model = MiniBatchKMeans(n_clusters=2)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([[1.0, 2.0]]))


class TestFitPredict:
    def test_fit_predict_returns_labels(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(blob_data)
        assert len(labels) == len(blob_data)


class TestReproducibility:
    def test_same_seed_same_result(self, blob_data):
        m1 = MiniBatchKMeans(n_clusters=2, random_state=42, max_iter=20)
        m1.fit(blob_data)
        m2 = MiniBatchKMeans(n_clusters=2, random_state=42, max_iter=20)
        m2.fit(blob_data)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)


class TestValidation:
    def test_n_clusters_zero_raises(self):
        with pytest.raises(ValueError):
            MiniBatchKMeans(n_clusters=0)

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError):
            MiniBatchKMeans(n_clusters=2, batch_size=0)

    def test_max_iter_zero_raises(self):
        with pytest.raises(ValueError):
            MiniBatchKMeans(n_clusters=2, max_iter=0)

    def test_max_no_improvement_zero_raises(self):
        with pytest.raises(ValueError):
            MiniBatchKMeans(n_clusters=2, max_no_improvement=0)

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError, match="metric"):
            MiniBatchKMeans(n_clusters=2, metric="manhattan")

    def test_empty_input_raises(self):
        model = MiniBatchKMeans(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))

    def test_1d_input_raises(self):
        model = MiniBatchKMeans(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0]))


class TestParameters:
    def test_custom_batch_size(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, batch_size=10, random_state=42)
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)

    def test_batch_size_larger_than_n(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        model = MiniBatchKMeans(n_clusters=2, batch_size=1000, random_state=42)
        model.fit(X)
        assert len(model.labels_) == 4

    def test_early_stopping(self, blob_data):
        model = MiniBatchKMeans(
            n_clusters=2, max_iter=1000, tol=1e-3,
            max_no_improvement=3, random_state=42,
        )
        model.fit(blob_data)
        assert model.n_iter_ < 1000


class TestDtypes:
    def test_float32(self, blob_data):
        X = blob_data.astype(np.float32)
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(X)
        assert model.cluster_centers_.dtype == np.float32

    def test_integer_input(self):
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        model = MiniBatchKMeans(n_clusters=2, random_state=42)
        model.fit(X)
        assert len(model.labels_) == 4


class TestCosine:
    def test_cosine_fit(self, blob_data):
        model = MiniBatchKMeans(n_clusters=2, random_state=42, metric="cosine")
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)


class TestRepr:
    def test_repr(self):
        model = MiniBatchKMeans(n_clusters=5, batch_size=512)
        r = repr(model)
        assert "MiniBatchKMeans" in r
        assert "512" in r
