"""Integration tests for rustcluster KMeans."""

import numpy as np
import pytest

from rustcluster import KMeans


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def blob_data():
    """Two well-separated clusters in 2D."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    cluster_b = rng.normal(loc=[20, 20], scale=0.5, size=(50, 2))
    return np.vstack([cluster_a, cluster_b])


@pytest.fixture
def simple_data():
    """Minimal 4-point dataset."""
    return np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [8.0, 8.0],
        [8.2, 8.1],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Basic fit behavior
# ---------------------------------------------------------------------------

class TestFit:
    def test_fit_returns_self(self, simple_data):
        model = KMeans(n_clusters=2, random_state=42)
        result = model.fit(simple_data)
        assert result is model

    def test_labels_length(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)

    def test_centers_shape(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert model.cluster_centers_.shape == (2, 2)

    def test_inertia_non_negative(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert model.inertia_ >= 0

    def test_n_iter_populated(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert 1 <= model.n_iter_ <= 300

    def test_separates_clusters(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        labels = model.labels_
        # First 50 should share a label, last 50 should share a different one
        assert len(set(labels[:50])) == 1
        assert len(set(labels[50:])) == 1
        assert labels[0] != labels[50]


# ---------------------------------------------------------------------------
# Predict behavior
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_after_fit(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        preds = model.predict(blob_data)
        np.testing.assert_array_equal(preds, model.labels_)

    def test_predict_before_fit_raises(self):
        model = KMeans(n_clusters=2)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(np.array([[1.0, 2.0]]))


# ---------------------------------------------------------------------------
# fit_predict
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_fit_predict_returns_labels(self, simple_data):
        model = KMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(simple_data)
        assert len(labels) == len(simple_data)

    def test_fit_predict_matches_fit_then_labels(self, blob_data):
        m1 = KMeans(n_clusters=2, random_state=42)
        m1.fit(blob_data)

        m2 = KMeans(n_clusters=2, random_state=42)
        labels = m2.fit_predict(blob_data)

        np.testing.assert_array_equal(labels, m1.labels_)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_result(self, blob_data):
        m1 = KMeans(n_clusters=2, random_state=42)
        m1.fit(blob_data)

        m2 = KMeans(n_clusters=2, random_state=42)
        m2.fit(blob_data)

        np.testing.assert_array_equal(m1.labels_, m2.labels_)
        assert abs(m1.inertia_ - m2.inertia_) < 1e-10
        assert m1.n_iter_ == m2.n_iter_

    def test_different_seed_may_differ(self):
        """Different seeds can produce different results on ambiguous data."""
        rng = np.random.default_rng(99)
        X = rng.standard_normal((100, 2))
        m1 = KMeans(n_clusters=5, random_state=0, n_init=1)
        m1.fit(X)
        m2 = KMeans(n_clusters=5, random_state=999, n_init=1)
        m2.fit(X)
        # Not guaranteed to differ, but with 5 clusters on random data, very likely
        # At minimum, both should produce valid results
        assert len(m1.labels_) == 100
        assert len(m2.labels_) == 100


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_1d_input_raises(self):
        model = KMeans(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_empty_input_raises(self):
        model = KMeans(n_clusters=2)
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))

    def test_k_greater_than_n_raises(self):
        model = KMeans(n_clusters=10)
        with pytest.raises(ValueError):
            model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_nan_input_raises(self):
        model = KMeans(n_clusters=2)
        X = np.array([[1.0, 2.0], [np.nan, 4.0]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_inf_input_raises(self):
        model = KMeans(n_clusters=2)
        X = np.array([[1.0, 2.0], [np.inf, 4.0]])
        with pytest.raises(ValueError):
            model.fit(X)

    def test_dimension_mismatch_predict(self, simple_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(simple_data)  # 2 features
        with pytest.raises(ValueError, match="dimension mismatch"):
            model.predict(np.array([[1.0, 2.0, 3.0]]))  # 3 features

    def test_invalid_n_clusters_zero(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=0)

    def test_invalid_max_iter_zero(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=2, max_iter=0)

    def test_invalid_n_init_zero(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=2, n_init=0)

    def test_invalid_tol_negative(self):
        with pytest.raises(ValueError):
            KMeans(n_clusters=2, tol=-1.0)


# ---------------------------------------------------------------------------
# Input normalization (Python wrapper handles conversion)
# ---------------------------------------------------------------------------

class TestInputNormalization:
    def test_fortran_order_works(self, simple_data):
        X = np.asfortranarray(simple_data)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)  # Wrapper converts to C-contiguous
        assert len(model.labels_) == len(X)

    def test_float32_works(self, simple_data):
        X = simple_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)  # Native f32 path
        assert len(model.labels_) == len(X)
        assert model.cluster_centers_.dtype == np.float32

    def test_integer_input_works(self):
        X = np.array([[1, 2], [3, 4], [10, 20], [12, 22]])
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)  # Wrapper converts to float64
        assert len(model.labels_) == 4


# ---------------------------------------------------------------------------
# n_init behavior
# ---------------------------------------------------------------------------

class TestNInit:
    def test_n_init_best_inertia(self, blob_data):
        m1 = KMeans(n_clusters=2, random_state=42, n_init=1)
        m1.fit(blob_data)

        m10 = KMeans(n_clusters=2, random_state=42, n_init=10)
        m10.fit(blob_data)

        # More initializations should find equal or better inertia
        assert m10.inertia_ <= m1.inertia_ + 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_k_equals_n(self):
        X = np.array([[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]])
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X)
        assert sorted(model.labels_) == [0, 1, 2]
        assert model.inertia_ < 1e-10

    def test_single_cluster(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = KMeans(n_clusters=1, random_state=42)
        model.fit(X)
        assert all(l == 0 for l in model.labels_)
        # Centroid should be the mean
        np.testing.assert_allclose(model.cluster_centers_[0], [3.0, 4.0])

    def test_identical_points(self):
        X = np.full((10, 3), 5.0)
        model = KMeans(n_clusters=1, random_state=42)
        model.fit(X)
        assert model.inertia_ < 1e-10
        assert model.n_iter_ == 1

    def test_large_n_clusters(self):
        """Stress test with k close to n."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        model = KMeans(n_clusters=19, random_state=42, n_init=1)
        model.fit(X)
        assert len(set(model.labels_)) <= 19

    def test_high_dimensional(self):
        """Works with many features."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 50))
        model = KMeans(n_clusters=3, random_state=42, n_init=1)
        model.fit(X)
        assert model.cluster_centers_.shape == (3, 50)


# ---------------------------------------------------------------------------
# Algorithm selection (Hamerly)
# ---------------------------------------------------------------------------

class TestAlgorithm:
    def test_lloyd_explicit(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, algorithm="lloyd")
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)
        assert model.inertia_ >= 0

    def test_hamerly_explicit(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)
        assert model.inertia_ >= 0

    def test_auto_default(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)

    def test_hamerly_matches_lloyd(self, blob_data):
        """Both algorithms should produce the same partitioning on well-separated data."""
        m_lloyd = KMeans(n_clusters=2, random_state=42, n_init=1, algorithm="lloyd")
        m_lloyd.fit(blob_data)

        m_hamerly = KMeans(n_clusters=2, random_state=42, n_init=1, algorithm="hamerly")
        m_hamerly.fit(blob_data)

        np.testing.assert_array_equal(m_lloyd.labels_, m_hamerly.labels_)
        assert abs(m_lloyd.inertia_ - m_hamerly.inertia_) < 1e-4

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            KMeans(n_clusters=2, algorithm="bogus")

    def test_hamerly_high_d_high_k(self):
        """Hamerly works on larger configurations."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((500, 32))
        model = KMeans(n_clusters=16, random_state=42, n_init=1, algorithm="hamerly")
        model.fit(X)
        assert model.cluster_centers_.shape == (16, 32)
        assert model.inertia_ >= 0

    def test_hamerly_single_cluster_falls_back(self):
        """k=1 with hamerly should fall back to lloyd (needs k>=2)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        model = KMeans(n_clusters=1, random_state=42, algorithm="hamerly")
        model.fit(X)
        assert all(l == 0 for l in model.labels_)

    def test_hamerly_predict(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        model.fit(blob_data)
        preds = model.predict(blob_data)
        np.testing.assert_array_equal(preds, model.labels_)

    def test_hamerly_fit_predict(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        labels = model.fit_predict(blob_data)
        np.testing.assert_array_equal(labels, model.labels_)

    def test_hamerly_reproducibility(self, blob_data):
        m1 = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        m1.fit(blob_data)
        m2 = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        m2.fit(blob_data)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)
        assert abs(m1.inertia_ - m2.inertia_) < 1e-10

    def test_algorithm_in_repr(self):
        model = KMeans(n_clusters=2, algorithm="hamerly")
        assert 'algorithm="hamerly"' in repr(model)


# ---------------------------------------------------------------------------
# f32 support
# ---------------------------------------------------------------------------

class TestFloat32:
    def test_f32_fit_predict(self, blob_data):
        X = blob_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42)
        labels = model.fit_predict(X)
        assert len(labels) == len(X)

    def test_f32_cluster_centers_dtype(self, blob_data):
        X = blob_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        assert model.cluster_centers_.dtype == np.float32

    def test_f32_matches_f64_partition(self, blob_data):
        """f32 and f64 produce same cluster assignments on well-separated data."""
        m64 = KMeans(n_clusters=2, random_state=42, n_init=1)
        m64.fit(blob_data)

        m32 = KMeans(n_clusters=2, random_state=42, n_init=1)
        m32.fit(blob_data.astype(np.float32))

        np.testing.assert_array_equal(m64.labels_, m32.labels_)

    def test_f32_inertia_type(self, blob_data):
        X = blob_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        assert isinstance(model.inertia_, float)

    def test_f32_predict(self, blob_data):
        X = blob_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42)
        model.fit(X)
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, model.labels_)

    def test_f32_hamerly(self, blob_data):
        X = blob_data.astype(np.float32)
        model = KMeans(n_clusters=2, random_state=42, algorithm="hamerly")
        model.fit(X)
        assert len(model.labels_) == len(X)
        assert model.cluster_centers_.dtype == np.float32


# ---------------------------------------------------------------------------
# Cosine distance
# ---------------------------------------------------------------------------

class TestCosineKMeans:
    def test_cosine_fit(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, metric="cosine")
        model.fit(blob_data)
        assert len(model.labels_) == len(blob_data)
        assert model.inertia_ >= 0

    def test_cosine_predict(self, blob_data):
        model = KMeans(n_clusters=2, random_state=42, metric="cosine")
        model.fit(blob_data)
        preds = model.predict(blob_data)
        np.testing.assert_array_equal(preds, model.labels_)

    def test_cosine_separates_directions(self):
        """Cosine K-means should separate by direction, not magnitude."""
        # Points in two different directions, varying magnitudes
        rng = np.random.default_rng(42)
        dir1 = rng.normal(loc=[1, 0], scale=0.1, size=(30, 2))  # pointing right
        dir2 = rng.normal(loc=[0, 1], scale=0.1, size=(30, 2))  # pointing up
        # Scale some points — cosine shouldn't care about magnitude
        dir1[:10] *= 10
        dir2[:10] *= 10
        X = np.vstack([dir1, dir2])

        model = KMeans(n_clusters=2, random_state=42, metric="cosine", n_init=5)
        model.fit(X)
        # First 30 should share a cluster, last 30 another
        assert len(set(model.labels_[:30])) == 1
        assert len(set(model.labels_[30:])) == 1
        assert model.labels_[0] != model.labels_[30]

    def test_cosine_forces_lloyd(self):
        """Cosine metric should work even with algorithm='hamerly' (forces Lloyd internally)."""
        X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        model = KMeans(n_clusters=2, random_state=42, metric="cosine", algorithm="hamerly")
        model.fit(X)
        assert len(model.labels_) == 4

    def test_cosine_reproducibility(self, blob_data):
        m1 = KMeans(n_clusters=2, random_state=42, metric="cosine")
        m1.fit(blob_data)
        m2 = KMeans(n_clusters=2, random_state=42, metric="cosine")
        m2.fit(blob_data)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            KMeans(n_clusters=2, metric="manhattan")
