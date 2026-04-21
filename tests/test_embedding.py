"""Tests for rustcluster.experimental.EmbeddingCluster."""

import numpy as np
import pytest

from rustcluster.experimental import EmbeddingCluster


def make_spherical_data(n_per=100, d=64, k=3, seed=42):
    """Generate k clusters on the unit sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((k, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    data, labels = [], []
    for i in range(k):
        noise = rng.standard_normal((n_per, d)) * 0.1
        cluster = centers[i] + noise
        data.append(cluster)
        labels.extend([i] * n_per)
    return np.vstack(data).astype(np.float32), np.array(labels)


class TestFit:
    def test_fit_returns_self(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None)
        assert model.fit(X) is model

    def test_labels_shape(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert model.labels_.shape == (300,)

    def test_centers_shape(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert model.cluster_centers_.shape == (3, 64)

    def test_objective_positive(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert model.objective_ > 0

    def test_n_iter_populated(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert 1 <= model.n_iter_ <= 100

    def test_separates_clusters(self):
        X, true_labels = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        # Each true cluster should map to one predicted cluster
        for start in [0, 100, 200]:
            assert len(set(model.labels_[start:start+100])) == 1


class TestPCA:
    def test_with_reduction(self):
        X, _ = make_spherical_data(d=256)
        model = EmbeddingCluster(n_clusters=3, reduction_dim=32).fit(X)
        assert model.cluster_centers_.shape == (3, 32)

    def test_skip_reduction(self):
        X, _ = make_spherical_data(d=64)
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert model.cluster_centers_.shape == (3, 64)


class TestEvaluation:
    def test_representatives_count(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert len(model.representatives_) == 3

    def test_representatives_in_range(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert all(0 <= r < 300 for r in model.representatives_)

    def test_resultant_lengths_range(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        for r in model.resultant_lengths_:
            assert 0 <= r <= 1

    def test_intra_similarity_positive(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        for s in model.intra_similarity_:
            assert s > 0


class TestVMF:
    def test_refine_vmf(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        model.refine_vmf(X)
        assert model.probabilities_.shape == (300, 3)

    def test_probabilities_sum_to_one(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        model.refine_vmf(X)
        row_sums = model.probabilities_.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_concentrations_positive(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        model.refine_vmf(X)
        assert all(k > 0 for k in model.concentrations_)

    def test_bic_finite(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        model.refine_vmf(X)
        assert np.isfinite(model.bic_)


class TestDtypes:
    def test_f32(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert len(model.labels_) == 300

    def test_f64(self):
        X, _ = make_spherical_data()
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X.astype(np.float64))
        assert len(model.labels_) == 300

    def test_integer_input(self):
        X = np.random.randint(0, 100, (50, 10))
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert len(model.labels_) == 50


class TestNotFitted:
    def test_labels_before_fit(self):
        with pytest.raises(RuntimeError):
            EmbeddingCluster().labels_

    def test_vmf_before_fit(self):
        X = np.random.randn(10, 5).astype(np.float32)
        with pytest.raises(RuntimeError):
            EmbeddingCluster().refine_vmf(X)


class TestRepr:
    def test_repr(self):
        model = EmbeddingCluster(n_clusters=10, reduction_dim=64)
        r = repr(model)
        assert "EmbeddingCluster" in r
        assert "10" in r
