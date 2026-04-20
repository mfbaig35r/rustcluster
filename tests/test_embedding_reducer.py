"""Tests for rustcluster.experimental.EmbeddingReducer."""

import os
import tempfile

import numpy as np
import pytest

from rustcluster.experimental import EmbeddingReducer


def make_embeddings(n=200, d=64, seed=42):
    """Generate random embedding-like data."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float64)
    # L2-normalize like real embeddings
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


class TestFitTransformShape:
    def test_output_shape(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        result = reducer.fit_transform(X)
        assert result.shape == (100, 16)

    def test_large_target_dim(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=32, method="pca")
        result = reducer.fit_transform(X)
        assert result.shape == (100, 32)


class TestTransformMatchesFitTransform:
    def test_separate_equals_combined(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        combined = reducer.fit_transform(X)

        reducer2 = EmbeddingReducer(target_dim=16, method="pca")
        reducer2.fit(X)
        separate = reducer2.transform(X)

        np.testing.assert_allclose(combined, separate, atol=1e-10)


class TestUnitNorm:
    def test_rows_are_unit_norm(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        result = reducer.fit_transform(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_matryoshka_unit_norm(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="matryoshka")
        result = reducer.fit_transform(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestSaveLoadRoundtrip:
    def test_pca_roundtrip(self):
        X = make_embeddings(n=100, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        original = reducer.fit_transform(X)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            reducer.save(path)
            loaded = EmbeddingReducer.load(path)
            restored = loaded.transform(X)
            np.testing.assert_allclose(original, restored, atol=1e-10)
        finally:
            os.unlink(path)

    def test_matryoshka_roundtrip(self):
        X = make_embeddings(n=50, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="matryoshka")
        original = reducer.fit_transform(X)

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name

        try:
            reducer.save(path)
            loaded = EmbeddingReducer.load(path)
            restored = loaded.transform(X)
            np.testing.assert_allclose(original, restored, atol=1e-10)
        finally:
            os.unlink(path)


class TestMatryoshka:
    def test_truncates_correctly(self):
        """Matryoshka should take the first target_dim columns."""
        X = make_embeddings(n=50, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="matryoshka")
        result = reducer.fit_transform(X)
        assert result.shape == (50, 16)
        # The direction should be proportional to the first 16 columns
        # (after L2-normalization)
        expected = X[:, :16].copy()
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestTransformNewData:
    def test_new_data_shape(self):
        X_train = make_embeddings(n=100, d=64, seed=42)
        X_test = make_embeddings(n=30, d=64, seed=99)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        reducer.fit(X_train)
        result = reducer.transform(X_test)
        assert result.shape == (30, 16)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestF32Input:
    def test_f32_produces_output(self):
        X = make_embeddings(n=100, d=64).astype(np.float32)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        result = reducer.fit_transform(X)
        assert result.shape == (100, 16)
        assert result.dtype == np.float64

    def test_f32_unit_norm(self):
        X = make_embeddings(n=100, d=64).astype(np.float32)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        result = reducer.fit_transform(X)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestExistingAlgorithmsOnReduced:
    def test_kmeans_on_reduced(self):
        from rustcluster import KMeans
        X = make_embeddings(n=200, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        X_reduced = reducer.fit_transform(X)
        model = KMeans(n_clusters=3, metric="cosine").fit(X_reduced)
        assert model.labels_.shape == (200,)

    def test_dbscan_on_reduced(self):
        from rustcluster import DBSCAN
        X = make_embeddings(n=200, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        X_reduced = reducer.fit_transform(X)
        model = DBSCAN(eps=0.5, metric="cosine").fit(X_reduced)
        assert model.labels_.shape == (200,)


class TestReducedDataProperty:
    """Tests for EmbeddingCluster.reduced_data_ property."""

    def test_reduced_data_with_pca(self):
        from rustcluster.experimental import EmbeddingCluster
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 256)).astype(np.float32)
        model = EmbeddingCluster(n_clusters=3, reduction_dim=32).fit(X)
        rd = model.reduced_data_
        assert rd is not None
        assert rd.shape == (100, 32)
        # Should be L2-normalized
        norms = np.linalg.norm(rd, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_reduced_data_none_without_pca(self):
        from rustcluster.experimental import EmbeddingCluster
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 64)).astype(np.float32)
        model = EmbeddingCluster(n_clusters=3, reduction_dim=None).fit(X)
        assert model.reduced_data_ is None

    def test_reduced_data_is_copy(self):
        from rustcluster.experimental import EmbeddingCluster
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 256)).astype(np.float32)
        model = EmbeddingCluster(n_clusters=3, reduction_dim=32).fit(X)
        rd1 = model.reduced_data_
        rd2 = model.reduced_data_
        np.testing.assert_array_equal(rd1, rd2)
        # Modifying one shouldn't affect the other
        rd1[0, 0] = 999.0
        assert rd2[0, 0] != 999.0


class TestErrors:
    def test_transform_before_fit(self):
        X = make_embeddings(n=50, d=64)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        with pytest.raises(RuntimeError):
            reducer.transform(X)

    def test_save_before_fit(self):
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        with pytest.raises(RuntimeError):
            reducer.save("/tmp/should_not_exist.bin")

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            EmbeddingReducer(target_dim=16, method="invalid")

    def test_dimension_mismatch(self):
        X_train = make_embeddings(n=100, d=64)
        X_bad = make_embeddings(n=50, d=32)
        reducer = EmbeddingReducer(target_dim=16, method="pca")
        reducer.fit(X_train)
        with pytest.raises(ValueError):
            reducer.transform(X_bad)
