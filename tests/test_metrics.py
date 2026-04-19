"""Integration tests for rustcluster clustering metrics."""

import numpy as np
import pytest

from rustcluster import (
    KMeans,
    DBSCAN,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


@pytest.fixture
def well_separated():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(30, 2))
    X = np.vstack([c1, c2])
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    return X, labels


@pytest.fixture
def poorly_separated():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 2))
    labels = np.array([0] * 30 + [1] * 30, dtype=np.int64)
    return X, labels


class TestSilhouette:
    def test_well_separated(self, well_separated):
        X, labels = well_separated
        score = silhouette_score(X, labels)
        assert score > 0.9

    def test_poorly_separated(self, poorly_separated):
        X, labels = poorly_separated
        score = silhouette_score(X, labels)
        assert score < 0.5

    def test_range(self, well_separated):
        X, labels = well_separated
        score = silhouette_score(X, labels)
        assert -1.0 <= score <= 1.0

    def test_with_kmeans(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(10, 0.3, (30, 2))])
        model = KMeans(n_clusters=2, random_state=42).fit(X)
        score = silhouette_score(X, model.labels_)
        assert score > 0.9

    def test_with_dbscan(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(10, 0.3, (30, 2))])
        model = DBSCAN(eps=1.0, min_samples=3).fit(X)
        score = silhouette_score(X, model.labels_)
        assert score > 0.9

    def test_f32(self, well_separated):
        X, labels = well_separated
        score = silhouette_score(X.astype(np.float32), labels)
        assert score > 0.9


class TestCalinskiHarabasz:
    def test_well_separated(self, well_separated):
        X, labels = well_separated
        score = calinski_harabasz_score(X, labels)
        assert score > 1000

    def test_poorly_separated_lower(self, well_separated, poorly_separated):
        X_good, labels_good = well_separated
        X_bad, labels_bad = poorly_separated
        good = calinski_harabasz_score(X_good, labels_good)
        bad = calinski_harabasz_score(X_bad, labels_bad)
        assert good > bad


class TestDaviesBouldin:
    def test_well_separated(self, well_separated):
        X, labels = well_separated
        score = davies_bouldin_score(X, labels)
        assert score < 0.1

    def test_poorly_separated_higher(self, well_separated, poorly_separated):
        X_good, labels_good = well_separated
        X_bad, labels_bad = poorly_separated
        good = davies_bouldin_score(X_good, labels_good)
        bad = davies_bouldin_score(X_bad, labels_bad)
        assert bad > good

    def test_non_negative(self, well_separated):
        X, labels = well_separated
        score = davies_bouldin_score(X, labels)
        assert score >= 0
