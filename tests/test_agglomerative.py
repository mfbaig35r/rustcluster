"""Integration tests for rustcluster AgglomerativeClustering."""

import numpy as np
import pytest

from rustcluster import AgglomerativeClustering


@pytest.fixture
def two_blobs():
    rng = np.random.default_rng(0)
    c1 = rng.normal(loc=[0, 0], scale=0.3, size=(20, 2))
    c2 = rng.normal(loc=[10, 10], scale=0.3, size=(20, 2))
    return np.vstack([c1, c2])


class TestFit:
    def test_fit_returns_self(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        assert model.fit(two_blobs) is model

    def test_labels_length(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_fit_predict(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        labels = model.fit_predict(two_blobs)
        assert len(labels) == len(two_blobs)

    def test_two_clusters(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(two_blobs)
        assert len(set(model.labels_[:20])) == 1
        assert len(set(model.labels_[20:])) == 1
        assert model.labels_[0] != model.labels_[20]


class TestAttributes:
    def test_children_shape(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(two_blobs)
        # n=40, target=2 → 38 merges
        assert model.children_.shape == (38, 2)

    def test_distances_length(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(two_blobs)
        assert len(model.distances_) == 38

    def test_n_clusters_(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(two_blobs)
        assert model.n_clusters_ == 3


class TestLinkage:
    def test_ward(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="ward")
        model.fit(two_blobs)
        assert model.labels_[0] != model.labels_[20]

    def test_complete(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="complete")
        model.fit(two_blobs)
        assert model.labels_[0] != model.labels_[20]

    def test_average(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="average")
        model.fit(two_blobs)
        assert model.labels_[0] != model.labels_[20]

    def test_single(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="single")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)


class TestMetric:
    def test_manhattan(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="complete", metric="manhattan")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_cosine(self, two_blobs):
        model = AgglomerativeClustering(n_clusters=2, linkage="complete", metric="cosine")
        model.fit(two_blobs)
        assert len(model.labels_) == len(two_blobs)

    def test_ward_requires_euclidean(self):
        with pytest.raises(ValueError, match="Ward"):
            AgglomerativeClustering(linkage="ward", metric="cosine")


class TestValidation:
    def test_invalid_linkage(self):
        with pytest.raises(ValueError, match="linkage"):
            AgglomerativeClustering(linkage="bogus")

    def test_invalid_metric(self):
        with pytest.raises(ValueError, match="metric"):
            AgglomerativeClustering(metric="hamming")

    def test_empty_input(self):
        model = AgglomerativeClustering()
        with pytest.raises(ValueError):
            model.fit(np.empty((0, 2)))

    def test_1d_input(self):
        model = AgglomerativeClustering()
        with pytest.raises(ValueError):
            model.fit(np.array([1.0, 2.0]))


class TestNotFitted:
    def test_labels_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            AgglomerativeClustering().labels_

    def test_children_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            AgglomerativeClustering().children_

    def test_distances_before_fit(self):
        with pytest.raises(RuntimeError, match="not been fitted"):
            AgglomerativeClustering().distances_


class TestDtypes:
    def test_float32(self, two_blobs):
        X = two_blobs.astype(np.float32)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(X)
        assert len(model.labels_) == len(X)

    def test_integer_input(self):
        X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(X)
        assert len(model.labels_) == 4


class TestEdgeCases:
    def test_single_cluster(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        model = AgglomerativeClustering(n_clusters=1)
        model.fit(X)
        assert all(l == 0 for l in model.labels_)

    def test_n_equals_n_clusters(self):
        X = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])
        model = AgglomerativeClustering(n_clusters=3)
        model.fit(X)
        assert len(set(model.labels_)) == 3

    def test_deterministic(self, two_blobs):
        m1 = AgglomerativeClustering(n_clusters=2).fit(two_blobs)
        m2 = AgglomerativeClustering(n_clusters=2).fit(two_blobs)
        np.testing.assert_array_equal(m1.labels_, m2.labels_)


class TestRepr:
    def test_repr(self):
        model = AgglomerativeClustering(n_clusters=5, linkage="complete")
        r = repr(model)
        assert "AgglomerativeClustering" in r
        assert "5" in r
        assert "complete" in r
