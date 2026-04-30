"""Brute-force correctness tests for IndexFlatL2 / IndexFlatIP."""

import numpy as np
import pytest

from rustcluster.index import IndexFlatIP, IndexFlatL2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((50, 8)).astype(np.float32)


@pytest.fixture
def medium_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal((500, 32)).astype(np.float32)


@pytest.fixture
def normalized_data():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 64)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def brute_force_l2(queries, data, k):
    diffs = queries[:, None, :] - data[None, :, :]
    sq = (diffs ** 2).sum(axis=2)
    order = np.argsort(sq, axis=1)[:, :k]
    dists = np.take_along_axis(sq, order, axis=1)
    return dists, order


def brute_force_ip(queries, data, k):
    scores = queries @ data.T
    order = np.argsort(-scores, axis=1)[:, :k]
    top_scores = np.take_along_axis(scores, order, axis=1)
    return top_scores, order


# ---------------------------------------------------------------------------
# Construction & introspection
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_l2_initial_state(self):
        idx = IndexFlatL2(dim=8)
        assert idx.dim == 8
        assert idx.ntotal == 0
        assert idx.metric == "l2"

    def test_ip_initial_state(self):
        idx = IndexFlatIP(dim=8)
        assert idx.dim == 8
        assert idx.ntotal == 0
        assert idx.metric == "ip"

    def test_add_grows_ntotal(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data[:10])
        assert idx.ntotal == 10
        idx.add(small_data[10:30])
        assert idx.ntotal == 30


# ---------------------------------------------------------------------------
# Search correctness vs brute force
# ---------------------------------------------------------------------------

class TestSearchL2:
    def test_top1_is_self_for_indexed_query(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data)
        dists, labels = idx.search(small_data[:5], k=1)
        np.testing.assert_array_equal(labels[:, 0], np.arange(5))
        np.testing.assert_allclose(dists[:, 0], 0.0, atol=1e-5)

    def test_topk_matches_brute_force(self, medium_data):
        rng = np.random.default_rng(42)
        queries = rng.standard_normal((20, 32)).astype(np.float32)
        idx = IndexFlatL2(dim=32)
        idx.add(medium_data)

        dists, labels = idx.search(queries, k=10)
        bf_d, bf_l = brute_force_l2(queries, medium_data, k=10)

        np.testing.assert_array_equal(labels, bf_l)
        np.testing.assert_allclose(dists, bf_d, rtol=1e-4, atol=1e-4)

    def test_exclude_self(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data)
        _, labels = idx.search(small_data[:5], k=1, exclude_self=True)
        # No query should match its own row
        for q in range(5):
            assert labels[q, 0] != q


class TestSearchIP:
    def test_top1_for_normalized(self, normalized_data):
        idx = IndexFlatIP(dim=64)
        idx.add(normalized_data)
        scores, labels = idx.search(normalized_data[:5], k=1)
        np.testing.assert_array_equal(labels[:, 0], np.arange(5))
        np.testing.assert_allclose(scores[:, 0], 1.0, atol=1e-4)

    def test_topk_matches_brute_force(self, medium_data):
        rng = np.random.default_rng(99)
        queries = rng.standard_normal((20, 32)).astype(np.float32)
        idx = IndexFlatIP(dim=32)
        idx.add(medium_data)

        scores, labels = idx.search(queries, k=10)
        bf_s, bf_l = brute_force_ip(queries, medium_data, k=10)

        np.testing.assert_array_equal(labels, bf_l)
        np.testing.assert_allclose(scores, bf_s, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# External IDs
# ---------------------------------------------------------------------------

class TestAddWithIds:
    def test_external_ids_returned_as_labels(self, small_data):
        ids = np.arange(1000, 1050, dtype=np.uint64)
        idx = IndexFlatL2(dim=8)
        idx.add_with_ids(small_data, ids)

        _, labels = idx.search(small_data[:3], k=1)
        np.testing.assert_array_equal(labels[:, 0], ids[:3])

    def test_mixing_modes_errors(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data[:5])
        with pytest.raises(ValueError):
            idx.add_with_ids(small_data[:5], np.arange(5, dtype=np.uint64))

    def test_duplicate_id_errors(self, small_data):
        idx = IndexFlatIP(dim=8)
        ids = np.array([1, 2, 3, 2, 5], dtype=np.uint64)
        with pytest.raises(ValueError):
            idx.add_with_ids(small_data[:5], ids)

    def test_id_count_mismatch_errors(self, small_data):
        idx = IndexFlatIP(dim=8)
        with pytest.raises(ValueError):
            idx.add_with_ids(small_data[:5], np.array([1, 2], dtype=np.uint64))


# ---------------------------------------------------------------------------
# Range search
# ---------------------------------------------------------------------------

class TestRangeSearch:
    def test_l2_threshold_inclusive(self):
        data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0]], dtype=np.float32)
        idx = IndexFlatL2(dim=2)
        idx.add(data)

        lims, dists, labs = idx.range_search(np.array([[0.0, 0.0]], dtype=np.float32), threshold=1.5)
        assert lims.tolist() == [0, 3]
        # Three matches: indices 0,1,2 with squared distances 0, 1, 1
        assert sorted(labs.tolist()) == [0, 1, 2]
        for d in dists:
            assert d <= 1.5

    def test_ip_threshold_inclusive(self):
        data = np.array([[1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0]], dtype=np.float32)
        idx = IndexFlatIP(dim=2)
        idx.add(data)

        lims, dists, labs = idx.range_search(np.array([[1.0, 0.0]], dtype=np.float32), threshold=0.5)
        assert lims.tolist() == [0, 2]
        for d in dists:
            assert d >= 0.5

    def test_lims_csr_shape(self, normalized_data):
        idx = IndexFlatIP(dim=64)
        idx.add(normalized_data)

        lims, dists, labs = idx.range_search(normalized_data[:10], threshold=0.0)
        assert len(lims) == 11
        assert lims[0] == 0
        assert lims[-1] == len(dists)
        assert len(dists) == len(labs)

    def test_range_search_exclude_self(self, normalized_data):
        idx = IndexFlatIP(dim=64)
        idx.add(normalized_data)

        # With exclude_self, no query should appear in its own neighbor list.
        lims, _, labs = idx.range_search(normalized_data[:5], threshold=0.5, exclude_self=True)
        for q in range(5):
            neighbors = labs[lims[q]:lims[q + 1]]
            assert q not in neighbors.tolist()

    def test_range_search_matches_search_at_high_threshold(self, normalized_data):
        """Range with very loose threshold should include all top-k results."""
        idx = IndexFlatIP(dim=64)
        idx.add(normalized_data)

        scores, labels = idx.search(normalized_data[:5], k=20)
        lims, range_dists, range_labs = idx.range_search(normalized_data[:5], threshold=-1.0)

        for q in range(5):
            top20 = set(labels[q].tolist())
            range_set = set(range_labs[lims[q]:lims[q + 1]].tolist())
            # Every top-20 must be in the range result (which includes everything).
            assert top20.issubset(range_set)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_search_empty_index(self):
        idx = IndexFlatL2(dim=4)
        with pytest.raises(RuntimeError):
            idx.search(np.zeros((1, 4), dtype=np.float32), k=1)

    def test_dim_mismatch_on_add(self, small_data):
        idx = IndexFlatL2(dim=8)
        with pytest.raises(ValueError):
            idx.add(small_data[:, :4])  # wrong dim

    def test_dim_mismatch_on_search(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data)
        with pytest.raises(ValueError):
            idx.search(np.zeros((1, 4), dtype=np.float32), k=1)

    def test_non_float32_input_rejected(self):
        idx = IndexFlatL2(dim=4)
        with pytest.raises(ValueError):
            idx.add(np.zeros((3, 4), dtype=np.float64))

    def test_non_finite_rejected(self):
        idx = IndexFlatL2(dim=2)
        bad = np.array([[1.0, np.nan]], dtype=np.float32)
        with pytest.raises(ValueError):
            idx.add(bad)

    def test_k_zero_rejected(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data)
        with pytest.raises(ValueError):
            idx.search(small_data[:1], k=0)

    def test_k_larger_than_ntotal_pads(self, small_data):
        idx = IndexFlatL2(dim=8)
        idx.add(small_data[:5])
        dists, labels = idx.search(small_data[:1], k=10)
        # First 5 are real, remaining 5 are sentinel/padding
        assert (labels[0, 5:] == -1).all()
        assert np.isinf(dists[0, 5:]).all()


# ---------------------------------------------------------------------------
# Production workload end-to-end (reproduces the FAISS notebook pattern)
# ---------------------------------------------------------------------------

class TestSimilarityNotebookPattern:
    def test_normalize_then_ip_range_search_matches_faiss_shape(self):
        """Mimic the Databricks notebook: build IP index, range_search per batch."""
        rng = np.random.default_rng(7)
        n, d = 200, 32
        embeddings = rng.standard_normal((n, d)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        gc_item_numbers = np.arange(10_000, 10_000 + n, dtype=np.uint64)

        idx = IndexFlatIP(dim=d)
        idx.add_with_ids(embeddings, gc_item_numbers)

        threshold = 0.3
        batch_size = 50
        all_edges = []
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = embeddings[start:end]
            lims, dists, labs = idx.range_search(batch, threshold, exclude_self=True)
            for j in range(end - start):
                src = gc_item_numbers[start + j]
                for k in range(lims[j], lims[j + 1]):
                    all_edges.append((int(src), int(labs[k]), float(dists[k])))

        # Every edge satisfies the threshold and uses external IDs from the input set.
        valid_ids = set(gc_item_numbers.tolist())
        for src, dst, score in all_edges:
            assert score >= threshold
            assert src in valid_ids
            assert dst in valid_ids
            assert src != dst

        # Symmetry: cosine similarity is symmetric, so for every (a,b,s) there
        # should be a corresponding (b,a,~s) within float tolerance.
        edges_by_pair = {(s, d): score for s, d, score in all_edges}
        for (s, d), score in list(edges_by_pair.items()):
            assert (d, s) in edges_by_pair
            assert abs(edges_by_pair[(d, s)] - score) < 1e-4
