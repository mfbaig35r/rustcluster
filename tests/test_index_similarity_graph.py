"""Correctness tests for IndexFlat*.similarity_graph()."""

import os

import numpy as np
import pytest

from rustcluster.index import IndexFlatIP, IndexFlatL2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normalized_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 32)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(7)
    return rng.standard_normal((150, 24)).astype(np.float32)


def brute_force_pairs_ip(X, threshold, unique=False):
    n = X.shape[0]
    scores = X @ X.T
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if unique and i >= j:
                continue
            if scores[i, j] >= threshold:
                out.append((i, j, float(scores[i, j])))
    return out


def brute_force_pairs_l2(X, threshold, unique=False):
    n = X.shape[0]
    diffs = X[:, None, :] - X[None, :, :]
    sq = (diffs ** 2).sum(axis=2)
    out = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if unique and i >= j:
                continue
            if sq[i, j] <= threshold:
                out.append((i, j, float(sq[i, j])))
    return out


# ---------------------------------------------------------------------------
# IP semantics
# ---------------------------------------------------------------------------

class TestSimilarityGraphIP:
    def test_emits_both_directions_by_default(self, normalized_data):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        src, dst, _ = idx.similarity_graph(threshold=0.3)
        edges = set(zip(src.tolist(), dst.tolist()))
        for a, b in list(edges):
            assert (b, a) in edges

    def test_unique_pairs_halves_edges(self, normalized_data):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        s_full, _, _ = idx.similarity_graph(threshold=0.3)
        s_uniq, d_uniq, _ = idx.similarity_graph(threshold=0.3, unique_pairs=True)
        assert len(s_full) == 2 * len(s_uniq)
        for a, b in zip(s_uniq, d_uniq):
            assert a < b

    def test_threshold_inclusive(self, normalized_data):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        _, _, scores = idx.similarity_graph(threshold=0.5)
        assert (scores >= 0.5).all()

    def test_no_self_loops(self, normalized_data):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        # Threshold below all values to force every non-self pair.
        src, dst, _ = idx.similarity_graph(threshold=-1e9)
        assert (src != dst).all()

    def test_parity_with_brute_force(self, normalized_data):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        src, dst, scores = idx.similarity_graph(threshold=0.4, unique_pairs=True)
        bf = brute_force_pairs_ip(normalized_data, threshold=0.4, unique=True)

        got = sorted(zip(src.tolist(), dst.tolist()))
        bf_sorted = sorted((a, b) for a, b, _ in bf)
        assert got == bf_sorted
        assert len(scores) == len(bf)

    def test_external_ids_used(self, normalized_data):
        ids = np.arange(5000, 5100, dtype=np.uint64)
        idx = IndexFlatIP(dim=32)
        idx.add_with_ids(normalized_data, ids)
        src, dst, _ = idx.similarity_graph(threshold=0.4)
        assert src.min() >= 5000
        assert src.max() < 5100
        assert dst.min() >= 5000
        assert dst.max() < 5100


# ---------------------------------------------------------------------------
# L2 semantics
# ---------------------------------------------------------------------------

class TestSimilarityGraphL2:
    def test_threshold_inclusive(self, raw_data):
        idx = IndexFlatL2(dim=24)
        idx.add(raw_data)
        _, _, scores = idx.similarity_graph(threshold=10.0)
        assert (scores <= 10.0).all()

    def test_parity_with_brute_force(self, raw_data):
        idx = IndexFlatL2(dim=24)
        idx.add(raw_data)
        src, dst, scores = idx.similarity_graph(threshold=15.0, unique_pairs=True)
        bf = brute_force_pairs_l2(raw_data, threshold=15.0, unique=True)

        got = sorted(zip(src.tolist(), dst.tolist()))
        bf_sorted = sorted((a, b) for a, b, _ in bf)
        assert got == bf_sorted

        # Spot-check distances within float tolerance.
        score_lookup = dict(zip(zip(src.tolist(), dst.tolist()), scores.tolist()))
        for a, b, d in bf:
            assert abs(score_lookup[(a, b)] - d) < 1e-3

    def test_returns_empty_below_two_vectors(self):
        idx = IndexFlatL2(dim=4)
        idx.add(np.zeros((1, 4), dtype=np.float32))
        src, dst, scores = idx.similarity_graph(threshold=0.0)
        assert len(src) == 0 and len(dst) == 0 and len(scores) == 0


# ---------------------------------------------------------------------------
# Tile size override (env var) and dim variety
# ---------------------------------------------------------------------------

class TestTileVariety:
    @pytest.mark.parametrize("d", [4, 32, 128, 512, 1536])
    def test_correct_across_dims(self, d):
        rng = np.random.default_rng(d)
        n = 60
        X = rng.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        idx = IndexFlatIP(dim=d)
        idx.add(X)
        src, dst, scores = idx.similarity_graph(threshold=0.0, unique_pairs=True)
        bf = brute_force_pairs_ip(X, threshold=0.0, unique=True)

        got = sorted(zip(src.tolist(), dst.tolist()))
        bf_sorted = sorted((a, b) for a, b, _ in bf)
        assert got == bf_sorted

    def test_tile_size_env_override(self, normalized_data, monkeypatch):
        # Force a tiny tile so multiple tile boundaries are exercised.
        monkeypatch.setenv("RUSTCLUSTER_TILE_SIZE", "8")
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        src, dst, _ = idx.similarity_graph(threshold=0.3, unique_pairs=True)
        bf = brute_force_pairs_ip(normalized_data, threshold=0.3, unique=True)
        got = sorted(zip(src.tolist(), dst.tolist()))
        bf_sorted = sorted((a, b) for a, b, _ in bf)
        assert got == bf_sorted


# ---------------------------------------------------------------------------
# End-to-end notebook simulation
# ---------------------------------------------------------------------------

class TestNotebookEndToEnd:
    def test_replaces_batched_range_search_loop(self):
        """The Databricks notebook builds the same edge list via a batched
        range_search loop. similarity_graph should produce the same set."""
        rng = np.random.default_rng(123)
        n, d = 200, 32
        embeddings = rng.standard_normal((n, d)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        gc_item_numbers = np.arange(10_000, 10_000 + n, dtype=np.uint64)

        idx = IndexFlatIP(dim=d)
        idx.add_with_ids(embeddings, gc_item_numbers)
        threshold = 0.3

        # Notebook approach: batched range_search + manual filter/remap.
        notebook_edges = set()
        batch_size = 50
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            lims, _, labs = idx.range_search(
                embeddings[start:end], threshold, exclude_self=True
            )
            for j in range(end - start):
                src = int(gc_item_numbers[start + j])
                for k in range(lims[j], lims[j + 1]):
                    notebook_edges.add((src, int(labs[k])))

        # New approach: similarity_graph in one call.
        s, dst, _ = idx.similarity_graph(threshold=threshold)
        graph_edges = set(zip(s.tolist(), dst.tolist()))

        assert notebook_edges == graph_edges
