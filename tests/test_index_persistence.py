"""Save/load round-trip tests for IndexFlatL2 / IndexFlatIP."""

import json
import os

import numpy as np
import pytest

from rustcluster.index import IndexFlatIP, IndexFlatL2


@pytest.fixture
def normalized_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 32)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_ip_sequential_ids_search_identical(self, normalized_data, tmp_path):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        path = str(tmp_path / "idx.rci")
        idx.save(path)

        loaded = IndexFlatIP.load(path)
        assert loaded.dim == 32
        assert loaded.ntotal == 100
        assert loaded.metric == "ip"

        d_pre, l_pre = idx.search(normalized_data[:5], k=10)
        d_post, l_post = loaded.search(normalized_data[:5], k=10)
        np.testing.assert_array_equal(l_pre, l_post)
        np.testing.assert_allclose(d_pre, d_post)

    def test_l2_with_external_ids_round_trip(self, tmp_path):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 16)).astype(np.float32)
        ids = np.arange(9000, 9040, dtype=np.uint64)

        idx = IndexFlatL2(dim=16)
        idx.add_with_ids(X, ids)
        path = str(tmp_path / "l2.rci")
        idx.save(path)

        loaded = IndexFlatL2.load(path)
        assert loaded.metric == "l2"
        assert loaded.ntotal == 40

        _, labels = loaded.search(X[:5], k=1)
        np.testing.assert_array_equal(labels[:, 0], ids[:5])

    def test_similarity_graph_identical_post_load(self, normalized_data, tmp_path):
        ids = np.arange(5000, 5100, dtype=np.uint64)
        idx = IndexFlatIP(dim=32)
        idx.add_with_ids(normalized_data, ids)
        path = str(tmp_path / "sim.rci")
        idx.save(path)

        loaded = IndexFlatIP.load(path)
        s1, d1, sc1 = idx.similarity_graph(threshold=0.4, unique_pairs=True)
        s2, d2, sc2 = loaded.similarity_graph(threshold=0.4, unique_pairs=True)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(d1, d2)
        np.testing.assert_allclose(sc1, sc2)

    def test_files_layout(self, normalized_data, tmp_path):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        path = str(tmp_path / "idx.rci")
        idx.save(path)

        files = set(os.listdir(path))
        assert files == {"vectors.safetensors", "metadata.json"}  # no ids.safetensors

        meta = json.loads((tmp_path / "idx.rci" / "metadata.json").read_text())
        assert meta["index_type"] == "IndexFlatIP"
        assert meta["metric"] == "ip"
        assert meta["dim"] == 32
        assert meta["ntotal"] == 100
        assert meta["has_ids"] is False
        assert meta["format_version"] == 1

    def test_files_layout_with_ids(self, tmp_path):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 4)).astype(np.float32)
        idx = IndexFlatL2(dim=4)
        idx.add_with_ids(X, np.arange(10, dtype=np.uint64))
        path = str(tmp_path / "idx.rci")
        idx.save(path)

        files = set(os.listdir(path))
        assert files == {"vectors.safetensors", "ids.safetensors", "metadata.json"}

        meta = json.loads((tmp_path / "idx.rci" / "metadata.json").read_text())
        assert meta["has_ids"] is True


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_load_wrong_type_errors(self, normalized_data, tmp_path):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        path = str(tmp_path / "idx.rci")
        idx.save(path)
        with pytest.raises(ValueError, match="type mismatch"):
            IndexFlatL2.load(path)

    def test_load_missing_dir_errors(self, tmp_path):
        with pytest.raises((IOError, OSError, ValueError)):
            IndexFlatIP.load(str(tmp_path / "does_not_exist"))

    def test_tampered_version_errors(self, normalized_data, tmp_path):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        path = tmp_path / "idx.rci"
        idx.save(str(path))

        meta_path = path / "metadata.json"
        text = meta_path.read_text().replace(
            '"format_version": 1', '"format_version": 999'
        )
        meta_path.write_text(text)

        with pytest.raises(ValueError, match="unsupported"):
            IndexFlatIP.load(str(path))

    def test_corrupt_metadata_errors(self, normalized_data, tmp_path):
        idx = IndexFlatIP(dim=32)
        idx.add(normalized_data)
        path = tmp_path / "idx.rci"
        idx.save(str(path))

        (path / "metadata.json").write_text("{not valid json")
        with pytest.raises(ValueError):
            IndexFlatIP.load(str(path))


# ---------------------------------------------------------------------------
# End-to-end: save → load → use with full notebook workflow
# ---------------------------------------------------------------------------

class TestNotebookPersistence:
    def test_save_then_load_then_similarity_graph_matches_baseline(self, tmp_path):
        rng = np.random.default_rng(42)
        n, d = 200, 32
        embeddings = rng.standard_normal((n, d)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        gc_ids = np.arange(20_000, 20_000 + n, dtype=np.uint64)

        # Build, save, load.
        idx = IndexFlatIP(dim=d)
        idx.add_with_ids(embeddings, gc_ids)
        path = str(tmp_path / "items.rci")
        idx.save(path)
        del idx
        loaded = IndexFlatIP.load(path)

        # Generate the edge list via the loaded index and verify scale + IDs.
        s, dst, scores = loaded.similarity_graph(threshold=0.3)
        assert len(s) == len(dst) == len(scores)
        assert s.dtype == np.uint64 and dst.dtype == np.uint64
        assert (scores >= 0.3).all()
        assert s.min() >= 20_000 and s.max() < 20_000 + n
