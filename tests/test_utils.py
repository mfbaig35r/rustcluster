"""Tests for rustcluster.utils.

extract_embeddings_from_spark is exercised against a fake DataFrame
that quacks like the parts of the pyspark API the function actually
touches (.count, .sample, .limit, .select, .toLocalIterator). pyspark
itself is stubbed into sys.modules so the function's `import pyspark`
guard succeeds without the real dependency.
"""

import sys
import types

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _stub_pyspark():
    """Make `import pyspark` succeed inside the helper without the dep."""
    real = sys.modules.get("pyspark")
    sys.modules["pyspark"] = types.ModuleType("pyspark")
    yield
    if real is None:
        sys.modules.pop("pyspark", None)
    else:
        sys.modules["pyspark"] = real


class _Row:
    """Stand-in for pyspark.sql.Row. Supports __getitem__ by column name."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def __getitem__(self, key):
        return self._data[key]


class _FakeDF:
    """Minimal DataFrame stub for testing the extraction helper."""

    def __init__(self, rows):
        self._rows = list(rows)

    def count(self):
        return len(self._rows)

    def sample(self, fraction, seed=0):
        # Deterministic: keep the first ceil(fraction * n) rows
        n = max(1, int(round(fraction * len(self._rows))))
        return _FakeDF(self._rows[:n])

    def limit(self, n):
        return _FakeDF(self._rows[:n])

    def select(self, *cols):
        keep = set(cols)
        filtered = [
            _Row(**{k: r[k] for k in r._data if k in keep}) for r in self._rows
        ]
        return _FakeDF(filtered)

    def toLocalIterator(self):
        return iter(self._rows)


def _make_rows(n=10, dim=4, with_metadata=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        emb = rng.standard_normal(dim).astype(np.float32).tolist()
        kwargs = {"embedding": emb}
        if with_metadata:
            kwargs["supplier_id"] = f"S{i:03d}"
            kwargs["commodity"] = f"C{i % 3}"
        rows.append(_Row(**kwargs))
    return rows


class TestExtractEmbeddingsFromSpark:
    def test_basic_shape_and_dtype(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=10, dim=4))
        emb, meta = extract_embeddings_from_spark(
            df,
            embedding_col="embedding",
            metadata_cols=["supplier_id", "commodity"],
            dtype=np.float32,
        )
        assert emb.shape == (10, 4)
        assert emb.dtype == np.float32
        assert meta is not None
        assert len(meta) == 10
        assert list(meta.columns) == ["supplier_id", "commodity"]

    def test_no_metadata(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=5, dim=8, with_metadata=False))
        emb, meta = extract_embeddings_from_spark(
            df, embedding_col="embedding", dtype=np.float32
        )
        assert emb.shape == (5, 8)
        assert meta is None

    def test_metadata_row_alignment(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=20, dim=4))
        emb, meta = extract_embeddings_from_spark(
            df,
            embedding_col="embedding",
            metadata_cols=["supplier_id"],
            dtype=np.float32,
        )
        # supplier_id matches the row index it was created with
        assert list(meta["supplier_id"]) == [f"S{i:03d}" for i in range(20)]
        # Embedding for that row matches what we generated
        rng = np.random.default_rng(0)
        for i in range(20):
            expected = rng.standard_normal(4).astype(np.float32)
            np.testing.assert_array_equal(emb[i], expected)

    def test_empty_dataframe(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF([])
        emb, meta = extract_embeddings_from_spark(
            df,
            embedding_col="embedding",
            metadata_cols=["supplier_id"],
            dtype=np.float32,
        )
        assert emb.shape == (0, 0)
        assert meta is not None
        assert len(meta) == 0

    def test_dtype_f64(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=5, dim=3, with_metadata=False))
        emb, _ = extract_embeddings_from_spark(
            df, embedding_col="embedding", dtype=np.float64
        )
        assert emb.dtype == np.float64

    def test_sample_n_caps_output(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=100, dim=4, with_metadata=False))
        emb, _ = extract_embeddings_from_spark(
            df,
            embedding_col="embedding",
            dtype=np.float32,
            sample_n=10,
        )
        assert emb.shape[0] <= 10

    def test_sample_n_larger_than_total_is_noop(self):
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=5, dim=4, with_metadata=False))
        emb, _ = extract_embeddings_from_spark(
            df,
            embedding_col="embedding",
            dtype=np.float32,
            sample_n=10_000,
        )
        assert emb.shape == (5, 4)

    def test_growth_path_beyond_initial_capacity(self):
        # initial_capacity = max(1024, dim). With dim=2 and n=2000 we
        # exercise the double-and-copy growth branch.
        from rustcluster.utils import extract_embeddings_from_spark
        df = _FakeDF(_make_rows(n=2000, dim=2, with_metadata=False))
        emb, _ = extract_embeddings_from_spark(
            df, embedding_col="embedding", dtype=np.float32
        )
        assert emb.shape == (2000, 2)


def test_raises_clear_error_without_pyspark(monkeypatch):
    # Override the autouse fixture so the import truly fails.
    monkeypatch.delitem(sys.modules, "pyspark", raising=False)
    builtins_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyspark":
            raise ImportError("No module named 'pyspark'")
        return builtins_import(name, *args, **kwargs)

    if isinstance(__builtins__, dict):
        monkeypatch.setitem(__builtins__, "__import__", fake_import)
    else:
        monkeypatch.setattr(__builtins__, "__import__", fake_import)

    from rustcluster.utils import extract_embeddings_from_spark
    with pytest.raises(ImportError, match="requires pyspark"):
        extract_embeddings_from_spark(_FakeDF([]), embedding_col="embedding")
