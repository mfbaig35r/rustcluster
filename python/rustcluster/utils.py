"""Utility helpers for working with rustcluster on common data sources.

Currently exposes Arrow-based embedding extraction from Spark DataFrames,
which avoids the multi-GB Python list-of-lists overhead that `toPandas()`
introduces when each row contains a 1536-dim embedding.

Pyspark is an optional runtime dependency; it is lazy-imported only when
the relevant utility is actually called.
"""

from __future__ import annotations

import numpy as np

__all__ = ["extract_embeddings_from_spark"]


def extract_embeddings_from_spark(
    df,
    embedding_col,
    metadata_cols=None,
    dtype=np.float32,
    sample_n=None,
    seed=0,
):
    """Stream a Spark DataFrame's embedding column into a NumPy array.

    The naive `df.toPandas()` path materializes each row's embedding as a
    Python list of Python floats before the numpy conversion runs. For
    312K x 1536d this is ~3 GB of pure Python overhead on top of the
    f32 array itself. This helper uses `toLocalIterator()` so the JVM
    can release each Arrow batch as Python consumes it, and writes
    rows directly into a pre-allocated numpy array without ever
    holding a list-of-lists in Python memory.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Source Spark DataFrame. Must contain `embedding_col` as an
        ArrayType(FloatType()) or similar.
    embedding_col : str
        Column name holding the embedding vector.
    metadata_cols : list[str] or None, default=None
        Other columns to return as a pandas DataFrame alongside the
        embedding array. Pass None or `[]` to skip the metadata pull.
    dtype : numpy dtype, default=np.float32
        Output array dtype. f32 is the right choice for embedding
        clustering; f64 is supported for compatibility.
    sample_n : int or None, default=None
        If set, randomly sample `sample_n` rows before extraction.
        Useful for fitting PCA on a subset of a large dataset.
    seed : int, default=0
        Sampling seed (only used when `sample_n` is set).

    Returns
    -------
    embeddings : ndarray of shape (n, embedding_dim), dtype=`dtype`
        Stacked embeddings in the order yielded by `toLocalIterator()`.
    metadata : pandas.DataFrame or None
        DataFrame of `metadata_cols` aligned row-for-row with `embeddings`,
        or None if `metadata_cols` is empty/None.

    Notes
    -----
    The first row is consumed to learn the embedding dimensionality.
    If `df` is empty an empty array is returned.

    Examples
    --------
    >>> embeddings, meta = extract_embeddings_from_spark(  # doctest: +SKIP
    ...     df,
    ...     embedding_col="embedding",
    ...     metadata_cols=["supplier_id", "commodity"],
    ...     dtype=np.float32,
    ... )
    """
    try:
        import pyspark  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "extract_embeddings_from_spark requires pyspark; install it or "
            "use a non-Spark loading path."
        ) from exc

    if metadata_cols is None:
        metadata_cols = []

    work_df = df
    if sample_n is not None:
        n_total = work_df.count()
        if n_total > sample_n:
            fraction = min(1.0, (sample_n * 1.2) / n_total)
            work_df = work_df.sample(fraction=fraction, seed=seed).limit(sample_n)
        # If sample_n >= n_total we just take everything; no sampling needed.

    select_cols = [embedding_col] + list(metadata_cols)
    work_df = work_df.select(*select_cols)

    iterator = work_df.toLocalIterator()
    first = next(iterator, None)
    if first is None:
        empty_emb = np.empty((0, 0), dtype=dtype)
        if metadata_cols:
            import pandas as pd
            return empty_emb, pd.DataFrame(columns=metadata_cols)
        return empty_emb, None

    embedding_dim = len(first[embedding_col])
    initial_capacity = max(1024, embedding_dim)
    embeddings = np.empty((initial_capacity, embedding_dim), dtype=dtype)
    metadata_rows = [] if metadata_cols else None

    def write_row(idx, row):
        nonlocal embeddings
        if idx >= embeddings.shape[0]:
            new_cap = embeddings.shape[0] * 2
            grown = np.empty((new_cap, embedding_dim), dtype=dtype)
            grown[: embeddings.shape[0]] = embeddings
            embeddings = grown
        embeddings[idx, :] = row[embedding_col]
        if metadata_rows is not None:
            metadata_rows.append({col: row[col] for col in metadata_cols})

    write_row(0, first)
    n = 1
    for row in iterator:
        write_row(n, row)
        n += 1

    embeddings = embeddings[:n]

    metadata = None
    if metadata_cols:
        import pandas as pd
        metadata = pd.DataFrame(metadata_rows, columns=list(metadata_cols))

    return embeddings, metadata
