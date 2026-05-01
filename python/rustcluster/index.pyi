"""Type stubs for ``rustcluster.index``.

Hand-written because PyO3 doesn't auto-generate stubs. Keep in sync with
the PyO3 method signatures in ``src/lib.rs``.
"""

from __future__ import annotations

from os import PathLike
from typing import Tuple, Union

import numpy as np

_PathLike = Union[str, PathLike]
_F32Array2D = np.ndarray
_U64Array1D = np.ndarray
_I64Array1D = np.ndarray
_I64Array2D = np.ndarray
_F32Array1D = np.ndarray
_F32Array2DResult = np.ndarray


class IndexFlatL2:
    """Flat exact L2 (squared Euclidean) index over f32 vectors.

    Distances returned by ``search`` and ``range_search`` are squared L2
    (matches FAISS).
    """

    @property
    def dim(self) -> int: ...
    @property
    def ntotal(self) -> int: ...
    @property
    def metric(self) -> str: ...

    def __init__(self, dim: int) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
    def __getnewargs__(self) -> tuple[int]: ...

    def add(self, vectors: _F32Array2D) -> "IndexFlatL2": ...
    def add_with_ids(
        self, vectors: _F32Array2D, ids: _U64Array1D
    ) -> "IndexFlatL2": ...

    def search(
        self,
        queries: _F32Array2D,
        k: int,
        exclude_self: bool = False,
    ) -> Tuple[_F32Array2DResult, _I64Array2D]: ...

    def range_search(
        self,
        queries: _F32Array2D,
        threshold: float,
        exclude_self: bool = False,
    ) -> Tuple[_I64Array1D, _F32Array1D, _I64Array1D]: ...

    def similarity_graph(
        self,
        threshold: float,
        unique_pairs: bool = False,
    ) -> Tuple[_U64Array1D, _U64Array1D, _F32Array1D]: ...

    def save(self, path: _PathLike) -> None: ...
    @staticmethod
    def load(path: _PathLike) -> "IndexFlatL2": ...


class IndexFlatIP:
    """Flat exact inner-product index over f32 vectors.

    For cosine similarity, L2-normalize vectors before ``add`` and queries
    before ``search``.
    """

    @property
    def dim(self) -> int: ...
    @property
    def ntotal(self) -> int: ...
    @property
    def metric(self) -> str: ...

    def __init__(self, dim: int) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __getstate__(self) -> bytes: ...
    def __setstate__(self, state: bytes) -> None: ...
    def __getnewargs__(self) -> tuple[int]: ...

    def add(self, vectors: _F32Array2D) -> "IndexFlatIP": ...
    def add_with_ids(
        self, vectors: _F32Array2D, ids: _U64Array1D
    ) -> "IndexFlatIP": ...

    def search(
        self,
        queries: _F32Array2D,
        k: int,
        exclude_self: bool = False,
    ) -> Tuple[_F32Array2DResult, _I64Array2D]: ...

    def range_search(
        self,
        queries: _F32Array2D,
        threshold: float,
        exclude_self: bool = False,
    ) -> Tuple[_I64Array1D, _F32Array1D, _I64Array1D]: ...

    def similarity_graph(
        self,
        threshold: float,
        unique_pairs: bool = False,
    ) -> Tuple[_U64Array1D, _U64Array1D, _F32Array1D]: ...

    def save(self, path: _PathLike) -> None: ...
    @staticmethod
    def load(path: _PathLike) -> "IndexFlatIP": ...
