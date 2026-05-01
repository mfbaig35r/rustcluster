# Plan: Python-Library Polish for rustcluster.index

## Goal

Close the polish gaps identified in the v1.1 review so `rustcluster.index`
meets the bar for a published Python library: sane interactive UX,
pickle round-trips, IDE/typechecker support, and consistency with the
rest of `rustcluster`.

The functional surface (Flat indexes, search, range_search,
similarity_graph, save/load) is unchanged. This is purely about how
the existing surface presents to a Python user.

## Scope

**In scope:**
1. `__repr__` for both indexes
2. Pickle support (`__getstate__` / `__setstate__` / `__getnewargs__`)
3. Method docstrings discoverable via `help()`
4. Hand-written `.pyi` type stubs
5. `__len__` returning ntotal
6. `pathlib.Path` accepted in `save` / `load`
7. `add` and `add_with_ids` return self (for fluent chaining)

**Out of scope:**
- Adding a Python wrapper class layer (Option B from the review).
  Index has no f32/f64 dispatch and no Python-side state, so a
  wrapper adds indirection without value. Everything below is
  done directly on the PyO3 class.
- Iteration / `__getitem__` — not idiomatic for vector indexes
  and FAISS doesn't expose it either.
- `__eq__` / `__hash__` — identity comparison is fine.
- New top-level helpers like `rustcluster.search(X, q, k)` —
  we can add later if usage actually warrants it.

## Architecture decision: stay with direct PyO3 exposure

The existing `IndexFlatL2` and `IndexFlatIP` are exposed directly from
the `_rustcluster` extension module — no Python wrapper class
delegating to a `_RustIndex...` like `KMeans` does. The KMeans wrapper
exists for two reasons:
- **f32/f64 dispatch** at the Python boundary (we don't need this —
  index is f32-only)
- **Pickle protocol routing** (we'll add `__getstate__`/`__setstate__`
  directly on the Rust class instead)

A wrapper would add a layer of forwarding methods with no logic in
them, which is worse for `help()`, worse for tracebacks, and creates
an extra place where docstrings or signatures could drift. Stay
direct.

## Per-gap design

### 1. `__repr__`

Format follows sklearn's reproducible-representation convention:

```rust
fn __repr__(&self) -> String {
    let n = self.inner.ntotal();
    let has_ids = matches!(self.inner.ids(), IdMap::Explicit { .. });
    format!(
        "IndexFlatL2(dim={}, ntotal={}{})",
        self.inner.dim(),
        n,
        if has_ids { ", has_ids=True" } else { "" }
    )
}
```

Renders as:
```
>>> IndexFlatIP(dim=128)
IndexFlatIP(dim=128, ntotal=0)
>>> idx.add(X)
>>> idx
IndexFlatIP(dim=128, ntotal=1000)
>>> idx.add_with_ids(X, ids)
IndexFlatIP(dim=128, ntotal=2000, has_ids=True)
```

### 2. Pickle support

Pickle requires three things on a class whose `__new__` takes args:
- `__getnewargs__` returning `(dim,)` so `__new__(dim)` reconstructs
  an empty instance
- `__getstate__` returning serialized state as bytes
- `__setstate__` accepting those bytes and overwriting `self.inner`

Reuse the existing `save_flat_l2` / `load_flat_l2` infrastructure but
target an in-memory bytes buffer instead of a directory. Add a
sibling pair of functions to `src/index/persistence.rs`:

```rust
pub fn serialize_flat_l2_to_bytes(idx: &IndexFlatL2) -> Result<Vec<u8>, ClusterError>;
pub fn deserialize_flat_l2_from_bytes(bytes: &[u8]) -> Result<IndexFlatL2, ClusterError>;
```

Implementation: pack metadata + tensors into a single zstd-compressed
or zip-archive blob using safetensors' in-memory APIs. Probably
zip-of-{metadata.json, vectors.safetensors, optional ids.safetensors}
since that's the cleanest way to keep the same on-disk format
in-memory.

PyO3 methods:

```rust
fn __getnewargs__(&self) -> (usize,) {
    (self.inner.dim(),)
}

fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = py
        .allow_threads(|| serialize_flat_l2_to_bytes(&self.inner))
        .map_err(pyo3::PyErr::from)?;
    Ok(PyBytes::new(py, &bytes))
}

fn __setstate__(&mut self, py: Python<'_>, state: &Bound<'_, PyBytes>) -> PyResult<()> {
    let bytes = state.as_bytes();
    self.inner = py
        .allow_threads(|| deserialize_flat_l2_from_bytes(bytes))
        .map_err(pyo3::PyErr::from)?;
    Ok(())
}
```

This automatically supports `copy.copy(idx)`, `copy.deepcopy(idx)`,
`pickle.dumps`, and `multiprocessing` workers.

### 3. Method docstrings

PyO3 picks up `///` doc comments on `#[pymethods]` functions
and exposes them as `__doc__`. Example:

```rust
/// Append vectors to the index using sequential IDs `ntotal..ntotal+n`.
///
/// Errors if the index has previously been populated with explicit IDs
/// via `add_with_ids`.
///
/// Parameters
/// ----------
/// vectors : np.ndarray of shape (n, dim), dtype float32
///     Vectors to append. Must be C-contiguous.
///
/// Returns
/// -------
/// self
///
/// Raises
/// ------
/// ValueError
///     If `vectors` is not 2-D float32 contiguous, dim mismatches, or
///     the array contains NaN/inf.
fn add(...) -> ...
```

Apply NumPy/sklearn docstring style across all methods: one-line
summary, blank line, longer description, `Parameters`, `Returns`,
`Raises`. ~30 minutes mechanical work for the ~10 methods.

### 4. Type stubs (`python/rustcluster/index.pyi`)

Hand-written, since PyO3 doesn't generate stubs. Format:

```python
from __future__ import annotations
from os import PathLike
from typing import Tuple
import numpy as np

_PathLike = str | PathLike[str]

class IndexFlatL2:
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
    def add(self, vectors: np.ndarray) -> "IndexFlatL2": ...
    def add_with_ids(
        self, vectors: np.ndarray, ids: np.ndarray
    ) -> "IndexFlatL2": ...
    def search(
        self,
        queries: np.ndarray,
        k: int,
        exclude_self: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]: ...
    def range_search(
        self,
        queries: np.ndarray,
        threshold: float,
        exclude_self: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def similarity_graph(
        self,
        threshold: float,
        unique_pairs: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    def save(self, path: _PathLike) -> None: ...
    @staticmethod
    def load(path: _PathLike) -> "IndexFlatL2": ...

class IndexFlatIP:
    # ...identical surface, copy-paste...
```

Stub lives at `python/rustcluster/index.pyi`. Also need a
`py.typed` marker file at `python/rustcluster/py.typed` (empty)
so type checkers know the package ships stubs (PEP 561).

### 5. `__len__`

```rust
fn __len__(&self) -> usize {
    self.inner.ntotal()
}
```

Now `len(idx)` works, and `bool(idx)` returns False for empty
indexes (which is the right Python idiom).

### 6. `pathlib.Path` in `save` / `load`

Accept anything with `__fspath__` by extracting `PathBuf`:

```rust
fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
    let path_str = path.to_str().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("path must be valid UTF-8")
    })?;
    py.allow_threads(|| save_flat_l2(&self.inner, path_str))
        .map_err(Into::into)
}
```

PyO3 extracts `PathBuf` from `str`, `bytes`, `pathlib.Path`, and
anything implementing `os.PathLike`. Internal `save_flat_l2` still
takes `&str` — we don't need to change the Rust-side signature.

Same change for `load`.

### 7. `add` / `add_with_ids` return self

Use PyO3's `slf` parameter pattern to return a borrowed reference to
self, which Python sees as the same object:

```rust
fn add<'py>(
    mut slf: PyRefMut<'py, Self>,
    py: Python<'py>,
    x: &Bound<'_, PyAny>,
) -> PyResult<PyRefMut<'py, Self>> {
    let arr = extract_f32_2d(x, "vectors")?;
    let view = arr.as_array();
    py.allow_threads(|| slf.inner.add(view))
        .map_err(pyo3::PyErr::from)?;
    Ok(slf)
}
```

This is the FAISS-incompatible change but matches sklearn `fit`
returning self. Acceptable because:
- Existing FAISS user code that does `idx.add(X)` (ignoring the
  return value) still works.
- Chaining `IndexFlatIP(dim=128).add(X)` becomes possible.
- Matches `rustcluster.KMeans.fit(X)` which already returns self.

If PyO3 ergonomics prove painful (lifetime gymnastics), this is the
gap to defer. Everything else is small and clean.

## Test plan

Add tests to `tests/test_index_flat.py`:

```python
class TestRepr:
    def test_repr_shows_dim_and_ntotal(self):
        idx = IndexFlatIP(dim=64)
        assert repr(idx) == "IndexFlatIP(dim=64, ntotal=0)"
        idx.add(np.zeros((5, 64), dtype=np.float32))
        assert "ntotal=5" in repr(idx)

    def test_repr_shows_has_ids_when_explicit(self):
        idx = IndexFlatL2(dim=4)
        idx.add_with_ids(np.zeros((3, 4), dtype=np.float32), np.arange(3, dtype=np.uint64))
        assert "has_ids=True" in repr(idx)


class TestLen:
    def test_len_matches_ntotal(self):
        idx = IndexFlatIP(dim=4)
        assert len(idx) == 0
        idx.add(np.zeros((10, 4), dtype=np.float32))
        assert len(idx) == 10

    def test_bool_empty_is_false(self):
        assert not IndexFlatIP(dim=4)
        idx = IndexFlatIP(dim=4)
        idx.add(np.zeros((1, 4), dtype=np.float32))
        assert idx


class TestPickle:
    def test_pickle_roundtrip_preserves_search_results(self, normalized_data):
        import pickle
        idx = IndexFlatIP(dim=normalized_data.shape[1])
        idx.add(normalized_data)
        pre = idx.search(normalized_data[:5], k=10)

        roundtripped = pickle.loads(pickle.dumps(idx))
        post = roundtripped.search(normalized_data[:5], k=10)

        np.testing.assert_array_equal(pre[0], post[0])
        np.testing.assert_array_equal(pre[1], post[1])

    def test_pickle_preserves_external_ids(self):
        ids = np.arange(1000, 1010, dtype=np.uint64)
        idx = IndexFlatL2(dim=4)
        idx.add_with_ids(np.zeros((10, 4), dtype=np.float32), ids)
        roundtripped = pickle.loads(pickle.dumps(idx))
        _, labels = roundtripped.search(np.zeros((1, 4), dtype=np.float32), k=1)
        assert labels[0, 0] == 1000

    def test_deepcopy(self):
        import copy
        idx = IndexFlatIP(dim=4)
        idx.add(np.zeros((5, 4), dtype=np.float32))
        clone = copy.deepcopy(idx)
        assert clone is not idx
        assert len(clone) == len(idx)


class TestPathlibSupport:
    def test_save_load_with_path(self, tmp_path):
        idx = IndexFlatIP(dim=4)
        idx.add(np.zeros((5, 4), dtype=np.float32))
        idx.save(tmp_path / "x.rci")  # Path, not str
        loaded = IndexFlatIP.load(tmp_path / "x.rci")
        assert len(loaded) == 5


class TestFluent:
    def test_add_returns_self(self):
        idx = IndexFlatIP(dim=4)
        result = idx.add(np.zeros((1, 4), dtype=np.float32))
        assert result is idx

    def test_chaining_construction(self):
        X = np.zeros((10, 4), dtype=np.float32)
        idx = IndexFlatIP(dim=4).add(X)
        assert len(idx) == 10
```

Plus a smoke test that the type stub parses cleanly:

```python
class TestTypeStubs:
    def test_pyi_exists(self):
        import rustcluster.index
        from pathlib import Path
        pkg_dir = Path(rustcluster.index.__file__).parent
        assert (pkg_dir / "index.pyi").exists()
        assert (pkg_dir / "py.typed").exists()
```

Plus mypy / pyright as a separate CI check (optional, deferred).

## Milestones

| # | Title | Effort | Tests |
|---|---|---|---|
| 1 | `__repr__` + `__len__` on both classes | 20 min | 2 test classes |
| 2 | Method docstrings on all PyO3 methods | 30 min | smoke test that `help()` returns non-empty |
| 3 | `pathlib.Path` in save/load | 15 min | TestPathlibSupport |
| 4 | Pickle: bytes serializer + `__getstate__`/`__setstate__`/`__getnewargs__` | 1.5-2 hr | TestPickle |
| 5 | `add` / `add_with_ids` return self | 30 min (longer if PyO3 fights) | TestFluent |
| 6 | `index.pyi` + `py.typed` marker | 1 hr | TestTypeStubs |
| 7 | Final regression sweep | 15 min | full pytest, perf-marker test |

**Total: ~4-5 hours.** Fits in one focused session.

## Sequencing

Land in milestone order — each milestone is independently shippable
and the easy ones come first. `add` returning self is the only one
where PyO3 ergonomics could push back; if it does, defer to a later
PR rather than blocking the rest.

## Risks & open questions

1. **In-memory serialization format.** Two options:
   - Reuse the existing safetensors-on-disk layout via a temporary
     `Vec<u8>` buffer. Cleanest because round-trip identical to
     `save`/`load`. Need safetensors' `serialize_to_bytes` if exposed,
     or write to a temp file + read back.
   - Custom binary format (length-prefixed sections). Smaller, simpler,
     no dep on file IO during pickle. Risk of layout drift vs `save`.

   **Decision:** if safetensors has `serialize_to_bytes`, use it. Otherwise
   use a tempdir + read approach, which is ~10 lines and bug-free.

2. **`add` returns self vs FAISS compat.** Returning self breaks
   strict FAISS-API parity (FAISS returns None). Already deviated
   from FAISS by adding `exclude_self`, `similarity_graph`, and
   external-id semantics, so this is a small additional
   deviation in the same direction. Tradeoff: chaining ergonomics
   vs migration friendliness. **Decision: return self**, document it.

3. **mypy / pyright CI.** Adding a typecheck CI job is its own
   ticket. The `.pyi` stub is the prerequisite; CI integration is
   a follow-up.

4. **`__getstate__` size.** A `pickle.dumps(idx)` of a 1M-vector
   1536d IP index is ~6GB. That's fine for pickle but worth
   documenting that `save()` is preferred for large indexes
   (no Python-side serialization overhead).

## What this gets us

After this lands:
- `repr(idx)` shows useful state in REPL/Jupyter.
- `len(idx)` works.
- `pickle.dumps(idx)` and `multiprocessing` Just Work.
- `help(IndexFlatIP.search)` shows real documentation.
- mypy / pyright / IDE autocomplete know the index API.
- `idx.save(Path("..."))` works.
- `IndexFlatIP(dim=128).add(X).search(...)` is valid.

The library moves from "functional but rough" to "polished" without
changing the actual capability surface or perf story. Strictly
incremental, all backwards-compatible (pickle being the only new
behavior; existing users see nothing change).
