# PyO3 Performance Best Practices

Research conducted 2026-04-18. Patterns drawn from PyO3 docs, rust-numpy, and real-world libraries (polars, tokenizers, pydantic-core).

## 1. Zero-Copy NumPy Input

Use `PyReadonlyArray` for read-only access (most common for input data). It borrows the NumPy buffer directly — no copy.

```rust
use numpy::{PyReadonlyArray2, PyArray1, PyArray2};

#[pymethod]
fn fit(&mut self, py: Python<'_>, x: PyReadonlyArray2<'_, f64>) -> PyResult<()> {
    let array = x.as_array(); // ndarray::ArrayView2<f64>, zero-copy
    py.allow_threads(|| {
        self.centroids = kmeans_inner(&array, self.k, self.max_iter);
    });
    Ok(())
}
```

Key points:
- `PyReadonlyArray2<f64>` gives `ArrayView2<f64>` via `.as_array()` — borrowed view into NumPy's memory, zero-copy
- Use `PyReadwriteArray` only when you need to mutate in-place (rare for inputs)
- **Memory layout**: NumPy defaults to C-contiguous (row-major), matching ndarray's default. If someone passes a Fortran-order array, `.as_array()` still works but strided access is slow. Require contiguous input or check `.is_contiguous()` in Rust.

## 2. GIL Management

Release the GIL for any computation longer than ~1ms. For K-means `fit()`, absolutely release it:

```rust
py.allow_threads(|| {
    // All pure-Rust computation here. No Python objects touched.
    run_kmeans(&data_view, k, max_iter)
})
```

Rules:
- Inside `allow_threads`, you cannot touch any `Py<T>`, `&PyAny`, or Python-bound object
- Copy what you need (scalars, k, max_iter) before entering
- `ArrayView` from `PyReadonlyArray` is safe — it's just a pointer+shape
- The `PyReadonlyArray` itself must not be accessed (its borrow keeps the underlying buffer alive, which is valid across GIL release)

Polars uses this pattern pervasively — every DataFrame operation releases the GIL and runs on Rust's rayon threadpool.

## 3. Performance Pitfalls to Avoid

- **Boundary crossings**: Never call back into Python inside a hot loop. One Rust call that does 1000 iterations, not 1000 Python-to-Rust calls.
- **Unnecessary copies**: Don't `.to_owned()` an `ArrayView` unless you need to mutate it. For centroids built internally, use Rust-owned `Array2<f64>` and convert to NumPy only at return time.
- **GIL contention**: Holding the GIL during a 10-second fit() blocks all Python threads and async tasks. Always release.
- **SIMD defeat**: Use `target-cpu=native` and keep inner loops simple enough for LLVM to vectorize. Avoid iterator chains that defeat autovectorization.

## 4. Returning Results to Python

Allocate in Rust, return as NumPy:

```rust
#[pymethod]
fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>) -> Bound<'py, PyArray1<i64>> {
    let labels: Vec<i64> = assign_clusters(&x.as_array(), &self.centroids);
    PyArray1::from_vec(py, labels) // Rust vec -> NumPy array, one copy (unavoidable)
}
```

For centroids (`Array2<f64>`), use `PyArray2::from_owned_array(py, centroids)` — transfers ownership with one memcpy. No way to avoid this final copy since Rust and Python have separate allocators, but it happens once and is negligible vs compute.

## 5. Real-World Library Patterns

| Library | Pattern |
|---------|---------|
| **Polars** | Releases GIL for all compute, uses Arrow memory (zero-copy to/from pandas), rayon for parallelism |
| **HuggingFace tokenizers** | `PyReadonlyArray` for input, `allow_threads` for encode batches, returns Python lists |
| **pydantic-core** | Minimizes boundary crossings by validating entire models in one Rust call rather than field-by-field |

## 6. Build Configuration

### Cargo.toml

```toml
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
strip = "symbols"
```

### RUSTFLAGS for local builds

```
RUSTFLAGS="-C target-cpu=native"
```

**Important**: Use `target-cpu=native` for local builds only. Distributed wheels need portable codegen.

### Impact

`lto = "fat"` + `codegen-units = 1` gives 10-20% speedups on numerical code by enabling cross-crate inlining.
