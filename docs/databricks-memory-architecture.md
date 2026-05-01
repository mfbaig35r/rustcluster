# How Memory Actually Works on Databricks

Your cluster says 28 GB. Your Python code gets maybe 12. Here's why.

## The Split

A Databricks node runs two runtimes that share physical RAM:

```
┌───────────────────────────────────────────────────┐
│                  28 GB Node                       │
│                                                   │
│  ┌──────────────────────┐  ┌───────────────────┐  │
│  │    JVM (Spark)       │  │    Python         │  │
│  │    ~14-16 GB         │  │    ~12-14 GB      │  │
│  │                      │  │                   │  │
│  │  ┌────────────────┐  │  │  numpy arrays     │  │
│  │  │ Spark Heap     │  │  │  pandas DFs       │  │
│  │  │ ~10 GB         │  │  │  rustcluster      │  │
│  │  ├────────────────┤  │  │  model objects    │  │
│  │  │ Off-heap       │  │  │  Rust allocations │  │
│  │  │ ~2 GB          │  │  │  (faer matrices)  │  │
│  │  ├────────────────┤  │  │                   │  │
│  │  │ Overhead       │  │  │                   │  │
│  │  │ GC, metadata   │  │  │                   │  │
│  │  │ ~2-4 GB        │  │  └───────────────────┘  │
│  │  └────────────────┘  │                         │
│  └──────────────────────┘                         │
└───────────────────────────────────────────────────┘
```

**The JVM takes its share first.** Databricks configures `spark.driver.memory` (heap) and `spark.driver.memoryOverhead` (off-heap, GC, internal buffers) before Python gets anything. On a Standard_DS4_v2 (28 GB), the JVM typically claims 14-16 GB. Python gets the remainder.

You don't see this split in the UI. The cluster page shows 28 GB. Your code runs in 12.

## JVM Memory Breakdown

### Heap (~60% of JVM allocation)

Where Spark stores:
- Cached DataFrames and Delta table partitions
- Shuffle data (joins, groupBys, repartitions)
- Broadcast variables
- Task execution buffers

Controlled by `spark.driver.memory`. Default varies by instance type — Databricks sets it based on the node spec.

### Off-Heap (~15% of JVM allocation)

- Arrow serialization buffers (used during `toPandas()`)
- Tungsten memory manager (Spark's internal binary format)
- Network buffers for shuffle

### Overhead (~25% of JVM allocation)

- JVM class metadata (metaspace)
- Garbage collector working memory
- Thread stacks (~1 MB per thread × hundreds of threads)
- JNI allocations

## Python Memory Breakdown

Python gets what's left after the JVM. Within Python, memory is consumed by:

### Python Objects

Every Python object has overhead. A Python `float` is 28 bytes (not 8). A `list` of 1536 floats consumes ~43 KB (not 12 KB). This is why `toPandas()` on an embedding column is devastating:

```
312K rows × 1536 floats × 28 bytes/float = 13.4 GB  ← Python list-of-lists
312K rows × 1536 × 4 bytes/float = 1.9 GB            ← numpy f32 array
```

Both exist simultaneously during `np.array(pdf["embedding"].tolist())`. The Python lists aren't freed until the numpy conversion completes and the DataFrame column is dropped.

### Numpy/Native Arrays

Numpy arrays store data in contiguous C memory — no Python object overhead per element. An f32 array is exactly 4 bytes per float, f64 is 8.

### Rust/C Extensions (rustcluster, faer, sklearn)

Native extensions allocate outside Python's heap via the system allocator. These allocations:
- Don't show up in `sys.getsizeof()` or `tracemalloc`
- Do count against the process RSS (and the OOM killer)
- Are invisible to Python's garbage collector

When faer creates a `Mat::from_fn(312000, 1536, ...)`, it allocates 3.8 GB of system memory. Python doesn't know about it. The OOM killer doesn't care who allocated it.

## The toPandas() Problem

`toPandas()` is where most Databricks OOMs happen. Here's what it does:

```
Step 1: Spark serializes DataFrame → Arrow format (JVM off-heap)
Step 2: Arrow buffers transfer to Python process
Step 3: PyArrow converts to Pandas DataFrame (Python heap)
Step 4: JVM releases Arrow buffers

Peak memory: JVM holds Arrow buffers + Python holds Pandas DataFrame
```

For 312K × 1536 embeddings:
- Step 1: ~1.9 GB in JVM off-heap (Arrow f32)
- Step 3: ~5 GB in Python (Pandas with Python float objects in lists)
- Peak: **~7 GB spanning both runtimes** before JVM releases its copy

If you then call `np.array(pdf["embedding"].tolist())`, you add another 1.9 GB while the Python lists still exist.

### Fix: Stream Instead of Collect

```python
# BAD — JVM and Python both hold everything
pdf = df.toPandas()
embeddings = np.array(pdf["embedding"].tolist(), dtype=np.float32)

# GOOD — stream row by row, JVM releases each batch
embeddings = np.vstack([
    np.array(row.embedding, dtype=np.float32)
    for row in df.select("embedding").toLocalIterator()
])
```

`toLocalIterator()` sends one partition at a time from JVM to Python. The JVM releases each partition after Python consumes it. Peak memory is roughly one partition + the growing numpy array.

### Fix: Separate Metadata from Embeddings

```python
# Load metadata (tiny — no embeddings)
pdf = df.select("supplier_name", "commodity", "sub_commodity").toPandas()

# Stream embeddings separately
emb_list = []
for row in df.select("embedding").toLocalIterator():
    emb_list.append(np.array(row.embedding, dtype=np.float32))
embeddings = np.vstack(emb_list)
```

## Single Node vs Multi-Node

### Single Node (Personal Compute)

```
┌──────────────────────────┐
│  Driver = entire cluster │
│  JVM: 14 GB              │
│  Python: 12 GB           │
│  No executors            │
└──────────────────────────┘
```

All computation happens on one machine. `toPandas()` collects to the same (and only) node. Single-node clusters are simpler but memory-constrained — there's nowhere to offload work.

### Multi-Node

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Driver     │   │  Executor 1  │   │  Executor 2  │
│  JVM: 14 GB  │   │  JVM: 14 GB  │   │  JVM: 14 GB  │
│  Python: 12  │   │  Python: 12  │   │  Python: 12  │
└──────────────┘   └──────────────┘   └──────────────┘
```

`toPandas()` collects ALL data to the driver — executors' memory doesn't help. `mapInPandas` and `applyInPandas` run on executors, distributing the memory load. But they add complexity (each partition has its own Python process, independent normalization, etc.).

For rustcluster workloads on reduced data (312K × 128d = 320 MB), single-node is fine. The bottleneck was the full-dimensional data during reduction, not clustering.

## Practical Memory Budget

On a Standard_DS4_v2 (28 GB, 8 cores, single node):

| Available to Python | ~12 GB |
|---|---|
| Minus numpy embeddings (312K × 1536 × f32) | -1.9 GB |
| Minus pandas metadata | -0.1 GB |
| Minus Python interpreter + libraries | -0.5 GB |
| **Remaining for computation** | **~9.5 GB** |

That 9.5 GB has to cover:
- PCA fit (faer centered matrix + intermediates)
- PCA transform (faer centered matrix + output)
- Clustering model internals
- Any temporary allocations

With chunked transform (50K chunks), peak for PCA is ~600 MB per chunk. Without chunking, it's 3.8 GB — leaving almost no headroom.

## Rules of Thumb

1. **Your Python budget is ~40-45% of the node's total RAM.** Plan accordingly.

2. **`toPandas()` on large columns is a trap.** Stream with `toLocalIterator()` or use Arrow directly.

3. **Python lists of floats use 7x more memory than numpy.** Convert to numpy immediately, drop the list.

4. **Native extensions (Rust, C) allocate invisibly.** faer's 3.8 GB matrix doesn't show up in Python profiling tools but still counts against your process limit.

5. **Process data in chunks when possible.** A 50K-row chunk of 1536d f64 is 600 MB. The full 312K is 3.8 GB. The math is the same, the memory isn't.

6. **f32 vs f64 matters.** f32 halves memory for all array operations. For embeddings, f32 has zero quality impact.

7. **The JVM doesn't shrink.** Even if Spark isn't doing anything, the JVM keeps its heap reservation. You can't reclaim it for Python mid-session.

8. **Restart the cluster after OOM.** After a driver crash, JVM memory can be fragmented. A fresh start gives you a clean slate.

## Quick Reference: Instance Memory Budgets

| Instance | Total RAM | Python Budget | Max f32 Embedding Rows (1536d) |
|----------|-----------|--------------|-------------------------------|
| Standard_DS3_v2 | 14 GB | ~5-6 GB | ~500K |
| Standard_DS4_v2 | 28 GB | ~12 GB | ~1.5M |
| Standard_DS5_v2 | 56 GB | ~25 GB | ~3M |
| Standard_E4ds_v5 | 32 GB | ~14 GB | ~1.8M |
| Standard_E8ds_v5 | 64 GB | ~30 GB | ~4M |

"Max f32 rows" assumes 1536d embeddings, numpy array only (no Pandas overhead, no PCA intermediates). Practical limit is ~40% of this when running PCA with chunked transform.
