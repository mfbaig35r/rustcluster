"""Performance regression tests for rustcluster.index.

These tests are SKIPPED by default. Run them with::

    pytest -m perf

They assert wall-clock budgets relative to FAISS at known shapes so a
regression that drops us back to the v1 baseline (16-25x off FAISS) gets
caught. Margins are generous (3-4x off FAISS or worse on slow CI hardware
is fine; we're catching dropped parallelism, not benchmarking).

If FAISS isn't installed, the tests skip rather than fail — they're a
nice-to-have, not a blocker.
"""

import time

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")
from rustcluster.index import IndexFlatIP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_best(fn, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def synth(n, d, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


# Shapes mirror the v1 walkthrough table. We don't iterate over all of them
# in pytest because each is slow; pick representative ones.
SHAPE_SMALL = (10_000, 128)   # quick smoke on the fast path
SHAPE_LARGE = (100_000, 128)  # the d=128 sweet spot


# ---------------------------------------------------------------------------
# Search top-k
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.parametrize("n,d", [SHAPE_SMALL, SHAPE_LARGE])
def test_search_within_4x_of_faiss(n, d):
    X = synth(n, d, seed=n)
    Q = X[:1000]

    f = faiss.IndexFlatIP(d)
    f.add(X)
    rc = IndexFlatIP(dim=d)
    rc.add(X)

    t_faiss = time_best(lambda: f.search(Q, 10))
    t_rc = time_best(lambda: rc.search(Q, 10))
    ratio = t_rc / t_faiss

    print(f"\n  search n={n} d={d}: FAISS {t_faiss*1000:.1f}ms, rustcluster {t_rc*1000:.1f}ms ({ratio:.2f}x off)")
    # v1.1 baseline measured ~3-4x off. Catch a regression to v1's 16-25x.
    assert ratio < 6.0, f"search regressed to {ratio:.2f}x off FAISS; expected within 6x"


# ---------------------------------------------------------------------------
# Range search
# ---------------------------------------------------------------------------

@pytest.mark.perf
@pytest.mark.parametrize("n,d", [SHAPE_SMALL, SHAPE_LARGE])
def test_range_search_within_3x_of_faiss(n, d):
    X = synth(n, d, seed=n + 1)
    Q = X[:1000]

    f = faiss.IndexFlatIP(d)
    f.add(X)
    rc = IndexFlatIP(dim=d)
    rc.add(X)

    t_faiss = time_best(lambda: f.range_search(Q, 0.5))
    t_rc = time_best(lambda: rc.range_search(Q, 0.5))
    ratio = t_rc / t_faiss

    print(f"\n  range_search n={n} d={d}: FAISS {t_faiss*1000:.1f}ms, rustcluster {t_rc*1000:.1f}ms ({ratio:.2f}x off)")
    assert ratio < 4.0, f"range_search regressed to {ratio:.2f}x off FAISS; expected within 4x"


# ---------------------------------------------------------------------------
# similarity_graph (we should be FASTER than FAISS at d=128)
# ---------------------------------------------------------------------------

@pytest.mark.perf
def test_similarity_graph_beats_faiss_loop_at_d128():
    """At d=128, the fused kernel should beat the FAISS batched range_search
    loop. Wide margin so this is a regression test, not a benchmark."""
    n, d = 50_000, 128
    X = synth(n, d, seed=99)
    threshold = 0.5

    def faiss_flow():
        idx = faiss.IndexFlatIP(d)
        idx.add(X)
        edges = []
        bs = 1000
        for i in range(0, n, bs):
            lims, dists, labs = idx.range_search(X[i:i+bs], threshold)
            for j in range(min(bs, n - i)):
                qi = i + j
                for k in range(lims[j], lims[j + 1]):
                    if labs[k] == qi:
                        continue
                    edges.append((qi, int(labs[k]), float(dists[k])))
        return edges

    rc = IndexFlatIP(dim=d)
    rc.add(X)

    def rc_flow():
        return rc.similarity_graph(threshold=threshold)

    # Build once outside the timer.
    rc_flow()

    t_faiss = time_best(faiss_flow, repeat=2)
    t_rc = time_best(rc_flow, repeat=2)
    speedup = t_faiss / t_rc

    print(f"\n  similarity_graph n={n} d={d}: FAISS loop {t_faiss*1000:.0f}ms, rustcluster {t_rc*1000:.0f}ms ({speedup:.2f}x faster)")
    assert speedup > 1.3, f"similarity_graph speedup dropped to {speedup:.2f}x; expected >1.3x at d=128"


@pytest.mark.perf
def test_similarity_graph_within_2x_at_d1536():
    """At d=1536 (raw text-embedding-3-small), faer's pure-Rust GEMM lags
    BLAS so we don't beat FAISS — but we should stay within 2x. This pins
    the tile-size heuristic at high d (default 384, set in
    similarity_graph::pick_tile_size). A regression to tile=64 would push
    us over the 2x bound."""
    n, d = 20_000, 1536
    X = synth(n, d, seed=1536)
    threshold = 0.5

    def faiss_flow():
        idx = faiss.IndexFlatIP(d)
        idx.add(X)
        for i in range(0, n, 1000):
            idx.range_search(X[i:i+1000], threshold)

    rc = IndexFlatIP(dim=d)
    rc.add(X)
    # Warm up.
    rc.similarity_graph(threshold=threshold)

    t_faiss = time_best(faiss_flow, repeat=2)
    t_rc = time_best(lambda: rc.similarity_graph(threshold=threshold), repeat=2)
    ratio = t_rc / t_faiss

    print(f"\n  similarity_graph n={n} d={d}: FAISS loop {t_faiss*1000:.0f}ms, rustcluster {t_rc*1000:.0f}ms ({ratio:.2f}x off)")
    assert ratio < 2.0, f"similarity_graph at d=1536 regressed to {ratio:.2f}x off FAISS; expected <2.0x"
