"""EmbeddingCluster experiments on 325K CROSS ruling embeddings.

Runs experiments from the white paper test plan:
1. Cluster recovery at chapter level (K=98)
2. Embedding A vs B comparison
3. PCA dimension ablation
4. Baseline comparison vs sklearn
5. BIC model selection sweep

Usage:
    python experiments/cross_rulings_experiments.py

Requires: data/cross_embeddings.duckdb from hts-api project
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import duckdb

# Add rustcluster to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rustcluster.experimental import EmbeddingCluster

# DuckDB path
DB_PATH = Path(os.environ.get(
    "CROSS_DB",
    os.path.expanduser("~/Projects/hts-api/data/cross_embeddings.duckdb")
))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_data(embedding_col="embedding_a", limit=None):
    """Load embeddings and labels from DuckDB."""
    db = duckdb.connect(str(DB_PATH), read_only=True)

    query = f"""
        SELECT {embedding_col}, hts_chapter, hts_heading, hts_code
        FROM cross_embeddings
        WHERE {embedding_col} IS NOT NULL
          AND hts_chapter IS NOT NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    rows = db.execute(query).fetchall()
    db.close()

    embeddings = np.array([r[0] for r in rows], dtype=np.float32)
    chapters = np.array([r[1] for r in rows])
    headings = np.array([r[2] for r in rows])
    hts_codes = np.array([r[3] for r in rows])

    return embeddings, chapters, headings, hts_codes


def compute_metrics(true_labels, pred_labels):
    """Compute ARI, NMI, and purity without sklearn dependency."""
    n = len(true_labels)
    pred_labels = np.asarray(pred_labels)

    # Purity
    purity_sum = 0
    for cluster_id in np.unique(pred_labels):
        mask = pred_labels == cluster_id
        if mask.sum() == 0:
            continue
        true_in_cluster = true_labels[mask]
        values, counts = np.unique(true_in_cluster, return_counts=True)
        purity_sum += counts.max()
    purity = purity_sum / n

    # Simplified NMI (normalized mutual information)
    # Using contingency table
    pred_unique = np.unique(pred_labels)
    true_unique = np.unique(true_labels)

    # Entropy of predictions
    pred_counts = np.array([np.sum(pred_labels == p) for p in pred_unique])
    pred_probs = pred_counts / n
    h_pred = -np.sum(pred_probs * np.log(pred_probs + 1e-30))

    # Entropy of true
    true_counts = np.array([np.sum(true_labels == t) for t in true_unique])
    true_probs = true_counts / n
    h_true = -np.sum(true_probs * np.log(true_probs + 1e-30))

    # Mutual information
    mi = 0.0
    for p in pred_unique:
        for t in true_unique:
            joint = np.sum((pred_labels == p) & (true_labels == t))
            if joint == 0:
                continue
            p_joint = joint / n
            mi += p_joint * np.log(p_joint / (np.sum(pred_labels == p) / n * np.sum(true_labels == t) / n) + 1e-30)

    nmi = 2 * mi / (h_pred + h_true + 1e-30)

    return {"purity": purity, "nmi": nmi, "n": n}


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


# =====================================================================
# Experiment 1: Cluster recovery at chapter level
# =====================================================================

def experiment_1_chapter_recovery():
    print_header("Experiment 1: Chapter-Level Cluster Recovery (K=98)")

    results = {}
    for emb_name, emb_col in [("A (description)", "embedding_a"), ("B (desc+materials+use)", "embedding_b")]:
        print(f"Loading {emb_name}...")
        embeddings, chapters, _, _ = load_data(emb_col)
        n_chapters = len(np.unique(chapters))
        print(f"  {len(embeddings):,} embeddings, {n_chapters} chapters")

        print(f"  Clustering with K={n_chapters}...")
        t0 = time.perf_counter()
        model = EmbeddingCluster(
            n_clusters=n_chapters,
            reduction_dim=128,
            random_state=42,
            n_init=3,
            max_iter=50,
        )
        model.fit(embeddings)
        elapsed = time.perf_counter() - t0

        labels = np.array(model.labels_)
        metrics = compute_metrics(chapters, labels)

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Purity: {metrics['purity']:.4f}")
        print(f"  NMI: {metrics['nmi']:.4f}")
        print(f"  Objective: {model.objective_:.1f}")
        print(f"  Iterations: {model.n_iter_}")
        print(f"  Resultant lengths: min={min(model.resultant_lengths_):.3f}, max={max(model.resultant_lengths_):.3f}, mean={np.mean(model.resultant_lengths_):.3f}")
        print()

        results[emb_name] = {
            "time_s": elapsed,
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "objective": model.objective_,
            "n_iter": model.n_iter_,
            "n": len(embeddings),
            "k": n_chapters,
        }

    return results


# =====================================================================
# Experiment 2: Embedding A vs B at multiple K
# =====================================================================

def experiment_2_a_vs_b():
    print_header("Experiment 2: Embedding A vs B Comparison")

    embeddings_a, chapters_a, headings_a, _ = load_data("embedding_a")
    embeddings_b, chapters_b, headings_b, _ = load_data("embedding_b")

    results = []
    for k in [20, 50, 98, 200]:
        print(f"K={k}:")
        for name, emb, chapters in [("A", embeddings_a, chapters_a), ("B", embeddings_b, chapters_b)]:
            t0 = time.perf_counter()
            model = EmbeddingCluster(
                n_clusters=k, reduction_dim=128,
                random_state=42, n_init=3, max_iter=50,
            )
            model.fit(emb)
            elapsed = time.perf_counter() - t0

            labels = np.array(model.labels_)
            metrics = compute_metrics(chapters, labels)

            print(f"  {name}: purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")

            results.append({
                "embedding": name, "k": k,
                "purity": metrics["purity"], "nmi": metrics["nmi"],
                "time_s": elapsed,
            })
        print()

    return results


# =====================================================================
# Experiment 3: PCA dimension ablation
# =====================================================================

def experiment_3_pca_ablation():
    print_header("Experiment 3: PCA Dimension Ablation")

    embeddings, chapters, _, _ = load_data("embedding_a")
    n_chapters = len(np.unique(chapters))

    results = []
    for reduction_dim in [None, 256, 128, 64, 32, 16]:
        dim_label = str(reduction_dim) if reduction_dim else "None (1536)"
        print(f"  reduction_dim={dim_label}...")

        t0 = time.perf_counter()
        model = EmbeddingCluster(
            n_clusters=n_chapters,
            reduction_dim=reduction_dim,
            random_state=42,
            n_init=3,
            max_iter=50,
        )
        model.fit(embeddings)
        elapsed = time.perf_counter() - t0

        labels = np.array(model.labels_)
        metrics = compute_metrics(chapters, labels)

        print(f"    purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")

        results.append({
            "reduction_dim": reduction_dim,
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "time_s": elapsed,
            "n_iter": model.n_iter_,
        })

    print()
    print("  Summary:")
    print(f"  {'dim':>10} {'purity':>8} {'nmi':>8} {'time':>8}")
    print(f"  {'-'*38}")
    for r in results:
        dim = r["reduction_dim"] or 1536
        print(f"  {dim:>10} {r['purity']:>8.4f} {r['nmi']:>8.4f} {r['time_s']:>7.1f}s")

    return results


# =====================================================================
# Experiment 4: vs sklearn KMeans
# =====================================================================

def experiment_4_vs_sklearn():
    print_header("Experiment 4: EmbeddingCluster vs sklearn KMeans")

    try:
        from sklearn.cluster import KMeans as SklearnKMeans
        has_sklearn = True
    except ImportError:
        has_sklearn = False
        print("  sklearn not installed, skipping comparison")
        return []

    # Use a subset for sklearn (it may be slow/OOM on 325K x 1536)
    embeddings, chapters, _, _ = load_data("embedding_a", limit=50000)
    n_chapters = len(np.unique(chapters))
    print(f"  Using {len(embeddings):,} embeddings, {n_chapters} chapters")

    results = []

    # EmbeddingCluster (PCA→128)
    print(f"\n  EmbeddingCluster (PCA→128, K={n_chapters})...")
    t0 = time.perf_counter()
    model = EmbeddingCluster(
        n_clusters=n_chapters, reduction_dim=128,
        random_state=42, n_init=3, max_iter=50,
    )
    model.fit(embeddings)
    elapsed = time.perf_counter() - t0
    labels = np.array(model.labels_)
    metrics = compute_metrics(chapters, labels)
    print(f"    purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")
    results.append({"method": "EmbeddingCluster", **metrics, "time_s": elapsed})

    # sklearn KMeans on raw embeddings
    print(f"\n  sklearn KMeans (raw 1536d, K={n_chapters})...")
    t0 = time.perf_counter()
    sk_model = SklearnKMeans(
        n_clusters=n_chapters, n_init=3,
        random_state=42, max_iter=50,
    )
    sk_model.fit(embeddings)
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(chapters, sk_model.labels_)
    print(f"    purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")
    results.append({"method": "sklearn_raw", **metrics, "time_s": elapsed})

    # sklearn KMeans on normalized embeddings (pseudo-cosine)
    print(f"\n  sklearn KMeans (normalized, K={n_chapters})...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    t0 = time.perf_counter()
    sk_model2 = SklearnKMeans(
        n_clusters=n_chapters, n_init=3,
        random_state=42, max_iter=50,
    )
    sk_model2.fit(normalized)
    elapsed = time.perf_counter() - t0
    metrics = compute_metrics(chapters, sk_model2.labels_)
    print(f"    purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")
    results.append({"method": "sklearn_normalized", **metrics, "time_s": elapsed})

    return results


# =====================================================================
# Experiment 5: BIC sweep for model selection
# =====================================================================

def experiment_5_bic_sweep():
    print_header("Experiment 5: BIC Model Selection Sweep")

    # Use a sample for speed
    embeddings, chapters, _, _ = load_data("embedding_a", limit=50000)
    true_k = len(np.unique(chapters))
    print(f"  {len(embeddings):,} embeddings, true K={true_k}")
    print()

    results = []
    k_values = [10, 20, 30, 50, 75, 98, 120, 150, 200]

    for k in k_values:
        print(f"  K={k:>4}...", end=" ", flush=True)
        t0 = time.perf_counter()

        model = EmbeddingCluster(
            n_clusters=k, reduction_dim=128,
            random_state=42, n_init=2, max_iter=30,
        )
        model.fit(embeddings)
        model.refine_vmf()
        elapsed = time.perf_counter() - t0

        labels = np.array(model.labels_)
        metrics = compute_metrics(chapters, labels)

        print(f"BIC={model.bic_:>12.0f}  purity={metrics['purity']:.4f}  nmi={metrics['nmi']:.4f}  time={elapsed:.1f}s")

        results.append({
            "k": k,
            "bic": model.bic_,
            "purity": metrics["purity"],
            "nmi": metrics["nmi"],
            "objective": model.objective_,
            "time_s": elapsed,
        })

    # Find BIC minimum
    best = min(results, key=lambda r: r["bic"])
    print(f"\n  BIC-optimal K = {best['k']} (true K = {true_k})")

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    print(f"CROSS Rulings Embedding Clustering Experiments")
    print(f"Database: {DB_PATH}")
    print(f"{'='*70}")

    all_results = {}

    # Run experiments
    all_results["exp1_chapter_recovery"] = experiment_1_chapter_recovery()
    all_results["exp2_a_vs_b"] = experiment_2_a_vs_b()
    all_results["exp3_pca_ablation"] = experiment_3_pca_ablation()
    all_results["exp4_vs_sklearn"] = experiment_4_vs_sklearn()
    all_results["exp5_bic_sweep"] = experiment_5_bic_sweep()

    # Save results
    output_path = RESULTS_DIR / "cross_rulings_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
