# Experiment 3: PCA Dimension Ablation

**Question:** What PCA target dimension gives the best quality/speed tradeoff for embedding clustering?

## Setup
- **Data:** 323,308 CROSS ruling extraction embeddings (embedding A, 1536d)
- **Ground truth:** 98 HTS chapters
- **Config:** EmbeddingCluster(K=98, n_init=3, max_iter=50)
- **Sweep:** PCA target = 32, 64, 128, 256

## Results

| PCA dim | Purity | NMI | Time | Speedup vs 256 |
|---------|--------|-----|------|----------------|
| 32 | 0.5544 | 0.5149 | 179s | **8.0x** |
| 64 | 0.5595 | 0.5223 | 319s | 4.5x |
| **128** | **0.5864** | **0.5384** | **627s** | **2.3x** |
| 256 | 0.5985 | 0.5407 | 1427s | 1.0x |

## Interpretation

- **Quality improves monotonically** with more PCA dimensions, but with diminishing returns.
- **128 → 256 gains only 1.2% purity** at 2.3x the cost. This is the clearest diminishing-return inflection point.
- **32 dimensions retains 92.6% of the purity of 256** at 8x speed — viable for latency-sensitive applications.
- **All configurations hit max_iter=50** — the algorithm has not fully converged. More iterations would improve all results, but the relative ranking would likely hold.

## Recommendation

- **Default: 128 dimensions** — best quality/speed tradeoff for general use.
- **Speed-critical: 32 dimensions** — 8x faster with only 4.4% purity loss.
- **Quality-critical: 256 dimensions** — marginal improvement, significant cost.
- **Beyond 256 is not worth testing** — the 256→128 gap (1.2%) is smaller than the 128→64 gap (2.7%), suggesting the curve is flattening.
