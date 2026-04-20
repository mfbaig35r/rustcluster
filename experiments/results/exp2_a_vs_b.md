# Experiment 2: Embedding A vs B

**Question:** Does adding materials and product_use to the embedded text improve clustering alignment with HTS codes?

## Setup
- **Data:** 323,308 CROSS ruling extraction embeddings
- **Embedding A:** description only (~30-80 tokens)
- **Embedding B:** description + materials + product_use (~80-150 tokens)
- **Config:** EmbeddingCluster(K=98, PCA→128, n_init=3, max_iter=50)
- **Ground truth:** 98 HTS chapters

## Results

| Embedding | Purity | NMI | Time |
|-----------|--------|-----|------|
| A (description only) | 0.5864 | 0.5384 | 673s |
| **B (desc + materials + use)** | **0.5995** | **0.5477** | 651s |

**Improvement: +1.3% purity, +0.9% NMI**

## Interpretation

- Adding materials and product_use provides a **measurable but marginal improvement** in clustering alignment with HTS codes.
- The product description alone captures **~98% of the classification-relevant signal** that the richer text provides.
- This suggests that modern embedding models extract material composition and intended use information implicitly from product descriptions — "liquid diet shake drinks containing skimmed milk powder" already encodes both the materials (milk powder) and use (diet drink).
- **For practical applications**, embedding the description alone is sufficient. The cost of embedding longer text (~2x tokens) is not justified by the marginal quality improvement.
