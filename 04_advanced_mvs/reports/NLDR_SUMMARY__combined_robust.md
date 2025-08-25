# NLDR Summary Report: combined_robust

## Dataset Overview

- **Samples**: 15
- **Features**: 1853
- **Methods**: t-SNE

## t-SNE Results

**Trustworthiness**: 0.890 ✅

### Dose Progression Analysis
- **Trend Assessment**: clear
- **Separation Ratio**: 9.39
- **Consecutive Dose Distance**: 67.958
- **Within-Dose Spread**: 7.237

### Pattern Overlap Analysis
- **Overlap Assessment**: high  
- **Inter-Pattern Distance**: 1.598
- **Intra-Pattern Spread**: 66.403

## Qualitative Assessment

### Dose Ordering
The dose progression forms **clear ordered trajectories** in the embedding space. Higher doses are well-separated from lower doses, indicating strong dose-dependent chemical changes.

### Pattern Separation
The experimental patterns (P1, P2, P3) show **high overlap**, suggesting limited reproducibility differences between patterns.

## Methodology

- **UMAP Parameters**: n_neighbors=5, min_dist=0.1, metric=cosine
- **t-SNE Parameters**: perplexity=5, metric=cosine
- **Preprocessing**: StandardScaler normalization
- **Random Seed**: 42 (for reproducibility)

---

*Generated: 2025-08-24 22:20:33*
