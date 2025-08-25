# NLDR Summary Report: pos_robust

## Dataset Overview

- **Samples**: 15
- **Features**: 930
- **Methods**: t-SNE

## t-SNE Results

**Trustworthiness**: 0.945 ✅

### Dose Progression Analysis
- **Trend Assessment**: clear
- **Separation Ratio**: 11.16
- **Consecutive Dose Distance**: 58.231
- **Within-Dose Spread**: 5.218

### Pattern Overlap Analysis
- **Overlap Assessment**: high  
- **Inter-Pattern Distance**: 3.731
- **Intra-Pattern Spread**: 52.674

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

*Generated: 2025-08-24 22:20:34*
