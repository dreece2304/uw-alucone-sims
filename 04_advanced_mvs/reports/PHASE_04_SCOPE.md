# Phase 4: Advanced Multivariate Statistical Analysis - Scope & Acceptance Criteria

## Scope

**Objective**: Apply advanced multivariate methods to the alucone dose-response ToF-SIMS dataset to extract chemically meaningful components and patterns beyond PCA.

### Methods to Implement

1. **Non-negative Matrix Factorization (NMF)**
   - Component range: k=2..6
   - Applied to robust_pqn_sqrt (no Pareto) as primary input
   - Per-polarity analysis (positive and negative ion modes)
   - Combined analysis (all polarities)

2. **Multivariate Curve Resolution - Alternating Least Squares (MCR-ALS)**
   - Component number chosen via NMF initial estimates
   - Constraints: non-negativity, unimodality where appropriate
   - Focus on pure component spectra extraction

3. **Non-Linear Dimensionality Reduction (NLDR)**
   - UMAP embedding for dose clustering visualization
   - t-SNE analysis for complementary perspective
   - 2D projections with dose-level coloring

### Primary Inputs

All inputs defined by `manifests/inputs.json` (no other matrices permitted):
- **Primary matrix**: `02_preprocessing/matrices/robust_pqn_sqrt.tsv`
- **Per-polarity matrices**: `robust_pqn_sqrt_pos.tsv`, `robust_pqn_sqrt_neg.tsv`
- **Comparison matrices**: baseline and Pareto variants as needed

## Acceptance Criteria (Phase-Level)

### Dose Response Performance
For at least one method (NMF or MCR-ALS), at least one component must exhibit:
- **|ρ(dose)| ≥ 0.6** (Spearman correlation across dose levels)
- Performance in **≥1 polarity** (positive, negative, or combined)

### Reconstruction Quality
- **Explained variance ≥ 70%** for the selected component number k
- Applies to combined analysis or best individual polarity
- Measured via residual sum of squares or similar metric

### Reproducibility
Components must be stable across analytical variations:
- **Cosine similarity ≥ 0.9** for matched components across 3 random seeds (NMF)
- **OR** MCR-ALS convergence with residual change < 1e-4

### Hygiene Requirements
- **Folder audit passes** with no stray outputs outside allowed paths
- All outputs conform to `manifests/outputs_whitelist.json`
- No forbidden file suffixes or unauthorized directories

## Success Metrics

### Primary Success
1. At least one method achieves dose correlation threshold (≥0.6)
2. Reconstruction quality meets variance threshold (≥70%)
3. Reproducibility criteria satisfied
4. Governance compliance maintained

### Secondary Success
1. Multiple methods show dose-responsive components
2. Per-polarity analysis reveals mode-specific patterns
3. NLDR visualizations show clear dose-level clustering
4. Components interpretable for chemical analysis

## Constraints & Guardrails

- **Input restriction**: Only matrices from Phase-2 handoff
- **No synthetic data**: All inputs must be authentic experimental data
- **Computational environment**: Use existing sims-pca environment only
- **Network isolation**: No external data sources or model downloads
- **Output compliance**: Strict adherence to allowed output paths

## Downstream Handoff

Results will provide:
1. Optimized component matrices for Phase 5 chemical interpretation
2. Component-dose relationships for mechanism studies
3. Method performance comparison for publication
4. Quality-controlled factors for regulatory compliance