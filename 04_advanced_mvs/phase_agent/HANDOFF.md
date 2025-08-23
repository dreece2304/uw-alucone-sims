# Phase 04: Advanced Multivariate Methods Agent Handoff

## Input Contracts

**Expected input files**:
- `02_preprocessing/normalized/*/*_normalized.tsv` - Normalized datasets for algorithm input
- `03_pca_analysis/baseline/positive_pca_scores.tsv` - PCA scores for comparison
- `03_pca_analysis/baseline/positive_pca_loadings.tsv` - PCA loadings for initialization
- Corresponding negative ion mode files

**Required data format**:
- Non-negative intensity values for NMF (post-normalization, no negative values)
- Consistent sample ordering across PCA and normalized input files
- Peak columns as m/z values matching PCA analysis inputs

## Output Contracts

**Guaranteed outputs**:
- `04_advanced_mvs/nmf/` - Non-negative Matrix Factorization results and plots
- `04_advanced_mvs/mcr_als/` - Multivariate Curve Resolution-ALS decomposition
- `04_advanced_mvs/nldr/` - Non-linear Dimensionality Reduction (UMAP, t-SNE)
- Component matrices, transformed data, and method comparison plots

**Output format specifications**:
- NMF components TSV: samples × factors and m/z × factors matrices
- MCR-ALS pure spectra and concentration profiles as separate TSV files
- NLDR embeddings: samples × 2D coordinates for visualization
- Method comparison plots showing NMF vs PCA vs MCR-ALS results

## Communication with Other Phases

**Upstream requirements from 02_preprocessing**:
- Clean, normalized data suitable for non-negative factorization
- Validation that no negative values exist post-normalization

**Upstream requirements from 03_pca_analysis**:
- PCA results as benchmark for component interpretability
- Optimal component number recommendations from PCA scree analysis

**Downstream handoff to 05_chemical_interpretation**:
- NMF components reveal chemically meaningful factor patterns
- MCR-ALS pure spectra enable direct peak assignment
- Factor loadings identify dose-specific chemical signatures

**Downstream handoff to 06_visualization**:
- NLDR embeddings for dose clustering visualization
- Factor comparison plots for methods paper figures

## Artifact Storage

**Primary outputs**: `04_advanced_mvs/nmf/`, `04_advanced_mvs/mcr_als/`, `04_advanced_mvs/nldr/` (method-specific results)
**Comparison outputs**: Cross-method validation and performance metrics
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.