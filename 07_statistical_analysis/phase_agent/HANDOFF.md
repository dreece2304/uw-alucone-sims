# Phase 07: Statistical Analysis Agent Handoff

## Input Contracts

**Expected input files**:
- `02_preprocessing/normalized/*/*_normalized.tsv` - Normalized data for hypothesis testing
- `03_pca_analysis/baseline/*_scores.tsv` - PCA scores for multivariate statistics
- `05_chemical_interpretation/indices/*_indices.tsv` - Chemical indices as response variables
- `05_chemical_interpretation/dose_response/*_curves.tsv` - Dose trends for validation

**Required data format**:
- Sample metadata with dose assignments as factors (500, 2000, 5000, 10000, 15000)
- Chemical indices as continuous response variables
- PCA scores for multivariate testing and effect size calculations
- Equal sample sizes per dose group for balanced ANOVA designs

## Output Contracts

**Guaranteed outputs**:
- `07_statistical_analysis/anova/` - ANOVA results for dose effects on chemical indices
- `07_statistical_analysis/correlation/` - Correlation matrices between indices and components  
- `07_statistical_analysis/validation/` - Cross-validation and bootstrap results
- Statistical test results, effect sizes, and power analysis summaries

**Output format specifications**:
- ANOVA results TSV: factor, F-statistic, p-value, effect_size, power
- Correlation matrices: symmetric matrices with significance annotations
- Validation results: cross-validation scores, confidence intervals, bootstrap distributions
- Multiple testing corrections applied (Benjamini-Hochberg FDR)

## Communication with Other Phases

**Upstream requirements from 02_preprocessing**:
- Quality-controlled data with known technical variance
- Sample replicates for error estimation

**Upstream requirements from 03_pca_analysis**:
- Principal component scores as multivariate response variables
- Explained variance estimates for effect size contextualization

**Upstream requirements from 05_chemical_interpretation**:
- Chemical indices with biological/chemical relevance
- Dose-response parameters for trend validation

**Downstream handoff to 06_visualization**:
- Statistical significance results for plot annotation
- Correlation matrices for heatmap visualization
- Effect sizes for results interpretation

**Downstream handoff to 08_integration**:
- Statistical validation of all major findings
- Power analysis confirming adequate sample sizes
- Multiple testing corrections for publication

## Artifact Storage

**Primary outputs**: `07_statistical_analysis/anova/` (dose effect testing)
**Correlation data**: `07_statistical_analysis/correlation/` (inter-variable relationships)
**Validation results**: `07_statistical_analysis/validation/` (statistical robustness)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.