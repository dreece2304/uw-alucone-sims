# Phase 03: PCA Analysis Agent Handoff

## Input Contracts

**Expected input files**:
- `02_preprocessing/normalized/positive/TIC_sqrt_normalized.tsv` - Baseline normalized positive data
- `02_preprocessing/normalized/positive/PQN_sqrt_pareto_normalized.tsv` - Robust normalized positive data
- `02_preprocessing/normalized/negative/TIC_sqrt_normalized.tsv` - Baseline normalized negative data  
- `02_preprocessing/normalized/negative/PQN_sqrt_pareto_normalized.tsv` - Robust normalized negative data
- Corresponding `*_metadata.json` files with normalization parameters

**Required data format**:
- Pre-normalized data ready for mean-centering only
- Sample identifiers encoding dose levels (500, 2000, 5000, 10000, 15000 µC/cm²)
- Peak columns as m/z values with consistent precision across files

## Output Contracts

**Guaranteed outputs**:
- `03_pca_analysis/baseline/` - PCA results using TIC_sqrt normalized data
- `03_pca_analysis/robust/` - PCA results using PQN_sqrt_pareto normalized data
- `03_pca_analysis/spatial/` - MAF analysis for spatial coherence assessment
- Model objects (*.pkl), scores/loadings (*.tsv), and diagnostic plots (*.png)

**Output format specifications**:
- PCA scores TSV: samples × components with dose grouping preserved
- PCA loadings TSV: m/z peaks × components for chemical interpretation
- Summary JSON: explained variance, component count, and model parameters
- Diagnostic plots: scree plots, score plots colored by dose, biplots

## Communication with Other Phases

**Upstream requirements from 02_preprocessing**:
- Normalized data that passes QC validation
- Consistent sample naming across positive/negative polarities
- Metadata confirming normalization method and parameters

**Downstream handoff to 05_chemical_interpretation**:
- PCA loadings identify dose-responsive m/z peaks
- Score plots reveal dose-dependent clustering patterns
- Explained variance guides component selection for chemical indices

**Downstream handoff to 06_visualization**:
- Publication-ready PCA plots with consistent styling
- Both baseline and robust PCA results for method comparison
- Spatial analysis results for imaging integration

## Artifact Storage

**Primary outputs**: `03_pca_analysis/baseline/` and `03_pca_analysis/robust/` (model results)
**Spatial outputs**: `03_pca_analysis/spatial/` (MAF analysis for imaging readiness)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.