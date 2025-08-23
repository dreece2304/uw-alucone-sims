# Phase 06: Visualization Agent Handoff

## Input Contracts

**Expected input files**:
- `03_pca_analysis/*/*_scores.tsv` and `*_loadings.tsv` - PCA results for score/loading plots
- `04_advanced_mvs/nmf/*_analysis.png` - NMF component visualization for integration
- `05_chemical_interpretation/dose_response/*_curves.tsv` - Dose-response data for trend plots
- `05_chemical_interpretation/indices/*_indices.tsv` - Chemical indices for heatmaps
- `07_statistical_analysis/correlation/*_matrix.tsv` - Correlation data for network plots

**Required data format**:
- Score/loading files with consistent sample and peak identifiers
- Dose-response data with dose levels as numeric values (500-15000)
- Chemical indices with sample metadata for grouping and coloring
- Statistical results with p-values and effect sizes for significance annotation

## Output Contracts

**Guaranteed outputs**:
- `06_visualization/publication/` - Publication-ready figures with consistent styling
- `06_visualization/scores/` - PCA score plots colored by dose levels
- `06_visualization/loadings/` - Loading plots with top contributing peaks labeled
- `06_visualization/heatmaps/` - Chemical index and correlation heatmaps
- All plots as both PNG (high-res) and SVG (vector) formats

**Output format specifications**:
- Consistent color scheme: dose levels mapped to viridis colormap
- Figure dimensions optimized for journal publication (300 DPI minimum)
- Font sizes and line weights appropriate for print reproduction
- Statistical annotations (p-values, R², confidence intervals) included where relevant

## Communication with Other Phases

**Upstream requirements from 03_pca_analysis**:
- PCA results with explained variance percentages for axis labeling
- Component interpretations for plot annotations

**Upstream requirements from 04_advanced_mvs**:
- Factor analysis results for multi-method comparison plots
- NLDR embeddings for dimensionality reduction visualization

**Upstream requirements from 05_chemical_interpretation**:
- Chemical indices with clear dose-response trends
- Peak assignments for mass spectrum annotations

**Upstream requirements from 07_statistical_analysis**:
- Statistical significance results for annotation and highlighting
- Correlation matrices for network and heatmap visualization

**Downstream handoff to 08_integration**:
- Complete figure library for manuscript integration
- Consistent visual style for report compilation

## Artifact Storage

**Primary outputs**: `06_visualization/publication/` (journal-ready figures)
**Working plots**: `06_visualization/scores/`, `06_visualization/loadings/`, `06_visualization/heatmaps/` (analysis plots)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.