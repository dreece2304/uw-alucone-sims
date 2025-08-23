# Phase 05: Chemical Interpretation Agent Handoff

## Input Contracts

**Expected input files**:
- `02_preprocessing/normalized/*/*_normalized.tsv` - Normalized intensity data for index calculations
- `03_pca_analysis/baseline/*_pca_loadings.tsv` - PCA loadings identifying dose-responsive peaks
- `04_advanced_mvs/nmf/*_nmf_loadings.tsv` - NMF components for pattern recognition
- `04_advanced_mvs/mcr_als/*_pure_spectra.tsv` - MCR-ALS pure component spectra

**Required data format**:
- Peak columns as m/z values with consistent precision for peak assignment
- Sample metadata encoding dose levels for dose-response modeling
- Component loadings with peak identifications for chemical mapping

## Output Contracts

**Guaranteed outputs**:
- `05_chemical_interpretation/peak_assignment/` - m/z peak assignments to chemical formulae
- `05_chemical_interpretation/dose_response/` - Dose-response curves and monotonicity analysis
- `05_chemical_interpretation/indices/` - Alucone chemical indices (aromatic, crosslink, degradation)
- Peak tables, dose curves, and chemical index time series

**Output format specifications**:
- Peak assignment TSV: m/z, formula, ion_type, confidence_score, literature_reference
- Dose-response TSV: dose_level × chemical_indices with fit parameters
- Index calculations: per-sample values and dose-aggregated trends
- Chemical family groupings based on fragmentation patterns

## Communication with Other Phases

**Upstream requirements from 03_pca_analysis**:
- Loading vectors identifying most dose-responsive m/z peaks
- Component scores revealing dose-dependent clustering

**Upstream requirements from 04_advanced_mvs**:
- NMF components separating chemical families
- MCR-ALS pure spectra for direct spectral interpretation

**Downstream handoff to 06_visualization**:
- Dose-response curves ready for publication plotting
- Chemical indices for heatmap and trend visualization
- Peak assignments for annotated mass spectra figures

**Downstream handoff to 07_statistical_analysis**:
- Chemical indices as response variables for ANOVA testing
- Dose-response parameters for statistical significance testing

## Artifact Storage

**Primary outputs**: `05_chemical_interpretation/dose_response/` (dose trend analysis)
**Chemical data**: `05_chemical_interpretation/indices/` (alucone-specific chemical markers)
**Reference data**: `05_chemical_interpretation/peak_assignment/` (m/z to formula mapping)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.