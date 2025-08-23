# Phase 02: Preprocessing Agent Handoff

## Input Contracts

**Expected input files**:
- `out/all_positive_data_renamed.tsv` - Raw positive ion mode data (samples × m/z peaks)
- `out/all_negative_data_renamed.tsv` - Raw negative ion mode data (samples × m/z peaks)
- `01_raw_data/extracted/*/metadata.json` - Sample metadata with dose assignments

**Required data format**:
- TSV files with first column as sample identifiers (P1_SQ1_500, P2_SQ3_2000, etc.)
- Remaining columns as m/z values (numeric headers like 15.023, 28.014, etc.)
- No missing values in intensity data (zeros acceptable for absent peaks)
- Sample names must encode pattern (P1-P3), dose (500-15000), and replicate info

## Output Contracts

**Guaranteed outputs**:
- `02_preprocessing/normalized/positive/` and `02_preprocessing/normalized/negative/` directories
- Normalization methods: TIC_sqrt, PQN_sqrt_pareto with corresponding metadata.json files
- `02_preprocessing/qc/` - Quality control plots and validation summaries
- `02_preprocessing/phase2_preprocessing_report.txt` - Summary of normalization performance

**Output format specifications**:
- Normalized TSV files maintain same sample×peak structure as inputs
- Metadata JSON files record normalization parameters, scaling factors, and QC metrics
- All outputs include provenance sidecars with input file paths and processing timestamps

## Communication with Other Phases

**Downstream handoff to 03_pca_analysis**:
- Normalized datasets ready for PCA without additional scaling
- QC metrics confirm data quality meets analysis requirements
- Sample groupings preserved for dose-response modeling

**Downstream handoff to 05_chemical_interpretation**:
- Both baseline (TIC_sqrt) and robust (PQN_sqrt_pareto) normalization paths available
- Peak intensity distributions documented for chemical index calculations

## Artifact Storage

**Primary outputs**: `02_preprocessing/normalized/` (canonical normalized datasets)
**QC outputs**: `02_preprocessing/qc/` (diagnostic plots and validation metrics)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.