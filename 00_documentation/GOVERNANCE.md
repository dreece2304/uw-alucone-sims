# Project Governance & Change Control

## Write-Scope Rules

**Phase-based write restrictions**: All analysis agents and scripts must only write within their designated phase working directory. Agents are restricted to writing in the following directories only:

**Allowed write locations**:
- `02_preprocessing/` - Normalization scripts and QC outputs
- `03_pca_analysis/` - PCA models, scores, loadings, and plots  
- `04_advanced_mvs/` - NMF, MCR-ALS, NLDR outputs
- `05_chemical_interpretation/` - Peak assignments and dose-response models
- `06_visualization/` - Publication plots and heatmaps
- `07_statistical_analysis/` - ANOVA, correlation, validation results
- `08_integration/` - Final reports and reproducibility documentation
- `results/` - Final curated artifacts only (with mandatory sidecar JSON metadata)

**Repository root policy**: Only `README.md` and `Makefile` permitted at repository root (besides top-level directories). All other files must reside in appropriate phase subdirectories.

**QC artifact organization**: All quality control outputs must be stored under `02_preprocessing/qc/`. The root `qc/` directory serves as index only with README.md pointing to actual QC locations.

**Forbidden locations**: No writes to repository root (except README.md/Makefile), `01_raw_data/`, cross-phase directories (`roi/`, `out/`, `meta/`), or other phase directories outside agent's designated scope. Configuration changes require explicit approval.

**Path resolution requirements**: All scripts must resolve paths via `_shared/utils/io_utils.py` using registry.v1 schema. No hardcoded absolute paths permitted. Registry contains separate `files`, `dirs`, and `urls` sections with URLs as references only (not filesystem-validated).

## Data Integrity Rules

**No synthetic data generation**: If required input data is missing, corrupted, or inaccessible, agents must NOT fabricate, interpolate, or generate substitute data. Instead, agents must:

1. Stop processing immediately
2. Create an incident entry in `/_shared/state/status.json` with timestamp, error description, and affected files
3. Report the issue to the user for manual resolution

**Data validation requirements**: All data loading operations must include integrity checks (file existence, format validation, expected column presence) before proceeding with analysis.

## Reproducibility Rules

**Mandatory provenance tracking**: Every generated figure, table, or analysis result must include complete provenance information either in:
- **Figure captions**: Input files, normalization method, analysis parameters, git commit hash
- **Sidecar JSON files**: Machine-readable metadata with same filename + `.provenance.json` extension

**Required metadata fields**:
```json
{
  "input_files": ["path/to/input1.tsv", "path/to/input2.tsv"],
  "analysis_method": "PCA with mean centering",
  "normalization": "TIC + sqrt transformation", 
  "parameters": {"n_components": 8, "center": true},
  "git_hash": "a1b2c3d4e5f6",
  "timestamp": "2025-08-23T10:30:00Z",
  "software_versions": {"python": "3.9.1", "sklearn": "1.0.2"}
}
```

## Version Control & Dependencies

**Shared utilities versioning**: Changes to modules in `_shared/utils/`, `adapters/`, `phase*_*.py` scripts, or configuration files require:
1. Pull request with review by project maintainer
2. Semantic version increment in affected module docstrings
3. Changelog entry documenting breaking changes
4. Backward compatibility testing with existing analysis pipelines

**Dependency management**: All phase scripts must declare their dependencies explicitly in module docstrings. New package requirements must be added to the appropriate environment specification file.

## Quality Assurance

**Analysis validation**: Before marking any analysis as complete, agents must:
1. Generate summary statistics and sanity checks
2. Create diagnostic plots showing data distribution and outliers  
3. Verify results against expected ranges based on literature values
4. Document any anomalies or unexpected findings

**Error handling**: All scripts must implement robust error handling with informative messages, including file paths, parameter values, and suggested remediation steps.

## Change Documentation

**Analysis audit trail**: All significant analysis decisions, parameter choices, and method modifications must be documented in the relevant phase directory's README or analysis log file.

**Result archival**: Intermediate results and model objects must be preserved to enable result reproduction and method comparison. Use consistent naming conventions with timestamps and method identifiers.