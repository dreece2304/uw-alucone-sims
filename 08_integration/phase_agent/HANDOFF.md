# Phase 08: Integration Agent Handoff

## Input Contracts

**Expected input files**:
- `03_pca_analysis/*/*` - All PCA results for methods comparison
- `04_advanced_mvs/*/*` - Advanced multivariate results for comprehensive analysis
- `05_chemical_interpretation/*/*` - Chemical interpretations and dose-response models
- `06_visualization/publication/*` - Publication-ready figures for manuscript integration
- `07_statistical_analysis/*/*` - Statistical validation and significance testing results

**Required data format**:
- Complete analysis pipeline results with consistent provenance metadata
- All figures in both high-resolution PNG and vector SVG formats
- Statistical results with multiple testing corrections applied
- Reproducibility metadata for all major findings

## Output Contracts

**Guaranteed outputs**:
- `08_integration/reports/` - Comprehensive analysis report and manuscript draft
- `08_integration/reproducibility/` - Complete reproducibility package with code and data
- `00_documentation/reports/` - Final technical reports and supplementary materials
- Integrated analysis summary with cross-phase validation

**Output format specifications**:
- Manuscript draft in LaTeX/Markdown with embedded figures and tables
- Reproducibility package with environment specifications and run scripts
- Technical appendix documenting all analysis decisions and parameters
- Cross-phase validation confirming consistency across analysis methods

## Communication with Other Phases

**Upstream requirements from all analysis phases**:
- Complete provenance metadata for reproducibility documentation
- Statistical validation confirming robustness of all major findings
- Publication-quality figures with consistent visual styling

**Final deliverables**:
- Manuscript ready for journal submission
- Complete computational reproducibility package
- Technical documentation for methods replication
- Archive-ready data and results package

**Integration validation**:
- Cross-phase consistency checks (PCA vs NMF component interpretations)
- Statistical coherence across different analysis approaches
- Chemical interpretation validation against literature references

## Artifact Storage

**Primary outputs**: `08_integration/reports/` (final manuscript and technical reports)
**Reproducibility**: `08_integration/reproducibility/` (complete computational package)
**Documentation**: `00_documentation/reports/` (archived final reports)
**Status updates**: Must update `_shared/state/STATUS.json` and append to `_shared/state/CHANGELOG.md` on completion

## Final Handoff

This phase produces the final project deliverables. No downstream phases exist - outputs go directly to publication and archival storage.

## Inter-Agent Protocol

**Before running**: Read upstream `output_contract.json` files from predecessor phases to validate input availability and formats. Verify that input file paths and data shapes match expected specifications.

**After running**: Write your `output_contract.json` documenting all produced artifacts with SHA256 checksums and data schemas. Update both `_shared/state/STATUS.json` (global project status) and local `phase_agent/STATUS.json` (phase-specific status).

**Input validation**: If any input shape, path, or format doesn't match the expected contract specifications, immediately stop execution and record an issues entry in the STATUS.json files. Do not attempt to coerce, transform, or fabricate missing inputs—this violates data integrity requirements.