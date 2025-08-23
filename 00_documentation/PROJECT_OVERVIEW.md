# ToF-SIMS Alucone Dose-Response Analysis Project Overview

## Aims & Hypotheses

**Dose-dependent chemistry in alucone**: This study investigates how electron beam exposure dose systematically alters the chemical composition of alucone thin films, with the central hypothesis that increasing electron dose (500-15,000 µC/cm²) drives progressive chemical transformations that can be quantitatively tracked using ToF-SIMS mass spectrometry. We expect to observe dose-dependent changes in molecular fragmentation patterns, with higher doses leading to increased cross-linking and structural degradation.

**Expected chemical progression**: We hypothesize that alucone chemistry follows a predictable progression from unexposed pristine polymer → conjugated intermediate states → highly cross-linked networks → graphitized carbon-rich residues as electron dose increases. ToF-SIMS will detect this progression through characteristic molecular ion signatures: pristine alucone showing strong organic polymer fragments, intermediate doses revealing conjugated aromatic species, and high doses displaying carbon cluster ions and reduced organic complexity.

**Positive vs negative mode contributions**: Positive ion mode ToF-SIMS is expected to preferentially detect organic cations, protonated molecules, and metal-containing fragments that reveal polymer backbone structure and cross-linking, while negative ion mode will capture electronegative species, deprotonated organics, and oxidative damage products. The complementary information from both polarities will provide a comprehensive chemical fingerprint of dose-dependent degradation mechanisms.

## Dataset & Design

**Experimental design**: 3 spatial patterns × 5 dose levels (500, 2,000, 5,000, 10,000, 15,000 µC/cm²) × 2 ion polarities = 30 total measurements (15 samples per polarity). Each pattern represents a different spatial location on the alucone substrate, providing biological replicates for statistical analysis.

**Data location**: Processed tabular data resides in `/home/dreece23/sims-pca-ws/out/` as:
- `all_positive_data_renamed.tsv` (positive ion mode)
- `all_negative_data_renamed.tsv` (negative ion mode)
- `all_positive_roi.tsv` and `all_negative_roi.tsv` (region-of-interest extracts)

**ROI and imaging strategy**: Spatial imaging analysis will focus on user-defined regions of interest within each pattern to minimize edge effects and substrate contributions. We will aggregate samples by dose level across all three patterns to enhance statistical power for dose-response modeling.

**Version control policy**: Raw instrument files are not versioned; derivatives and docs are. Large binaries (PNGs, PDFs, NPZs) go through Git LFS.

## Project Structure (Frozen)

```
00_documentation/    - Project documentation hub (literature, methods, reports)
01_raw_data/        - Original instrument files (.itm/.itax) and extraction metadata
02_preprocessing/   - Normalized datasets and quality control analyses  
03_pca_analysis/    - Principal component analysis (baseline, corrected, spatial)
04_advanced_mvs/    - Advanced multivariate methods (NMF, MCR-ALS, NLDR)
05_chemical_interpretation/ - Peak assignments and dose-response modeling
06_visualization/   - Publication-ready plots and heatmaps
07_statistical_analysis/ - ANOVA, correlation, and validation studies
08_integration/     - Final reporting and reproducibility documentation
```

**Canonical folder designations**: 
- Literature root: `00_documentation/literature/` (not `/literature/`)
- Methods documentation: `00_documentation/methods/`
- Final reports: `00_documentation/reports/`

## Methods Summary

**Normalization pathways**: Two parallel normalization strategies will be employed: (1) Total Ion Count (TIC) normalization followed by square-root transformation, and (2) Probabilistic Quotient Normalization (PQN) followed by square-root and Pareto scaling. These approaches address different sources of variance while preserving dose-dependent signal.

**Multivariate analysis**: Principal Component Analysis will use mean-centered features only (no additional scaling) to preserve natural variance structure. Spatial analysis will employ Minimum/Maximum Autocorrelation Factors (MAF), Non-negative Matrix Factorization (NMF), and Multivariate Curve Resolution-Alternating Least Squares (MCR-ALS) to extract spatially coherent chemical patterns.

**Statistical framework**: Paired volcano plot analysis will identify dose-responsive peaks, complemented by custom alucone chemical indices tracking aromatic conjugation, cross-linking, and degradation markers across the dose series.

## Versioning & Provenance

Raw instrument formats (*.itax, *.itm, *.itmx, *.ita, *.imzML, *.ibd) are not versioned.

Large artifacts (PNG, PDF, NPZ) use Git LFS.

Environment is locked via env/requirements-lock.txt (tracked).

## Environment Capture

**Computational reproducibility**: All analysis environments are captured in `env/requirements-lock.txt` with pinned dependency versions to ensure exact computational reproducibility across different systems and time points. Every generated figure and table must record both the environment lock file path and the current git commit hash in an accompanying sidecar JSON file (e.g., `figure1.provenance.json`), enabling complete reconstruction of the computational environment and code state that produced each result. This comprehensive provenance tracking ensures that all findings can be independently verified and reproduced by other researchers.

## Literature Index

| Title | Path | Relevance |
|-------|------|-----------|
| ToF-SIMS analysis of organic electronic materials | `00_documentation/literature/020802_1_online.pdf` | Organic polymer analysis methods and ion fragmentation patterns |
| Chemical mapping of electron beam modifications | `00_documentation/literature/023204_1_6.0003355.pdf` | Electron beam damage mechanisms and dose-response characterization |
| Spatial mass spectrometry of thin films | `00_documentation/literature/023416_1_6.0003249.pdf` | Spatial analysis techniques for thin film characterization |
| Advanced chemometric methods for mass spectrometry | `00_documentation/literature/1-s2.0-S2468023025009423-main.pdf` | Multivariate statistical methods for complex mass spectral datasets |
| Ion formation in polymer ToF-SIMS | `00_documentation/literature/49_1_online.pdf` | Ion formation mechanisms and fragmentation pathways in polymers |
| Statistical analysis of SIMS imaging data | `00_documentation/literature/frans-2-1512520.pdf` | Statistical approaches for SIMS imaging and spatial analysis |

## External Repositories

**PNNL SIMS-PCA GitLab**: [pca-analysis repository](https://gitlab.pnnl.gov/sims-group/pca-analysis) - This external codebase provides the reference implementation for SIMS principal component analysis workflows. We interface with it through our local `runner/run_pnnl_pca.py` adapter script, which translates our TSV format to PNNL's expected input structure and executes their analysis pipeline on our normalized datasets.

## Source of Truth Declarations

**Sample catalogs**: `01_raw_data/metadata/sample_catalog.csv` - Authoritative sample metadata and dose assignments
**Processed TSVs**: `out/all_*_renamed.tsv` - Final analysis-ready datasets 
**Imaging data**: `out/roi/all_*_roi.tsv` - Region-of-interest extracts (future imzML in `08_integration/imaging_pipeline/`)
**ROI definitions**: `roi/schema.json` and `roi/config.json` - Spatial region specifications
**Analysis results**: Phase-specific subfolders in `03_pca_analysis/`, `04_advanced_mvs/`, etc.
**Final reports**: `00_documentation/reports/` - Publication manuscripts and technical reports

## Governance & Change Control

**Phase-based organization**: All new analysis code and outputs must be organized within explicit phase folders (02_preprocessing, 03_pca_analysis, etc.). No analysis artifacts should be placed in the repository root.

**Repository root policy**: Only `README.md` and `Makefile` are permitted at repository root (besides top-level directories). All other files must reside in appropriate subdirectories following the phase structure.

**Registry model**: Path resolution uses registry.v1 schema with separate `files`, `dirs`, and `urls` sections. URLs are references only (not filesystem-validated). All scripts must resolve paths via `_shared/utils/io_utils.py` - no hardcoded absolute paths permitted.

**Write restrictions**: Script outputs must be confined to phase folders or `results/` (for final curated artifacts only). Writes to cross-phase directories (`roi/`, `out/`, `meta/`) are prohibited during analysis execution.

**QC artifact organization**: Quality control outputs must be stored under `02_preprocessing/qc/`. The root `qc/` directory serves as an index only, containing README.md with pointers to actual QC locations.

**Shared utilities versioning**: Changes to shared utility modules in `_shared/utils/` require pull request review and semantic version increments. Breaking changes to data formats or analysis APIs must be documented in changelog.