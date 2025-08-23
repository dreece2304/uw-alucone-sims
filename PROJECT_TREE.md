# ToF-SIMS Alucone Dose-Response Analysis - Project Tree

**Generated**: 2025-08-23  
**Status**: Post-cleanup organized structure  
**Registry Version**: registry.v1

## Repository Structure

```
sims-pca-ws/
├── 00_documentation/                    # Project documentation hub
│   ├── CLEANUP_PLAN.md                 # Repository reorganization plan
│   ├── GOVERNANCE.md                   # Project governance and change control
│   ├── PROJECT_OVERVIEW.md             # Comprehensive project overview
│   ├── PROJECT_TREE.md                 # This file - repository structure
│   ├── literature/                     # Literature collection (canonical)
│   │   ├── 020802_1_online.pdf
│   │   ├── 023204_1_6.0003355.pdf
│   │   ├── 023416_1_6.0003249.pdf
│   │   ├── 1-s2.0-S2468023025009423-main.pdf
│   │   ├── 49_1_online.pdf
│   │   └── frans-2-1512520.pdf
│   ├── methods/                        # Analysis methods documentation
│   │   └── phase0_conventions.py       # Project conventions and standards
│   └── reports/                        # Phase completion reports
│       ├── PHASE_SUMMARY.template.md   # Template for phase summaries
│       └── README.md                   # Reporting protocol documentation
│
├── 01_raw_data/                        # Original instrument data and metadata
│   ├── extracted/                      # Processed extraction results
│   │   ├── extraction_results.json
│   │   ├── extraction_summary.png
│   │   ├── format_comparison_summary.tsv
│   │   ├── hdf5/                      # HDF5 format exports
│   │   ├── json/                      # JSON metadata exports
│   │   ├── npz/                       # NumPy compressed arrays
│   │   └── tsv/                       # TSV format exports
│   ├── metadata/                      # Sample catalogs and metadata
│   │   ├── project_metadata.json
│   │   └── sample_catalog.csv         # Authoritative sample metadata
│   ├── negative/                      # Negative ion mode raw data (canonical)
│   │   └── [45 .itax/.itm/.itmx files]
│   ├── positive/                      # Positive ion mode raw data (canonical)
│   │   └── [45 .itax/.itm/.itmx files]
│   ├── scripts/                       # Phase 1 analysis scripts
│   │   ├── phase1_catalogs_qc.py
│   │   ├── phase1_imaging_readiness.py
│   │   └── phase1_working_extraction.py
│   └── working_extraction/            # Test files and temporary outputs
│       ├── NegDataTest.xlsx
│       ├── PosDataTest.xlsx
│       ├── TempNegList.itmil
│       ├── TempPosList.itmil
│       ├── mock_catalog.csv
│       ├── mock_positive_test.tsv
│       ├── test_catalog.csv
│       ├── test_positive_data.tsv
│       └── test_spatial_output/
│
├── 02_preprocessing/                   # Data normalization and QC
│   ├── normalized/                    # Normalized datasets
│   │   ├── negative/                  # TIC, PQN, Sum, Vector normalizations
│   │   └── positive/                  # Multiple scaling methods
│   ├── phase2_preprocessing_report.txt
│   ├── phase2_preprocessing_stats.json
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   ├── qc/                           # Quality control outputs
│   │   ├── phase1_qc_report.txt
│   │   ├── phase1_qc_results.json
│   │   ├── phase1_qc_summary.png
│   │   ├── negative_preprocessing_comparison.png
│   │   └── positive_preprocessing_comparison.png
│   └── scripts/                      # Phase 2 and 4 normalization scripts
│       ├── phase1_complete_data_extraction.py
│       ├── phase1_data_qc.py
│       ├── phase2_dual_normalization.py
│       ├── phase2_preprocessing_pipeline.py
│       ├── phase2_roi_definition.py
│       └── phase4_roi_normalization.py
│
├── 03_pca_analysis/                   # Principal Component Analysis
│   ├── baseline/                      # TIC_sqrt normalized PCA results
│   │   ├── negative_pca_*.tsv/.pkl/.png
│   │   └── positive_pca_*.tsv/.pkl/.png
│   ├── comparison/
│   ├── corrected/
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   ├── robust/
│   ├── scripts/                      # PCA analysis scripts
│   │   ├── phase3_standard_pca.py
│   │   ├── phase3_statistical_corrections.py
│   │   └── phase4_spatial_maf.py
│   └── spatial/                      # MAF spatial analysis
│       ├── negative_maf_*.tsv/.json/.png
│       └── positive_maf_*.tsv/.json/.png
│
├── 04_advanced_mvs/                   # Advanced multivariate methods
│   ├── mcr_als/
│   ├── nldr/
│   ├── nmf/                          # Non-negative Matrix Factorization
│   │   ├── negative_nmf_*.tsv/.json/.png
│   │   └── positive_nmf_*.tsv/.json/.png
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   └── scripts/
│       └── phase5_nmf_analysis.py
│
├── 05_chemical_interpretation/         # Peak assignment and chemical indices
│   ├── dose_response/
│   ├── indices/
│   ├── peak_assignment/
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   └── scripts/                      # Chemical interpretation scripts
│       ├── phase3_alucone_indices.py
│       ├── phase3_roi_extraction.py
│       ├── phase3_roi_simulation.py
│       └── phase5_roi_alucone_indices.py
│
├── 06_visualization/                  # Publication-ready plots and figures
│   ├── heatmaps/
│   ├── loadings/
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   ├── publication/
│   └── scores/
│
├── 07_statistical_analysis/           # ANOVA, correlation, validation
│   ├── anova/
│   ├── correlation/
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   ├── scripts/
│   │   └── phase4_differential_features.py
│   └── validation/
│
├── 08_integration/                    # Final reporting and reproducibility
│   ├── imaging_pipeline/
│   ├── phase_agent/                   # Phase management
│   │   ├── HANDOFF.md
│   │   ├── STATUS.json
│   │   └── agent.config.yaml
│   ├── reports/
│   └── reproducibility/
│
├── ATOFSIMSCLASS/                     # External PNNL PCA analysis tool
│   ├── DISCLAIMER.rst
│   ├── LICENSE.rst
│   ├── PCA_Analysis_Manual_rev4.docx
│   ├── README.md
│   └── SIMS_PCA/                     # PNNL analysis suite
│       └── [Complete PNNL PCA implementation]
│
├── pySPM_source/                      # External pySPM library source
│   ├── LICENSE
│   ├── README.md
│   ├── doc/                          # Documentation and tutorials
│   ├── pySPM/                        # Core library modules
│   ├── pyproject.toml
│   ├── resources/
│   └── tests/
│
├── adapters/                          # Data format conversion utilities
│   ├── __init__.py
│   ├── cli.py                        # Command-line interface
│   ├── iontof_*.py                   # ION-TOF instrument adapters
│   ├── catalog_maker.py
│   ├── check_spatial_data.py
│   └── [13 additional adapter modules]
│
├── env/                              # Environment specifications
│   ├── environment.yml               # Conda environment
│   └── requirements-lock.txt         # Pinned dependencies (canonical)
│
├── meta/                             # Sample catalogs (canonical)
│   ├── neg_catalog.csv
│   └── pos_catalog.csv
│
├── out/                              # Processed TSV files (canonical)
│   ├── all_negative_data.meta.txt
│   ├── all_negative_data.tsv
│   ├── all_negative_data_renamed.tsv  # Registry: neg_tsv
│   ├── all_positive_data.meta.txt
│   ├── all_positive_data.tsv
│   ├── all_positive_data_renamed.tsv  # Registry: pos_tsv
│   ├── raw_sims.meta.txt
│   ├── raw_sims.tsv
│   └── roi/                          # Region-of-interest extracts
│       ├── all_negative_roi.tsv
│       └── all_positive_roi.tsv
│
├── qc/                               # Quality control plots and summaries
│   ├── correlation_matrix_*.png
│   ├── intensity_heatmap_*.png
│   ├── mean_spectrum_*.png
│   ├── normalization_comparison_*.png
│   ├── tic_distribution_*.png
│   └── [10 additional QC files]
│
├── results/                          # Analysis results by polarity/method
│   ├── negative/
│   │   ├── baseline_TICsqrt/         # Baseline normalization results
│   │   └── robust_PQNsqrtPareto/     # Robust normalization results
│   └── positive/
│       ├── baseline_TICsqrt/
│       └── robust_PQNsqrtPareto/
│
├── roi/                              # ROI configuration and overlays
│   ├── config.json
│   ├── schema.json
│   └── overlays/                     # ROI visualization overlays
│       ├── P1_negative_roi_overlay.png
│       ├── P1_positive_roi_overlay.png
│       └── [5 additional overlay images]
│
├── runner/                           # Analysis tool interfaces
│   ├── PNNL_PCA_SHIM.md             # PNNL integration documentation
│   ├── pnnl_pca_shim.py             # PNNL PCA interface
│   └── run_pnnl_pca.py              # PNNL runner script
│
├── _shared/                          # Shared utilities and project state
│   ├── contracts/                    # Inter-agent communication templates
│   │   ├── input_contract.template.json
│   │   └── output_contract.template.json
│   ├── registry.json                 # Artifact registry (v1 schema)
│   ├── state/                        # Project status tracking
│   │   ├── CHANGELOG.md
│   │   └── STATUS.json
│   └── utils/                        # Shared utility library
│       ├── __init__.py
│       ├── CONVENTIONS.md
│       ├── README.md
│       ├── TODO.md
│       ├── VERSION
│       ├── io_utils.py
│       ├── pca_utils.py
│       ├── report_utils.py
│       ├── roi_utils.py
│       └── stats_utils.py
│
├── claude.config.yaml                # Claude Code configuration
├── Makefile                          # Build and analysis targets
├── RUN_RULES.md                      # Analysis execution rules
└── [Root configuration files]
```

## Key Statistics

**Total Phases**: 8 (00-08)
**Phase Agent Scaffolding**: ✅ Complete (7/7 analysis phases)
**Literature Files**: 6 PDFs in canonical location
**Raw Data Files**: 90 instrument files (45 positive, 45 negative)
**External Dependencies**: 2 (ATOFSIMSCLASS, pySPM_source)
**Analysis Scripts**: 12 organized by phase
**Shared Utilities**: 5 modules with comprehensive API

## Governance Compliance

- ✅ **Phase Isolation**: All analysis scripts in appropriate phase directories
- ✅ **Canonical Paths**: Registry defines authoritative file locations
- ✅ **Write Restrictions**: Agent configs enforce phase boundary controls
- ✅ **Documentation**: Comprehensive governance and protocol documentation
- ✅ **Provenance Tracking**: Status, changelog, and contract systems in place
- ✅ **No Duplicates**: All verified duplicates removed during cleanup

## Recent Changes (Post-Cleanup)

**Removed**: ~100+ duplicate files (literature PDFs, raw data, system artifacts)
**Reorganized**: 12 phase scripts moved to appropriate directories
**Consolidated**: Test files moved to `01_raw_data/working_extraction/`
**Updated**: Registry schema to v1 with organized structure
**Cleaned**: All Zone Identifier files and empty directories removed

---
*Last Updated: 2025-08-23*  
*Registry Version: registry.v1*  
*Governance: See `00_documentation/GOVERNANCE.md`*