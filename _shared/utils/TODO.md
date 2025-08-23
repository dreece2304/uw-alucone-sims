# Shared Utilities Implementation TODO

## High Priority (Phase 2-3)

**data_io.py**:
- [ ] `load_tsv_with_validation()` - TSV loading with column/format validation
- [ ] `save_tsv_with_metadata()` - Save with provenance sidecar JSON
- [ ] `load_registry()` - Access artifact registry for standard file paths
- [ ] `validate_sample_catalog()` - Ensure catalog consistency with data files

**stats.py**:
- [ ] `calculate_volcano_stats()` - Fold change and significance testing for dose comparisons
- [ ] `dose_response_fitting()` - Monotonic trend analysis and curve fitting
- [ ] `calculate_alucone_indices()` - Chemical indices for aromatic, crosslink, degradation markers
- [ ] `pca_variance_analysis()` - Explained variance and component interpretation helpers

**plotting.py**:
- [ ] `create_pca_biplot()` - Standardized PCA visualization with dose coloring
- [ ] `create_volcano_plot()` - Volcano plots with significance thresholds
- [ ] `create_dose_response_curves()` - Dose-response trend visualization
- [ ] `create_heatmap_with_dendro()` - Clustered heatmaps for chemical families

## Medium Priority (Phase 4-5)

**config.py**:
- [ ] `load_claude_config()` - Parse claude.config.yaml settings
- [ ] `get_write_permissions()` - Validate file write permissions against config
- [ ] `resolve_registry_path()` - Convert registry keys to absolute paths

**roi_utils.py**:
- [ ] `load_roi_schema()` - Parse ROI configuration from JSON
- [ ] `extract_roi_spectra()` - Spatial region extraction from imaging data
- [ ] `validate_roi_boundaries()` - Ensure ROI coordinates are within data bounds
- [ ] `overlay_roi_on_image()` - Generate ROI visualization overlays

## Lower Priority (Phase 6-8)

**imaging.py**:
- [ ] `maf_analysis()` - Minimum/Maximum Autocorrelation Factor implementation
- [ ] `nmf_decomposition()` - Non-negative Matrix Factorization wrapper
- [ ] `mcr_als_fitting()` - Multivariate Curve Resolution setup
- [ ] `spatial_coherence_metrics()` - Calculate spatial clustering measures

**provenance.py**:
- [ ] `record_analysis_provenance()` - Generate standardized metadata JSON
- [ ] `get_git_hash()` - Capture current commit for reproducibility
- [ ] `track_input_dependencies()` - Record all input files used in analysis
- [ ] `validate_reproducibility()` - Check if analysis can be reproduced from metadata

**validation.py**:
- [ ] `cross_validate_normalization()` - Compare normalization method performance  
- [ ] `bootstrap_pca_stability()` - Bootstrap resampling for component stability
- [ ] `leave_one_out_validation()` - Cross-validation for dose classification
- [ ] `permutation_testing()` - Statistical significance via permutation tests

## Implementation Notes

**Priority guidelines**: Implement utilities as they are needed by active phases. Focus on Phase 2-3 utilities first since preprocessing and PCA are immediate needs.

**Code reuse**: Check existing `adapters/` modules before implementing - some functionality may already exist and just needs to be moved to shared utilities.

**Testing strategy**: Each utility should include at least one unit test with synthetic or example data to verify correct behavior.