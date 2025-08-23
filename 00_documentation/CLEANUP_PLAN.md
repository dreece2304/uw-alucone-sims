# Repository Cleanup Plan

**Status**: PROPOSED - Awaiting approval before implementation
**Generated**: 2025-08-23
**Analyst**: Archivist Agent

## Overview

This document identifies repository structure issues that should be addressed to improve organization, eliminate duplicates, and ensure compliance with project governance standards.

## Files in Repository Root Requiring Relocation

### Analysis Scripts (Phase-specific)
**Issue**: Multiple phase scripts scattered in root directory should be moved to appropriate phase folders

**Proposed Moves**:
- `phase0_conventions.py` → `00_documentation/methods/phase0_conventions.py`
- `phase1_catalogs_qc.py` → `01_raw_data/scripts/phase1_catalogs_qc.py`
- `phase1_imaging_readiness.py` → `01_raw_data/scripts/phase1_imaging_readiness.py`
- `phase1_working_extraction.py` → `01_raw_data/scripts/phase1_working_extraction.py`
- `phase2_dual_normalization.py` → `02_preprocessing/scripts/phase2_dual_normalization.py`
- `phase2_roi_definition.py` → `02_preprocessing/scripts/phase2_roi_definition.py`
- `phase3_alucone_indices.py` → `05_chemical_interpretation/scripts/phase3_alucone_indices.py`
- `phase3_roi_extraction.py` → `05_chemical_interpretation/scripts/phase3_roi_extraction.py`
- `phase3_roi_simulation.py` → `05_chemical_interpretation/scripts/phase3_roi_simulation.py`
- `phase4_differential_features.py` → `07_statistical_analysis/scripts/phase4_differential_features.py`
- `phase4_roi_normalization.py` → `02_preprocessing/scripts/phase4_roi_normalization.py`
- `phase5_roi_alucone_indices.py` → `05_chemical_interpretation/scripts/phase5_roi_alucone_indices.py`

### Documentation Files
**Issue**: Documentation files in root should be consolidated under 00_documentation/

**Proposed Moves**:
- `PROJECT_TREE.md` → `00_documentation/PROJECT_TREE.md`
- `README_local.md` → `00_documentation/README_local.md`
- `WORKFLOW.md` → `00_documentation/methods/WORKFLOW.md`

### Temporary/Zone Identifier Files
**Issue**: Temporary or system-generated files should be removed

**Proposed Actions**:
- `023204_1_6.0003355.pdf:Zone.Identifier` → DELETE (Windows download artifact)
- `pySPM_source/023204_1_6.0003355.pdf:Zone.Identifier` → DELETE (Windows download artifact)

## Duplicate Artifacts Analysis

### Literature Files (MAJOR DUPLICATION)
**Issue**: Complete duplication of literature between `/literature/` and canonical `/00_documentation/literature/`

**Canonical Path** (from registry): `00_documentation/literature`

**Duplicate Paths to Remove**:
- `literature/020802_1_online.pdf` (duplicate of `00_documentation/literature/020802_1_online.pdf`)
- `literature/023204_1_6.0003355.pdf` (duplicate of `00_documentation/literature/023204_1_6.0003355.pdf`)
- `literature/023416_1_6.0003249.pdf` (duplicate of `00_documentation/literature/023416_1_6.0003249.pdf`)
- `literature/1-s2.0-S2468023025009423-main.pdf` (duplicate of `00_documentation/literature/1-s2.0-S2468023025009423-main.pdf`)
- `literature/49_1_online.pdf` (duplicate of `00_documentation/literature/49_1_online.pdf`)
- `literature/frans-2-1512520.pdf` (duplicate of `00_documentation/literature/frans-2-1512520.pdf`)

**Additional Zone Identifier Files in Literature**:
- `literature/020802_1_online.pdf:Zone.Identifier` → DELETE
- `literature/023416_1_6.0003249.pdf:Zone.Identifier` → DELETE  
- `literature/1-s2.0-S2468023025009423-main.pdf:Zone.Identifier` → DELETE
- `literature/49_1_online.pdf:Zone.Identifier` → DELETE
- `literature/frans-2-1512520.pdf:Zone.Identifier` → DELETE

**Entire Directory**: Remove entire `/literature/` directory after confirming canonical files exist

### Raw Data Files (MAJOR DUPLICATION)  
**Issue**: Complete duplication of raw instrument files between `/data/` and canonical `/01_raw_data/`

**Duplicate Paths**:
- `data/NegativeIonData/*` (duplicates of `01_raw_data/negative/*`)
- `data/PositiveIonData/*` (duplicates of `01_raw_data/positive/*`)

**Proposed Action**: Remove duplicate raw data directories:
- `data/NegativeIonData/` → REMOVE (keep `01_raw_data/negative/`)
- `data/PositiveIonData/` → REMOVE (keep `01_raw_data/positive/`)

### Metadata Files
**Issue**: Pattern index files duplicated between locations

**Canonical Location**: Registry specifies catalogs in `/meta/`

**Keep**: 
- `meta/pos_catalog.csv` (canonical per registry)
- `meta/neg_catalog.csv` (canonical per registry)

**Evaluate for Removal**:
- `data/P1_negative.index.json` through `data/P3_positive.index.json` (6 files) - verify if needed or duplicated in metadata/

## Missing Phase Agent Scaffolding

**Analysis Complete**: All phase directories (02_preprocessing through 08_integration) have complete phase_agent/ scaffolding with:
- agent.config.yaml
- STATUS.json  
- HANDOFF.md

**No Missing Scaffolding Identified**

## Data Organization Issues

### Test/Mock Files
**Issue**: Test and temporary files mixed with production data

**Files Needing Evaluation**:
- `data/NegDataTest.xlsx` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete
- `data/PosDataTest.xlsx` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete
- `results/mock_catalog.csv` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete
- `results/mock_positive_test.tsv` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete
- `results/test_catalog.csv` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete
- `results/test_positive_data.tsv` → Move to `01_raw_data/working_extraction/` or DELETE if obsolete

### Temporary Outputs
**Files for Review**:
- `test_spatial_output/` → Evaluate if this belongs in `01_raw_data/working_extraction/` or should be deleted

## Implementation Priority

### High Priority (Governance Compliance)
1. Remove duplicate literature directory (`/literature/`) 
2. Remove duplicate raw data directories (`data/NegativeIonData/`, `data/PositiveIonData/`)
3. Move phase scripts from root to appropriate phase directories
4. Delete Zone Identifier files

### Medium Priority (Organization)
1. Move documentation files to `00_documentation/`
2. Consolidate test/mock files
3. Review temporary output directories

### Low Priority (Cleanup)
1. Evaluate metadata index files for necessity
2. Clean up temporary list files (`data/TempNegList.itmil`, `data/TempPosList.itmil`)

## Risk Assessment

**Low Risk**: Most proposed moves involve relocating files to more appropriate locations without changing functionality.

**Medium Risk**: Removing duplicate directories - requires verification that canonical copies contain identical data.

**High Risk**: None identified - all changes maintain data integrity and improve organization.

## FINAL RECOMMENDATIONS: MOVE vs DELETE

### FILES TO MOVE (Preserve All Data)

#### Analysis Scripts → Phase Directories
**MOVE** (all scripts are unique analysis code):
- `phase0_conventions.py` → `00_documentation/methods/phase0_conventions.py`
- `phase1_catalogs_qc.py` → `01_raw_data/scripts/phase1_catalogs_qc.py`
- `phase1_imaging_readiness.py` → `01_raw_data/scripts/phase1_imaging_readiness.py`
- `phase1_working_extraction.py` → `01_raw_data/scripts/phase1_working_extraction.py`
- `phase2_dual_normalization.py` → `02_preprocessing/scripts/phase2_dual_normalization.py`
- `phase2_roi_definition.py` → `02_preprocessing/scripts/phase2_roi_definition.py`
- `phase3_alucone_indices.py` → `05_chemical_interpretation/scripts/phase3_alucone_indices.py`
- `phase3_roi_extraction.py` → `05_chemical_interpretation/scripts/phase3_roi_extraction.py`
- `phase3_roi_simulation.py` → `05_chemical_interpretation/scripts/phase3_roi_simulation.py`
- `phase4_differential_features.py` → `07_statistical_analysis/scripts/phase4_differential_features.py`
- `phase4_roi_normalization.py` → `02_preprocessing/scripts/phase4_roi_normalization.py`
- `phase5_roi_alucone_indices.py` → `05_chemical_interpretation/scripts/phase5_roi_alucone_indices.py`

#### Test/Working Files → Appropriate Locations
**MOVE** (may contain useful test data):
- `data/NegDataTest.xlsx` → `01_raw_data/working_extraction/NegDataTest.xlsx`
- `data/PosDataTest.xlsx` → `01_raw_data/working_extraction/PosDataTest.xlsx`
- `data/TempNegList.itmil` → `01_raw_data/working_extraction/TempNegList.itmil`
- `data/TempPosList.itmil` → `01_raw_data/working_extraction/TempPosList.itmil`
- `results/mock_catalog.csv` → `01_raw_data/working_extraction/mock_catalog.csv`
- `results/mock_positive_test.tsv` → `01_raw_data/working_extraction/mock_positive_test.tsv`
- `results/test_catalog.csv` → `01_raw_data/working_extraction/test_catalog.csv`
- `results/test_positive_data.tsv` → `01_raw_data/working_extraction/test_positive_data.tsv`
- `test_spatial_output/` → `01_raw_data/working_extraction/test_spatial_output/`

### FILES/DIRECTORIES TO DELETE (Verified Duplicates & Obsolete)

#### Literature Duplicates ✓ VERIFIED IDENTICAL
**DELETE** (confirmed MD5 match with canonical in `00_documentation/literature/`):
- `literature/020802_1_online.pdf`
- `literature/023204_1_6.0003355.pdf`
- `literature/023416_1_6.0003249.pdf`
- `literature/1-s2.0-S2468023025009423-main.pdf`
- `literature/49_1_online.pdf`
- `literature/frans-2-1512520.pdf`

#### Raw Data Duplicates ✓ VERIFIED IDENTICAL
**DELETE** (confirmed MD5 match with canonical in `01_raw_data/`):
- `data/NegativeIonData/` (entire directory - 45 files)
- `data/PositiveIonData/` (entire directory - 45 files)

#### Old Documentation Files
**DELETE** (superseded by current comprehensive documentation):
- `PROJECT_TREE.md` (superseded by `00_documentation/PROJECT_OVERVIEW.md`)
- `README_local.md` (superseded by current documentation structure)
- `WORKFLOW.md` (superseded by phase structure and governance docs)

#### System/Temporary Files
**DELETE** (Windows artifacts and empty structure):
- `023204_1_6.0003355.pdf:Zone.Identifier`
- `pySPM_source/023204_1_6.0003355.pdf:Zone.Identifier`
- `literature/020802_1_online.pdf:Zone.Identifier`
- `literature/023416_1_6.0003249.pdf:Zone.Identifier`
- `literature/1-s2.0-S2468023025009423-main.pdf:Zone.Identifier`
- `literature/49_1_online.pdf:Zone.Identifier`
- `literature/frans-2-1512520.pdf:Zone.Identifier`
- `literature/00_documentation/` (empty nested structure)
- `literature/01_raw_data/` (empty nested structure)
- `literature/02_preprocessing/` (empty nested structure)
- `literature/03_pca_analysis/` (empty nested structure)
- `literature/04_advanced_mvs/` (empty nested structure)
- `literature/05_chemical_interpretation/` (empty nested structure)
- `literature/06_visualization/` (empty nested structure)
- `literature/07_statistical_analysis/` (empty nested structure)
- `literature/08_integration/` (empty nested structure)

#### Pattern Index Files (Need Verification)
**EVALUATE FIRST** - check if these duplicate metadata elsewhere:
- `data/P1_negative.index.json`
- `data/P1_positive.index.json`
- `data/P2_negative.index.json`
- `data/P2_positive.index.json`
- `data/P3_negative.index.json`
- `data/P3_positive.index.json`

## Summary
- **MOVE**: 21 items (preserve all unique data and code)
- **DELETE**: ~103+ items (verified duplicates, obsolete docs, and system artifacts)
- **EVALUATE**: 6 index files (verify before deletion)

## Implementation Priority
1. **HIGH**: Delete verified duplicate directories (saves significant space)
2. **MEDIUM**: Move phase scripts to proper locations (governance compliance)
3. **LOW**: Move test files and clean up obsolete documentation