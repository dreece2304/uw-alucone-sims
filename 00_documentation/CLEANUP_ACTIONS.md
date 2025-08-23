# Cleanup Actions Documentation

**Date**: 2025-08-23  
**Script**: `scripts/cleanup_remove_macos_artifacts.sh`  
**Status**: Authored but not executed  

## Purpose

Remove macOS metadata artifacts (specifically `__MACOSX` directories) that were created during archive extraction on macOS systems. These directories contain resource forks and metadata that are not needed for the analysis and consume disk space.

## What the Script Does

1. **Checks for presence** of `ATOFSIMSCLASS/__MACOSX/` directory
2. **Reports size** of artifacts found (if any)
3. **Removes directory** using `rm -rf ATOFSIMSCLASS/__MACOSX`
4. **Logs actions** taken to console output

## Targeted Artifacts

- **Primary target**: `ATOFSIMSCLASS/__MACOSX/`
- **Content**: macOS resource forks and metadata files
- **Origin**: Created when ZIP archives are extracted on macOS systems
- **Impact**: Non-essential for analysis, consumes disk space

## How to Run

```bash
# From repository root
bash scripts/cleanup_remove_macos_artifacts.sh
```

**Prerequisites**: 
- Bash shell
- Write permissions in `ATOFSIMSCLASS/` directory

## Safety Considerations

⚠️ **DESTRUCTIVE OPERATION**: This script permanently deletes files

**Before running**:
1. Verify you don't need macOS resource fork metadata
2. Ensure you have backups if uncertain
3. Review the script contents: `cat scripts/cleanup_remove_macos_artifacts.sh`

**Script safety features**:
- Uses `set -euo pipefail` for error handling
- Checks for directory existence before attempting removal
- Reports size before deletion
- Logs all actions taken

## Undo/Recovery

❌ **No undo capability**: macOS artifacts cannot be recovered once deleted

**Prevention**: 
- The ATOFSIMSCLASS repository can be re-cloned from source if needed
- Original ZIP archives can be re-extracted if available

## Expected Outcome

**If artifacts present**:
```
Found macOS artifacts at: ATOFSIMSCLASS/__MACOSX
Size: [size]
Removing ATOFSIMSCLASS/__MACOSX...
✅ Successfully removed: ATOFSIMSCLASS/__MACOSX
```

**If no artifacts**:
```
ℹ️  No macOS artifacts found at: ATOFSIMSCLASS/__MACOSX
```

## Integration with Project

This cleanup script is part of the repository maintenance procedures documented in:
- `00_documentation/CLEANUP_PLAN.md` (general cleanup strategy)  
- `00_documentation/GOVERNANCE.md` (change control policies)
- `PROJECT_TREE.md` (current repository structure)

**Change tracking**: All cleanup actions are logged in `_shared/state/CHANGELOG.md`

---

## QC Preprocessing Relocation

**Date**: 2025-08-23  
**Script**: `scripts/cleanup_move_qc_to_preprocessing.sh`  
**Status**: Authored but not executed  

### Purpose

Reorganize preprocessing-related QC artifacts from the root `qc/` directory to `02_preprocessing/qc/` for better phase organization and governance compliance.

### Files Planned for Relocation

**From `qc/` to `02_preprocessing/qc/`:**
- `qc/normalization_comparison_negative.png` → `02_preprocessing/qc/normalization_comparison_negative.png`
- `qc/normalization_comparison_positive.png` → `02_preprocessing/qc/normalization_comparison_positive.png`  
- `qc/tic_distribution_negative.png` → `02_preprocessing/qc/tic_distribution_negative.png`
- `qc/tic_distribution_positive.png` → `02_preprocessing/qc/tic_distribution_positive.png`
- Files matching `qc/*qc*.*` pattern (includes phase1_qc_*.*)
- `qc/intensity_heatmap_negative.png` → `02_preprocessing/qc/intensity_heatmap_negative.png`
- `qc/intensity_heatmap_positive.png` → `02_preprocessing/qc/intensity_heatmap_positive.png`
- `qc/correlation_matrix_negative.png` → `02_preprocessing/qc/correlation_matrix_negative.png`  
- `qc/correlation_matrix_positive.png` → `02_preprocessing/qc/correlation_matrix_positive.png`

### Files Remaining in `qc/`

**Index-only files** (not moved):
- Phase 3-8 analysis results (differential_summary.txt, roi_*.txt, etc.)
- Generic documentation (folder_structure.txt, validation_summary.txt)
- Plus new `qc/README.md` with index-only policy

### Final Structure

```
02_preprocessing/qc/          # Preprocessing QC artifacts
qc/README.md                  # Index-only policy
qc/[other phase outputs]      # Remain in place
```

### Safety & Recovery

- Uses `git mv` if in git repository, otherwise `mv`
- Non-destructive operation (files moved, not deleted)
- Can be undone by moving files back if needed