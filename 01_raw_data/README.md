# 01_raw_data Policy

This directory contains only audit artifacts and minimal processing helpers, following strict size and format policies:

• **No raw instrument files** (.itax, .itm, .itmx, .ita, .imzML, .ibd) - store externally
• **File size limit**: 50 MB per file - larger files trigger warnings  
• **Allowed content**: Audit reports, JSON metadata, NPZ sidecars, helper scripts only
• **Working extraction**: Limited scratch space under working_extraction/ (keep small, .gitignored)
• **Spatial data**: Optional NPZ stacks in spatial/ subfolder with mandatory sidecar JSONs

## Allowed Directories
- `scripts/` - Helper scripts for data validation
- `spatial/` - Optional NPZ spatial stacks  
- `phase_agent/` - Phase-specific agent configuration
- `working_extraction/` - Temporary scratch space

## Allowed Files
- `AUDIT_REPORT.md` - Phase 1 audit summary
- `tsv_audit.json` - Machine-readable audit results
- `imaging_index.json` - Spatial data catalog
- `README.md` - This policy document

## Allowed File Types
- `*.sidecar.json` - NPZ metadata sidecars
- `*.npz` - Spatial data arrays (if kept here)
- `*.sha256` - File integrity checksums