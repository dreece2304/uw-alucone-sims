# ROI Directory

**Purpose**: ROI configurations and overlays are cross-phase inputs; not outputs.

**Policy**: Phase tools read from here; no writes into `roi/` during analysis.

## Directory Role

### Input-Only Repository
- **ROI definitions**: Static configuration files defining regions of interest
- **Overlay visualizations**: Reference images showing ROI boundaries
- **Cross-phase resource**: Used as input by multiple analysis phases
- **Read-only during analysis**: No modifications permitted during phase execution

### Usage Pattern
1. **Pre-analysis Setup**: ROI definitions created/validated before analysis phases
2. **Phase Consumption**: Analysis scripts read ROI configs as inputs
3. **No Phase Writes**: Analysis phases never write to `roi/` directory
4. **Governance Enforcement**: Write restrictions prevent accidental modifications

## Current Contents

### Configuration Files
- **`config.json`**: Project-wide configuration including ROI parameters
- **`schema.json`**: ROI coordinate definitions and validation schema

### Overlay Directory
- **`overlays/`**: Visualization images showing ROI boundaries
  - `P1_positive_roi_overlay.png` - Pattern 1 positive mode ROI visualization
  - `P1_negative_roi_overlay.png` - Pattern 1 negative mode ROI visualization
  - `P2_positive_roi_overlay.png` - Pattern 2 positive mode ROI visualization
  - `P2_negative_roi_overlay.png` - Pattern 2 negative mode ROI visualization
  - `P3_positive_roi_overlay.png` - Pattern 3 positive mode ROI visualization
  - `P3_negative_roi_overlay.png` - Pattern 3 negative mode ROI visualization
  - `roi_summary.png` - Overall ROI summary visualization

## Integration with Analysis Phases

### Phases that READ from roi/
- **Phase 2 (Preprocessing)**: ROI extraction and normalization
- **Phase 3 (PCA Analysis)**: ROI-specific component analysis
- **Phase 5 (Chemical Interpretation)**: ROI-based chemical indices
- **Phase 7 (Statistical Analysis)**: ROI differential analysis

### Governance Compliance
- **Input Integrity**: ROI definitions remain stable across analysis phases
- **Reproducibility**: Consistent ROI application across all analyses
- **Change Control**: ROI modifications require explicit governance approval
- **Phase Isolation**: Analysis phases cannot modify shared ROI resources

## ROI Modification Protocol

If ROI definitions need updating:
1. **Stop all analysis phases**
2. **Update ROI configurations** in dedicated maintenance mode
3. **Validate overlays** and regenerate if necessary
4. **Update `_shared/state/CHANGELOG.md`** with ROI changes
5. **Restart analysis** with updated ROI definitions

## File Relationships

```
roi/config.json          → Used by all phases requiring ROI parameters
roi/schema.json          → Validates ROI coordinate definitions  
roi/overlays/*.png       → Reference visualizations (read-only)
↓
Phase analysis scripts   → Read ROI configs as inputs
Phase outputs            → Generated in respective phase directories
```

---
*Generated: 2025-08-23*  
*Policy: Input-only repository for cross-phase ROI resources*  
*Governance: See `00_documentation/GOVERNANCE.md`*