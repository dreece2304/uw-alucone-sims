# Results Directory

**Purpose**: Final, curated outputs only (figures/tables used in reports/papers).

**Policy**: Working/intermediate files must live in phase folders.

## Content Requirements

### Output Standards
- Only publication-ready figures and tables
- Final, curated outputs that have passed QC validation
- Direct inputs to reports, papers, and presentations
- No intermediate or working files

### Provenance Requirements
Every figure/table **must** have a sidecar JSON file containing:
- **inputs**: Source data files and parameters used
- **git_hash**: Exact commit hash when generated  
- **environment**: Reference to `env/requirements-lock.txt`
- **phase**: Originating analysis phase
- **date**: Generation timestamp

### Sidecar JSON Example
```json
{
  "inputs": [
    "02_preprocessing/normalized/positive/TIC_sqrt_normalized.tsv",
    "roi/config.json"
  ],
  "git_hash": "a1b2c3d4e5f6789...",
  "environment": "env/requirements-lock.txt",
  "phase": "03_pca_analysis", 
  "date": "2025-08-23T14:30:00Z",
  "script": "03_pca_analysis/scripts/phase3_standard_pca.py",
  "parameters": {
    "n_components": 8,
    "normalization": "TIC_sqrt"
  }
}
```

## Directory Structure

```
results/
├── positive/                           # Positive ion mode results
│   ├── baseline_TICsqrt/              # TIC√ normalization outputs
│   │   ├── figure1_pca_scores.png
│   │   ├── figure1_pca_scores.json    # Required sidecar
│   │   ├── table1_loadings.tsv
│   │   ├── table1_loadings.json       # Required sidecar
│   │   └── ...
│   └── robust_PQNsqrtPareto/          # PQN√+Pareto normalization outputs  
│       ├── figure2_volcano_plot.png
│       ├── figure2_volcano_plot.json  # Required sidecar
│       └── ...
├── negative/                          # Negative ion mode results
│   ├── baseline_TICsqrt/
│   └── robust_PQNsqrtPareto/
└── comparison/                        # Cross-polarity comparisons
    ├── dose_response_summary.png
    ├── dose_response_summary.json     # Required sidecar
    └── ...
```

## Workflow Integration

1. **Generation**: Outputs created by phase analysis scripts
2. **Validation**: QC validation in originating phase folder
3. **Curation**: Copy final outputs to `results/` with sidecar JSON
4. **Documentation**: Reference in phase reports and final manuscripts

## Governance Compliance

- **Phase Isolation**: Working files remain in phase directories
- **Reproducibility**: Full provenance via sidecar metadata
- **Quality Control**: Only validated outputs permitted
- **Change Control**: Git tracking for all modifications

---
*Generated: 2025-08-23*  
*Governance: See `00_documentation/GOVERNANCE.md`*