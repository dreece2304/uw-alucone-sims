#!/usr/bin/env python3
"""
Phase 0: Project Conventions Setup
- Establish folder structure and naming conventions
- Document dose mapping and mass families
- Create configuration files
"""

import json
from pathlib import Path

def setup_conventions():
    """Set up project conventions and configuration"""
    
    print("=== Phase 0: Project Conventions Setup ===")
    
    # Dose mapping (fixed)
    dose_map = {
        'SQ1r': 500,
        'SQ1': 2000,
        'SQ2': 5000,
        'SQ3': 10000,
        'SQ4': 15000
    }
    
    # Mass family vocabulary (unit mass)
    mass_families = {
        'positive': {
            'aromatic_proxies': [65, 77, 91, 63, 79],
            'aliphatic': [41, 43, 55, 57, 67, 69],
            'oxygenates': [31, 45, 59, 60, 28, 44],
            'metal_oxides': [27, 43, 59],
            'hydrocarbons': [29, 41, 43, 55]
        },
        'negative': {
            'oxygenates': [31, 45, 59, 60, 28, 44], 
            'al_o_families': [43, 59],
            'h_loss': [16, 17, 18],
            'oxide_formation': [16, 17],
            'al_c_bonds': [39, 55, 71],
            'carbonyl': [44, 45, 60]
        }
    }
    
    # File naming conventions
    naming_conventions = {
        'imzml_files': {
            'pattern': 'P{pattern}_{polarity}.imzML',
            'examples': ['P1_positive.imzML', 'P1_negative.imzML', 'P2_positive.imzML']
        },
        'roi_tsvs': {
            'pattern': 'all_{polarity}_roi.tsv',
            'examples': ['all_positive_roi.tsv', 'all_negative_roi.tsv']
        },
        'column_labels': {
            'pattern': 'P{pattern}_{dose}uC-{pol}',
            'examples': ['P1_500uC-P', 'P2_2000uC-N']
        },
        'index_files': {
            'pattern': 'P{pattern}_{polarity}.index.json',
            'examples': ['P1_positive.index.json', 'P1_negative.index.json']
        }
    }
    
    # Create master configuration
    config = {
        'project_name': 'ToF-SIMS Alucone E-beam Dose Series',
        'analysis_date': '2025-08-22',
        'dose_mapping': dose_map,
        'mass_families': mass_families,
        'naming_conventions': naming_conventions,
        'folders': {
            'data': 'imzML + ibd exports (pos/neg; P1-P3)',
            'out': 'sum-spectrum TSVs and ROI TSVs', 
            'roi': 'ROI definitions and overlays',
            'results': 'analysis results by polarity and method',
            'qc': 'quality control and diagnostics',
            'meta': 'metadata and catalogs'
        },
        'analysis_methods': {
            'normalization': ['baseline_TICsqrt', 'robust_PQNsqrtPareto'],
            'statistical_tests': ['welch_t_test', 'spearman_correlation'],
            'multiple_comparison': 'benjamini_hochberg_fdr',
            'pca_ellipses': 'hotelling_t2_90pct'
        }
    }
    
    # Save configuration
    config_path = Path('roi/config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {config_path}")
    
    # Create folder structure documentation
    folder_structure = """
ToF-SIMS Imaging Analysis Folder Structure
==========================================

data/           # .imzML + .ibd exports (pos/neg; P1–P3)
out/            # existing sum-spectrum TSVs (already done)
├── roi/        # ROI-extracted TSV files
roi/            # ROI definitions and overlays  
├── schema.json # ROI coordinate definitions
├── overlays/   # ROI visualization PNGs
results/        # analysis results
├── positive/
│   ├── baseline_TICsqrt/
│   └── robust_PQNsqrtPareto/
├── negative/
│   ├── baseline_TICsqrt/
│   └── robust_PQNsqrtPareto/
├── comparison/ # ROI vs sum-spectrum comparisons
qc/            # quality control reports
meta/          # metadata catalogs

File Naming Conventions:
- imzML: P1_positive.imzML, P1_negative.imzML, etc.
- ROI TSVs: all_positive_roi.tsv, all_negative_roi.tsv
- Columns: P{pattern}_{dose}uC-{pol} (e.g., P1_500uC-P)
- Index: P{pattern}_{polarity}.index.json

Dose Mapping:
500, 2000, 5000, 10000, 15000 µC/cm² (SQ1r, SQ1, SQ2, SQ3, SQ4)
"""
    
    with open(Path('qc/folder_structure.txt'), 'w') as f:
        f.write(folder_structure)
    
    print("Folder structure documented in: qc/folder_structure.txt")
    
    # Return configuration for use by other phases
    return config

if __name__ == "__main__":
    config = setup_conventions()
    print("\n=== Phase 0 Complete ===")
    print("Project conventions established and documented")