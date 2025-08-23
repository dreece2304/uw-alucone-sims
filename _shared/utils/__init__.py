"""
Shared utilities package for ToF-SIMS alucone dose-response analysis.

This package provides centralized utility functions for statistical analysis,
PCA operations, ROI handling, I/O operations, and report generation.

Version: 0.1.0
"""

# Statistical analysis utilities
from .stats_utils import (
    exact_spearman_correlation,
    benjamini_hochberg_fdr,
    paired_volcano_analysis,
    calculate_rsd_percent,
    dose_response_monotonicity
)

# PCA analysis utilities
from .pca_utils import (
    mean_center_rows,
    align_pc_signs_to_dose,
    compute_hotelling_t2_ellipse,
    create_loadings_table,
    overlay_raw_spectra_on_loadings,
    compute_explained_variance_ci
)

# Region of Interest utilities
from .roi_utils import (
    validate_roi_schema,
    extract_roi_from_npz,
    extract_roi_from_imzml,
    create_fixed_scale_image,
    overlay_roi_boundaries,
    calculate_roi_statistics
)

# Input/Output utilities
from .io_utils import (
    load_registry,
    resolve_safe_path,
    write_provenance_sidecar,
    load_tsv_with_validation,
    save_analysis_results,
    backup_critical_files
)

# Report generation utilities
from .report_utils import (
    generate_figure_caption_with_provenance,
    create_html_results_table,
    create_markdown_summary_table,
    compile_analysis_summary_report,
    format_statistical_results_table,
    create_methods_section_text
)

# Package metadata
__version__ = '0.1.0'
__author__ = 'ToF-SIMS Analysis Team'
__email__ = 'contact@example.com'
__description__ = 'Shared utilities for ToF-SIMS alucone dose-response analysis'

# Public API - functions that should be imported by analysis scripts
__all__ = [
    # Statistical functions
    'exact_spearman_correlation',
    'benjamini_hochberg_fdr', 
    'paired_volcano_analysis',
    'calculate_rsd_percent',
    'dose_response_monotonicity',
    
    # PCA functions
    'mean_center_rows',
    'align_pc_signs_to_dose',
    'compute_hotelling_t2_ellipse',
    'create_loadings_table',
    'overlay_raw_spectra_on_loadings',
    'compute_explained_variance_ci',
    
    # ROI functions
    'validate_roi_schema',
    'extract_roi_from_npz',
    'extract_roi_from_imzml',
    'create_fixed_scale_image',
    'overlay_roi_boundaries',
    'calculate_roi_statistics',
    
    # I/O functions
    'load_registry',
    'resolve_safe_path',
    'write_provenance_sidecar',
    'load_tsv_with_validation',
    'save_analysis_results',
    'backup_critical_files',
    
    # Report functions
    'generate_figure_caption_with_provenance',
    'create_html_results_table',
    'create_markdown_summary_table',
    'compile_analysis_summary_report',
    'format_statistical_results_table',
    'create_methods_section_text'
]