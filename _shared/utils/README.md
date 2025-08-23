# Shared Utilities Directory

## Purpose

This directory contains centralized utility functions and modules that must be imported by all analysis agents throughout the project. **No agent should reimplement common functionality** - all shared code lives here to ensure consistency, maintainability, and avoid duplication.

## Usage Requirements

**Mandatory imports**: All phase scripts (02_preprocessing, 03_pca_analysis, etc.) MUST import from `_shared/utils/` for common operations rather than implementing their own versions of:
- Data loading and validation
- Statistical functions  
- Plotting utilities
- File I/O operations
- Configuration management
- Provenance tracking

**Import convention**: Use relative imports from the project root:
```python
from _shared.utils.data_io import load_tsv_with_validation
from _shared.utils.stats import calculate_volcano_stats
from _shared.utils.plotting import create_pca_biplot
```

## Code Organization

**Module structure**: Utilities are organized by functional area:
- `data_io.py` - File loading, validation, and format conversion
- `stats.py` - Statistical functions and hypothesis testing
- `plotting.py` - Standardized visualization functions
- `config.py` - Configuration file management and registry access
- `provenance.py` - Metadata tracking and reproducibility helpers

**Version control**: All utility functions must include docstrings with version information and change logs. Breaking changes require version increments and backward compatibility testing.

## Quality Standards

**Testing requirements**: All utility functions must include unit tests with example data. Test files should be placed in `_shared/tests/`.

**Documentation**: Each module must include comprehensive docstrings with parameter descriptions, return value specifications, and usage examples.

**Error handling**: All functions must implement robust error handling with informative messages that include file paths and parameter values.

## Adding New Utilities

**Contribution process**:
1. Check TODO.md for planned utilities before implementing
2. Create function with comprehensive docstring and error handling
3. Add unit tests demonstrating correct behavior
4. Update this README if adding new functional areas
5. Increment version numbers for any breaking changes