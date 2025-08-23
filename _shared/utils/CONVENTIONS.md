# Shared Utils Library Conventions

## Core Principles

**No script may write outside its designated phase directory**: All analysis scripts must respect phase boundary restrictions defined in their `phase_agent/agent.config.yaml`. Violation of write boundaries will result in execution termination.

**All scripts must import from _shared/utils**: No analysis script should reimplement common functionality. All shared operations including statistical analysis, data I/O, plotting utilities, and report generation must use functions from this centralized library.

**New shared functions require PR + version bump**: Any addition, modification, or breaking change to shared utility functions requires:
1. Pull request with code review
2. Version increment in `_shared/utils/VERSION`
3. Update to `__init__.py` exports if adding new public functions
4. Comprehensive testing with example data

## Import Conventions

**Standard import pattern**:
```python
from _shared.utils import (
    load_tsv_with_validation,
    paired_volcano_analysis,
    create_pca_biplot,
    write_provenance_sidecar
)
```

**Avoid module-level imports**:
```python
# Discouraged - makes dependencies unclear
from _shared.utils import stats_utils
result = stats_utils.exact_spearman_correlation(x, y)

# Preferred - explicit function import  
from _shared.utils import exact_spearman_correlation
result = exact_spearman_correlation(x, y)
```

## Function Design Standards

**All functions must include comprehensive docstrings** with:
- Purpose description
- Parameter specifications with types and defaults
- Return value descriptions
- Raises section for expected exceptions
- Notes section for important usage details

**Error handling requirements**:
- All functions must handle expected error conditions gracefully
- Use informative error messages including file paths and parameter values
- Prefer specific exception types over generic `Exception`

**Testing requirements**:
- All public functions must include unit tests
- Tests should use synthetic or example data
- Test edge cases and error conditions

## Versioning Policy

**Semantic versioning**: `MAJOR.MINOR.PATCH`
- **MAJOR**: Breaking changes to function signatures or behavior
- **MINOR**: New functions or non-breaking enhancements
- **PATCH**: Bug fixes and documentation updates

**Breaking change protocol**:
1. Deprecation warning in current version
2. Maintain backward compatibility for one minor version
3. Remove deprecated functionality with major version bump

## Code Quality Standards

**Style consistency**:
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter compatible)

**Documentation requirements**:
- All modules must include purpose and version information
- Functions must have NumPy-style docstrings
- Complex algorithms should include mathematical notation in docstring

**Performance considerations**:
- Optimize for readability over micro-optimizations
- Profile performance-critical functions with realistic data sizes
- Document computational complexity for expensive operations

## Maintenance Responsibilities

**Code ownership**:
- Each module has a designated maintainer responsible for reviews
- Breaking changes require approval from project lead
- Bug reports should include minimal reproduction examples

**Dependency management**:
- Minimize external dependencies
- Document all required packages in module docstrings
- Prefer standard library solutions when performance is adequate