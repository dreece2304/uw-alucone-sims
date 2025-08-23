# PNNL PCA Integration Shim

## Purpose

This module provides an interface layer between our ToF-SIMS alucone analysis workflow and the PNNL SIMS-PCA analysis tools. The shim translates our data formats and directory structure to match PNNL's expected inputs while ensuring all outputs are written to designated phase directories.

## PNNL PCA Repository

**Source**: [PNNL ATOFSIMSCLASS/pca-analysis](https://gitlab.com/pacific-northwest-national-laboratory/ATOFSIMSCLASS/pca-analysis)

The PNNL repository provides the reference implementation for SIMS principal component analysis workflows, including standardized preprocessing, PCA computation, and visualization routines.

## Input Requirements

**Catalog file**: Sample metadata CSV with columns for sample names, dose assignments, and grouping variables. Format must match PNNL catalog schema with proper column headers and data types.

**TSV data file**: Mass spectral data in samples×peaks format with sample names in first column matching catalog entries. Peak columns should use m/z values as headers with consistent precision.

**Configuration**: Analysis parameters including normalization method (baseline=TIC_sqrt, robust=PQN_sqrt_pareto), component count, and output preferences.

## Output Specifications  

**PCA scores**: Sample×component matrix showing principal component coordinates for each sample, saved as TSV with proper sample identifiers.

**PCA loadings**: Peak×component matrix indicating contribution of each m/z to principal components, enabling chemical interpretation of dose-response patterns.

**Scree plot data**: Explained variance ratios and cumulative variance for component selection and interpretation.

**Diagnostic plots**: Score plots, loading plots, and biplots for visual assessment of PCA results and dose-dependent clustering.

## Write Restrictions

**Designated output directory**: All PNNL PCA outputs must be written exclusively to `03_pca_analysis/` subdirectories (baseline/, robust/, spatial/) to maintain phase isolation and governance compliance.

**No repository root writes**: The shim enforces strict write controls preventing any output generation in the project root or unauthorized directories.

## Integration Workflow

1. **Input validation**: Verify catalog and TSV files exist and conform to expected formats
2. **Path translation**: Convert our file paths to PNNL-expected structure  
3. **Parameter mapping**: Translate normalization methods to PNNL parameter values
4. **Execution**: Call PNNL analysis with appropriate arguments and environment
5. **Output relocation**: Move/rename PNNL outputs to our phase directory structure
6. **Provenance tracking**: Generate metadata documenting PNNL version and parameters used