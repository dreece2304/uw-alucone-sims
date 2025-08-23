#!/usr/bin/env python
"""
Headless runner for PNNL SIMS-PCA analysis.

This script provides a clean CLI interface to run PCA analysis without the
interactive prompts or GUI dependencies from the original main.py and main_gui.py.

Based on PNNL's ATOFSIMSCLASS/pca-analysis main.py workflow.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Add the PNNL source directory to Python path for imports
def setup_pnnl_imports(pca_dir):
    """Add PNNL source directory to sys.path for importing pca_sims."""
    pnnl_src_dir = Path(pca_dir) / "src"
    if pnnl_src_dir.exists():
        sys.path.insert(0, str(pnnl_src_dir))
    else:
        raise FileNotFoundError(f"PNNL source directory not found: {pnnl_src_dir}")


def validate_inputs(args):
    """Validate all input files and directories exist."""
    errors = []
    
    # Check PCA directory
    if not Path(args.pca_dir).exists():
        errors.append(f"PCA directory not found: {args.pca_dir}")
    
    # Check raw data file
    if not Path(args.raw).exists():
        errors.append(f"Raw data file not found: {args.raw}")
    
    # Check catalog file
    if not Path(args.catalog).exists():
        errors.append(f"Catalog file not found: {args.catalog}")
    
    # Check doc mass file
    if not Path(args.doc_mass).exists():
        errors.append(f"Document mass file not found: {args.doc_mass}")
    
    # Validate ion polarity
    if args.ion not in ['P', 'N']:
        errors.append("Ion polarity must be 'P' (positive) or 'N' (negative)")
    
    # Validate top-n
    if args.top_n < 1:
        errors.append("top-n must be a positive integer")
    
    if errors:
        print("ERROR: Input validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def load_catalog_data(catalog_path):
    """
    Load catalog data and create selected_data DataFrame.
    
    Following PNNL's main.py approach: read catalog and select all rows
    as the selected_data DataFrame for PCA analysis.
    
    Args:
        catalog_path: Path to catalog.csv file
        
    Returns:
        pandas.DataFrame: selected_data with all catalog entries
    """
    try:
        # Read the full catalog
        catalog = pd.read_csv(catalog_path)
        
        # PNNL uses selected_data to filter which samples to analyze
        # For headless operation, we'll include all samples in the catalog
        selected_data = catalog.copy()
        
        print(f"Loaded catalog with {len(selected_data)} samples:")
        for _, row in selected_data.iterrows():
            print(f"  Sample {row['Sample #']}: {row['Sample Short Name']}")
        
        return selected_data
        
    except Exception as e:
        print(f"ERROR: Failed to load catalog from {catalog_path}: {e}")
        sys.exit(1)


def run_pca_analysis(args):
    """
    Run the complete PCA analysis workflow.
    
    This follows the main.py workflow:
    1. Load catalog and create selected_data 
    2. Initialize pca_sims instance
    3. Perform PCA analysis
    4. Identify chemical components  
    5. Plot results
    6. Generate report
    """
    
    # Set up imports for PNNL code
    setup_pnnl_imports(args.pca_dir)
    
    # Import pca_sims class (must be after setup_pnnl_imports)
    try:
        from pca_sims.pca_sims import pca_sims
    except ImportError as e:
        print(f"ERROR: Could not import PNNL pca_sims module: {e}")
        print("Make sure --pca-dir points to ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA")
        sys.exit(1)
    
    # Convert paths to absolute paths
    pca_dir = os.path.abspath(args.pca_dir)
    out_dir = os.path.abspath(args.out_dir)
    f_rawsims_data = os.path.abspath(args.raw)
    f_doc_mass = os.path.abspath(args.doc_mass)
    catalog_path = os.path.abspath(args.catalog)
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Load catalog data (following PNNL's approach)
    selected_data = load_catalog_data(catalog_path)
    
    # Convert ion flag to PNNL format
    ion_polarity = 'positive' if args.ion == 'P' else 'negative'
    
    # Get catalog directory (PNNL pca_sims constructor requires this)
    catalog_dir = os.path.dirname(catalog_path)
    
    print(f"\n{'='*60}")
    print("STARTING PNNL SIMS-PCA ANALYSIS")
    print(f"{'='*60}")
    print(f"PCA Directory:     {pca_dir}")
    print(f"Output Directory:  {out_dir}")
    print(f"Raw Data:          {f_rawsims_data}")
    print(f"Catalog:           {catalog_path}")
    print(f"Document Mass:     {f_doc_mass}")
    print(f"Ion Polarity:      {ion_polarity} ({args.ion})")
    print(f"Top Candidates:    {args.top_n}")
    print(f"{'='*60}")
    
    try:
        # Step 1: Initialize pca_sims instance (following main.py line 56)
        print("\n-------->Initializing PCA-SIMS analysis...")
        pcasims = pca_sims(
            f_rawsims_data=f_rawsims_data,
            f_doc_mass=f_doc_mass,
            pcaDir=pca_dir,
            outDir=out_dir,
            positive_or_negative_ion=ion_polarity,
            catalog_dir=catalog_dir,
            selected_data=selected_data
        )
        
        # Step 2: Perform PCA (following main.py line 82)
        print("\n-------->Performing PCA analysis...")
        pcasims.perform_pca()
        
        # Step 3: Identify chemical components (following main.py line 87)
        print(f"\n-------->Identifying chemical components (top {args.top_n})...")
        pcasims.identify_components_from_file(n=args.top_n)
        
        # Step 4: Plot PCA results (following main.py line 90)
        print("\n-------->Generating PCA plots...")
        max_pcacomp = 5  # Following main.py default
        pcasims.plot_pca_result(max_pcacomp=max_pcacomp)
        
        # Step 5: Generate report (following main.py line 93)
        print("\n-------->Generating analysis report...")
        report_filename = "report.docx"
        f_report = os.path.join(out_dir, report_filename)
        
        pcasims.generate_report(
            out_dir=out_dir,
            f_report=f_report,
            ion_sign=ion_polarity,
            max_pcacomp=max_pcacomp
        )
        
        # Success message
        print("\n-------->Analysis Complete!")
        print(f"-------->Report saved to: {f_report}")
        print(f"-------->Check all results in: {out_dir}")
        
        # Return the absolute path to the report
        return f_report
        
    except Exception as e:
        print(f"\nERROR: PCA analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Headless runner for PNNL SIMS-PCA analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python runner/run_pnnl_pca.py \\
    --pca-dir ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA \\
    --out-dir results_run1 \\
    --raw ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA/sims-data/OriginalData/raw_sims.tsv \\
    --catalog ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA/sims-data/Catalog/catalog.csv \\
    --doc-mass ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA/sims-data/positive_doc_mass_record.csv \\
    --ion P \\
    --top-n 5

Note: This script provides a headless alternative to main_gui.py, which requires
GUI dependencies that can be awkward in WSL environments.
        """
    )
    
    parser.add_argument(
        "--pca-dir",
        required=True,
        help="Path to PNNL SIMS_PCA directory (e.g., ATOFSIMSCLASS/SIMS_PCA/SIMS_PCA)"
    )
    
    parser.add_argument(
        "--out-dir", 
        required=True,
        help="Output directory for results (will be created if it doesn't exist)"
    )
    
    parser.add_argument(
        "--raw",
        required=True, 
        help="Path to raw SIMS data TSV file"
    )
    
    parser.add_argument(
        "--catalog",
        required=True,
        help="Path to catalog.csv file with sample metadata"
    )
    
    parser.add_argument(
        "--doc-mass",
        required=True,
        help="Path to document mass CSV file (positive_doc_mass_record.csv or negative_doc_mass_record.csv)"
    )
    
    parser.add_argument(
        "--ion",
        choices=['P', 'N'],
        required=True,
        help="Ion polarity: P (positive) or N (negative)"
    )
    
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top candidates to include in component identification (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Run the analysis
    report_path = run_pca_analysis(args)
    
    # Print final report path
    print(f"\n✓ SUCCESS: Report generated at {report_path}")


if __name__ == "__main__":
    main()