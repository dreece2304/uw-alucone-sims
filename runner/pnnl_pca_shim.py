#!/usr/bin/env python3
"""
PNNL PCA Integration Shim

Provides interface between ToF-SIMS alucone workflow and PNNL SIMS-PCA tools.
Handles input validation, path translation, and output management while 
enforcing write restrictions to designated phase directories.

Version: 0.1.0
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_registry(registry_path="_shared/registry.json"):
    """Load project registry with file path mappings."""
    try:
        with open(registry_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Registry file not found: {registry_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in registry: {e}")
        sys.exit(1)


def validate_input_files(tsv_path, catalog_path):
    """Validate that required input files exist and are readable."""
    missing_files = []
    
    if not os.path.exists(tsv_path):
        missing_files.append(tsv_path)
    if not os.path.exists(catalog_path):
        missing_files.append(catalog_path)
        
    if missing_files:
        print("Error: Missing input files:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    return True


def validate_output_directory(outdir):
    """Ensure output directory is within allowed phase directory."""
    allowed_prefix = "03_pca_analysis/"
    
    if not outdir.startswith(allowed_prefix):
        print(f"Error: Output directory must be within {allowed_prefix}")
        print(f"  Requested: {outdir}")
        return False
    
    return True


def create_output_directory(outdir):
    """Create output directory if it doesn't exist."""
    try:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        return True
    except PermissionError:
        print(f"Error: Permission denied creating directory: {outdir}")
        return False
    except Exception as e:
        print(f"Error creating directory {outdir}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Interface to PNNL SIMS-PCA analysis tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pnnl_pca_shim.py --pol P --in-tsv out/pos_data.tsv --catalog meta/pos_catalog.csv --outdir 03_pca_analysis/baseline --norm baseline
  python pnnl_pca_shim.py --pol N --in-tsv out/neg_data.tsv --catalog meta/neg_catalog.csv --outdir 03_pca_analysis/robust --norm robust --dry-run
        """
    )
    
    parser.add_argument('--pol', 
                       choices=['P', 'N'], 
                       required=True,
                       help='Ion polarity: P (positive) or N (negative)')
    
    parser.add_argument('--in-tsv', 
                       required=True,
                       help='Input TSV file with mass spectral data')
    
    parser.add_argument('--catalog', 
                       required=True,
                       help='Sample catalog CSV file')
    
    parser.add_argument('--outdir', 
                       required=True,
                       help='Output directory (must be within 03_pca_analysis/)')
    
    parser.add_argument('--norm', 
                       choices=['baseline', 'robust'], 
                       required=True,
                       help='Normalization method: baseline (TIC_sqrt) or robust (PQN_sqrt_pareto)')
    
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Validate inputs and exit without running analysis')
    
    args = parser.parse_args()
    
    # Load registry for path mappings
    registry = load_registry()
    
    print(f"PNNL PCA Shim v0.1.0")
    print(f"Polarity: {args.pol}")
    print(f"Input TSV: {args.in_tsv}")
    print(f"Catalog: {args.catalog}")
    print(f"Output directory: {args.outdir}")
    print(f"Normalization: {args.norm}")
    print(f"Dry run: {args.dry_run}")
    print()
    
    # Validate input files exist
    if not validate_input_files(args.in_tsv, args.catalog):
        sys.exit(1)
    
    # Validate output directory is allowed
    if not validate_output_directory(args.outdir):
        sys.exit(1)
    
    # Create output directory
    if not create_output_directory(args.outdir):
        sys.exit(1)
    
    print("✓ Input validation passed")
    print("✓ Output directory validated")
    print(f"✓ Output directory created: {args.outdir}")
    
    if args.dry_run:
        print("\n--- DRY RUN MODE ---")
        print("All validations passed. Would proceed with PNNL PCA analysis.")
        print("Registry loaded successfully:")
        for key, value in registry.items():
            print(f"  {key}: {value}")
        sys.exit(0)
    
    # TODO: Implement actual PNNL PCA execution
    print("\n--- ANALYSIS MODE ---")
    print("PNNL PCA execution not yet implemented")
    print("This is a skeleton version - no external calls will be made")
    
    # Exit successfully for now
    sys.exit(0)


if __name__ == "__main__":
    main()