#!/usr/bin/env python3
"""Audit TSV files and rebuild catalogs (idempotent)."""
import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

def validate_tsv(tsv_path: Path, polarity: str) -> dict:
    """Validate a single TSV file."""
    results = {
        "file": str(tsv_path),
        "polarity": polarity,
        "valid": True,
        "issues": [],
        "metrics": {}
    }
    
    try:
        # Read TSV
        df = pd.read_csv(tsv_path, sep='\t')
        
        # Check first column
        if df.columns[0] != "Mass (u)":
            results["valid"] = False
            results["issues"].append(f"First column must be 'Mass (u)', got '{df.columns[0]}'")
        
        # Check mass column
        mass_col = df.iloc[:, 0]
        if not mass_col.is_monotonic_increasing:
            results["valid"] = False
            results["issues"].append("Mass column not strictly increasing")
        
        if not all(mass_col.astype(int) == mass_col):
            results["valid"] = False
            results["issues"].append("Mass column contains non-integer values")
        
        # Check column count (1 mass + 15 samples)
        if len(df.columns) != 16:
            results["valid"] = False
            results["issues"].append(f"Expected 16 columns (1 + 15), got {len(df.columns)}")
        
        # Check sample headers - Accept both μ (mu) and u characters
        sample_pattern = re.compile(r'^P[123]_(500|2000|5000|10000|15000)[μu]?C-[PN]$')
        sample_cols = df.columns[1:]  # Skip Mass column
        
        invalid_headers = []
        for col in sample_cols:
            if not sample_pattern.match(col):
                invalid_headers.append(col)
        
        if invalid_headers:
            results["valid"] = False
            results["issues"].append(f"Invalid sample headers: {invalid_headers}")
        
        # Check intensities
        intensity_data = df.iloc[:, 1:].values  # All except mass column
        
        if np.any(intensity_data < 0):
            results["valid"] = False
            results["issues"].append("Negative intensities found")
        
        if np.any(~np.isfinite(intensity_data)):
            results["valid"] = False
            results["issues"].append("Non-finite intensities found")
        
        # Compute TICs
        tics = np.sum(intensity_data, axis=0)
        if np.any(tics <= 0):
            results["valid"] = False
            results["issues"].append("Samples with TIC <= 0 found")
        
        # Store metrics
        results["metrics"] = {
            "n_masses": len(df),
            "n_samples": len(sample_cols),
            "mass_range": [int(mass_col.min()), int(mass_col.max())],
            "tic_range": [float(tics.min()), float(tics.max())],
            "zero_counts": {col: int(np.sum(df[col] == 0)) for col in sample_cols}
        }
        
    except Exception as e:
        results["valid"] = False
        results["issues"].append(f"Error reading file: {str(e)}")
    
    return results

def build_catalog(tsv_path: Path, polarity: str) -> list:
    """Build sample catalog from TSV."""
    df = pd.read_csv(tsv_path, sep='\t')
    sample_cols = df.columns[1:]  # Skip Mass column
    
    catalog = []
    sample_num = 1
    
    # Sort by pattern, dose, polarity
    pattern_order = {"P1": 1, "P2": 2, "P3": 3}
    dose_order = {"500": 1, "2000": 2, "5000": 3, "10000": 4, "15000": 5}
    
    def sort_key(col):
        # Extract pattern and dose from column name like "P1_2000μC-P"
        parts = col.split('_')
        pattern = parts[0] if len(parts) > 0 else "P1"
        dose_part = parts[1].split('μC')[0] if len(parts) > 1 else "500"
        return (pattern_order.get(pattern, 999), dose_order.get(dose_part, 999))
    
    sorted_cols = sorted(sample_cols, key=sort_key)
    
    for col in sorted_cols:
        # Parse column name
        match = re.match(r'(P[123])_(\d+)μC-([PN])', col)
        if match:
            pattern, dose, pol = match.groups()
            catalog.append({
                "Sample #": sample_num,
                "Pattern": pattern,
                "Dose (uC/cm2)": int(dose),
                "Polarity": pol,
                "ColumnName": col
            })
            sample_num += 1
    
    return catalog

def main():
    root = Path(__file__).parent.parent
    
    # Input files
    pos_tsv = root / "out/all_positive_data_renamed.tsv"
    neg_tsv = root / "out/all_negative_data_renamed.tsv"
    
    # Validate TSVs
    pos_results = validate_tsv(pos_tsv, "positive")
    neg_results = validate_tsv(neg_tsv, "negative")
    
    # Build audit summary
    audit = {
        "audit_date": "2025-08-23",
        "positive": pos_results,
        "negative": neg_results,
        "overall_valid": pos_results["valid"] and neg_results["valid"]
    }
    
    # Write audit JSON
    audit_json = root / "01_raw_data/tsv_audit.json"
    with open(audit_json, 'w') as f:
        json.dump(audit, f, indent=2)
    
    # Write audit report
    audit_md = root / "01_raw_data/AUDIT_REPORT.md"
    with open(audit_md, 'w') as f:
        f.write("# TSV Data Audit Report\n\n")
        f.write("**Date:** 2025-08-23\n")
        f.write("**Files Audited:**\n")
        f.write("- `out/all_positive_data_renamed.tsv`\n")
        f.write("- `out/all_negative_data_renamed.tsv`\n\n")
        
        f.write("## Schema Validation\n\n")
        for results in [pos_results, neg_results]:
            status = "PASS ✓" if results["valid"] else "FAIL ✗"
            f.write(f"**{results['polarity'].title()} TSV:** {status}\n")
            if results["issues"]:
                for issue in results["issues"]:
                    f.write(f"- {issue}\n")
            f.write(f"- Masses: {results['metrics'].get('n_masses', 'N/A')}\n")
            f.write(f"- Samples: {results['metrics'].get('n_samples', 'N/A')}\n")
            if "mass_range" in results["metrics"]:
                f.write(f"- Mass range: {results['metrics']['mass_range'][0]}-{results['metrics']['mass_range'][1]}\n")
            if "tic_range" in results["metrics"]:
                f.write(f"- TIC range: {results['metrics']['tic_range'][0]:.0f}-{results['metrics']['tic_range'][1]:.0f}\n")
            f.write("\n")
        
        f.write("## Overall Status\n\n")
        if audit["overall_valid"]:
            f.write("**PASS ✓** - All validation criteria met\n")
        else:
            f.write("**FAIL ✗** - Issues found, see details above\n")
    
    # Build catalogs
    if pos_results["valid"]:
        pos_catalog = build_catalog(pos_tsv, "P")
        catalog_path = root / "meta/pos_catalog.csv"
        catalog_path.parent.mkdir(exist_ok=True)
        
        with open(catalog_path, 'w') as f:
            f.write("Sample #,Pattern,Dose (uC/cm2),Polarity,ColumnName\n")
            for entry in pos_catalog:
                f.write(f"{entry['Sample #']},{entry['Pattern']},{entry['Dose (uC/cm2)']},{entry['Polarity']},{entry['ColumnName']}\n")
    
    if neg_results["valid"]:
        neg_catalog = build_catalog(neg_tsv, "N")
        catalog_path = root / "meta/neg_catalog.csv"
        catalog_path.parent.mkdir(exist_ok=True)
        
        with open(catalog_path, 'w') as f:
            f.write("Sample #,Pattern,Dose (uC/cm2),Polarity,ColumnName\n")
            for entry in neg_catalog:
                f.write(f"{entry['Sample #']},{entry['Pattern']},{entry['Dose (uC/cm2)']},{entry['Polarity']},{entry['ColumnName']}\n")
    
    print(f"TSV audit complete. Overall valid: {audit['overall_valid']}")
    print(f"Results: {audit_json}")
    print(f"Report: {audit_md}")

if __name__ == "__main__":
    main()