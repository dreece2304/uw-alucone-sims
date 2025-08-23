import json
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]

def sha256(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_matrix(file_path: Path):
    """Validate a preprocessing matrix file."""
    issues = []
    
    # Check file exists and is non-empty
    if not file_path.exists():
        issues.append(f"File does not exist: {file_path}")
        return issues, None
    
    if file_path.stat().st_size == 0:
        issues.append(f"File is empty: {file_path}")
        return issues, None
    
    try:
        df = pd.read_csv(file_path, sep="\t")
    except Exception as e:
        issues.append(f"Failed to read {file_path}: {e}")
        return issues, None
    
    # Check schema
    if len(df.columns) == 0:
        issues.append(f"No columns in {file_path}")
        return issues, None
    
    # First column must be exactly "Mass (u)"
    if df.columns[0] != "Mass (u)":
        issues.append(f"First column must be 'Mass (u)', found: '{df.columns[0]}'")
    
    # Sample header pattern (accept both uC and μC)
    sample_pattern = re.compile(r'^P\d+_\d+(uC|μC)-[PN]$')
    sample_cols = df.columns[1:]
    
    for col in sample_cols:
        if not sample_pattern.match(col):
            issues.append(f"Sample header doesn't match pattern: '{col}'")
    
    # Check data types and values
    mass_col = df.iloc[:, 0]
    try:
        mass_values = pd.to_numeric(mass_col, errors='coerce')
        if mass_values.isna().any():
            issues.append(f"Non-numeric values in mass column")
    except Exception:
        issues.append(f"Failed to parse mass column as numeric")
    
    # Check intensity columns
    intensity_data = df.iloc[:, 1:]
    try:
        intensity_values = intensity_data.astype(float)
        
        # Check for non-finite values
        if not np.isfinite(intensity_values.values).all():
            issues.append(f"Non-finite values found in intensity data")
        
        # Check for negative values
        if (intensity_values.values < 0).any():
            issues.append(f"Negative values found in intensity data")
            
    except Exception as e:
        issues.append(f"Failed to parse intensity data: {e}")
        return issues, None
    
    # Check for duplicate masses (informational only)
    if not mass_values.isna().any():
        duplicates = mass_values.duplicated().sum()
        if duplicates > 0:
            # This is expected with pos/neg stacking, so just informational
            pass
    
    return issues, df

def compute_agreement_metrics(baseline_df, robust_df):
    """Compute agreement metrics between baseline and robust matrices."""
    # Ensure same shape
    if baseline_df.shape != robust_df.shape:
        return None, None, ["Matrix shapes don't match"]
    
    # Get intensity data (excluding mass column)
    baseline_data = baseline_df.iloc[:, 1:].astype(float).values
    robust_data = robust_df.iloc[:, 1:].astype(float).values
    
    issues = []
    
    # 1. Samplewise Pearson correlation
    sample_correlations = []
    for i in range(baseline_data.shape[1]):  # For each sample
        baseline_sample = baseline_data[:, i]
        robust_sample = robust_data[:, i]
        
        # Remove any rows where either value is NaN or infinite
        valid_mask = np.isfinite(baseline_sample) & np.isfinite(robust_sample)
        if valid_mask.sum() < 10:  # Need at least 10 points for correlation
            continue
            
        try:
            r, _ = pearsonr(baseline_sample[valid_mask], robust_sample[valid_mask])
            if np.isfinite(r):
                sample_correlations.append(r)
        except Exception:
            continue
    
    if len(sample_correlations) == 0:
        samplewise_median = samplewise_mean = 0.0
        issues.append("No valid samplewise correlations could be computed")
    else:
        samplewise_median = np.median(sample_correlations)
        samplewise_mean = np.mean(sample_correlations)
    
    # 2. Mean-spectrum Pearson correlation
    try:
        baseline_mean_spectrum = np.nanmean(baseline_data, axis=1)
        robust_mean_spectrum = np.nanmean(robust_data, axis=1)
        
        valid_mask = np.isfinite(baseline_mean_spectrum) & np.isfinite(robust_mean_spectrum)
        if valid_mask.sum() < 10:
            mean_spectrum_r = 0.0
            issues.append("Insufficient valid data for mean-spectrum correlation")
        else:
            mean_spectrum_r, _ = pearsonr(baseline_mean_spectrum[valid_mask], 
                                        robust_mean_spectrum[valid_mask])
            if not np.isfinite(mean_spectrum_r):
                mean_spectrum_r = 0.0
                issues.append("Mean-spectrum correlation is not finite")
    except Exception as e:
        mean_spectrum_r = 0.0
        issues.append(f"Failed to compute mean-spectrum correlation: {e}")
    
    return (samplewise_median, samplewise_mean, mean_spectrum_r), sample_correlations, issues

def main():
    """Main validation function."""
    baseline_path = ROOT / "02_preprocessing/matrices/baseline_tic_sqrt.tsv"
    robust_path = ROOT / "02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv"
    
    all_issues = []
    
    # Validate individual matrices
    baseline_issues, baseline_df = validate_matrix(baseline_path)
    robust_issues, robust_df = validate_matrix(robust_path)
    
    all_issues.extend(baseline_issues)
    all_issues.extend(robust_issues)
    
    # Compute metrics if both matrices loaded successfully
    metrics = None
    sample_correlations = []
    if baseline_df is not None and robust_df is not None:
        metrics, sample_correlations, agreement_issues = compute_agreement_metrics(baseline_df, robust_df)
        all_issues.extend(agreement_issues)
        
        # Check agreement thresholds
        if metrics:
            samplewise_median, samplewise_mean, mean_spectrum_r = metrics
            
            if samplewise_mean < 0.60:
                all_issues.append(f"Samplewise mean correlation {samplewise_mean:.3f} below threshold 0.60")
            
            if mean_spectrum_r < 0.60:
                all_issues.append(f"Mean-spectrum correlation {mean_spectrum_r:.3f} below threshold 0.60")
    
    # Determine overall status
    validation_ok = len(all_issues) == 0
    
    # Collect file info
    file_info = {}
    for name, path in [("baseline", baseline_path), ("robust", robust_path)]:
        if path.exists():
            file_info[name] = {
                "path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path)
            }
        else:
            file_info[name] = {"path": str(path.relative_to(ROOT)), "exists": False}
    
    # Create JSON report
    json_report = {
        "ok": validation_ok,
        "issues": all_issues,
        "metrics": {
            "samplewise_correlation_median": metrics[0] if metrics else None,
            "samplewise_correlation_mean": metrics[1] if metrics else None,
            "mean_spectrum_correlation": metrics[2] if metrics else None
        } if metrics else None,
        "matrix_info": {
            "baseline_rows": len(baseline_df) if baseline_df is not None else None,
            "robust_rows": len(robust_df) if robust_df is not None else None,
            "baseline_samples": len(baseline_df.columns) - 1 if baseline_df is not None else None,
            "robust_samples": len(robust_df.columns) - 1 if robust_df is not None else None
        },
        "files": file_info
    }
    
    # Write JSON report
    json_path = ROOT / "02_preprocessing/logs/validation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    # Create Markdown report
    md_lines = [
        "# Phase 2 Preprocessing Validation Report",
        "",
        f"- **ok**: {validation_ok}",
        "",
        "## Summary",
        ""
    ]
    
    if baseline_df is not None and robust_df is not None:
        md_lines.extend([
            f"- **Rows**: {len(baseline_df)} (baseline), {len(robust_df)} (robust)",
            f"- **Samples**: {len(baseline_df.columns) - 1}",
            ""
        ])
    
    if metrics:
        samplewise_median, samplewise_mean, mean_spectrum_r = metrics
        md_lines.extend([
            "## Agreement Metrics",
            "",
            f"- **Samplewise correlation (median)**: {samplewise_median:.4f}",
            f"- **Samplewise correlation (mean)**: {samplewise_mean:.4f}",
            f"- **Mean-spectrum correlation**: {mean_spectrum_r:.4f}",
            "",
            f"Threshold: 0.60 for all metrics",
            ""
        ])
    
    if all_issues:
        md_lines.extend([
            "## Issues",
            ""
        ])
        for issue in all_issues:
            md_lines.append(f"- {issue}")
        md_lines.append("")
    else:
        md_lines.extend([
            "## Issues",
            "",
            "None detected.",
            ""
        ])
    
    # File information
    md_lines.extend([
        "## File Information",
        ""
    ])
    
    for name, info in file_info.items():
        if info.get("exists", True):
            md_lines.extend([
                f"### {name.title()} Matrix",
                f"- Path: `{info['path']}`",
                f"- Size: {info['size_bytes']:,} bytes",
                f"- SHA256: `{info['sha256']}`",
                ""
            ])
        else:
            md_lines.extend([
                f"### {name.title()} Matrix",
                f"- Path: `{info['path']}`",
                f"- Status: **File not found**",
                ""
            ])
    
    # Write Markdown report
    md_path = ROOT / "02_preprocessing/VALIDATION_REPORT.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"Validation complete. ok: {validation_ok}")
    print(f"Reports written to:")
    print(f"  - {json_path}")
    print(f"  - {md_path}")
    
    if all_issues:
        print(f"\nIssues found:")
        for issue in all_issues:
            print(f"  - {issue}")

if __name__ == "__main__":
    main()