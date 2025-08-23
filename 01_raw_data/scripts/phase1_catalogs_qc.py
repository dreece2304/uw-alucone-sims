#!/usr/bin/env python3
"""
Phase 1: Catalogs & Enhanced QC Analysis
- Create metadata catalogs for both polarities
- Perform comprehensive quality control checks
- Generate validation summary
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

def parse_column_headers(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Parse column headers into structured metadata"""
    catalog_data = []
    
    for col in df.columns:
        if col == 'Mass (u)':
            continue
            
        # Parse format: P1_2000μC-P or P1_2000μC-N
        match = re.match(r'P(\d+)_(\d+)μC-([PN])', col)
        if match:
            pattern_num = int(match.group(1))
            dose = int(match.group(2))
            pol = match.group(3)
            
            if pol == polarity:
                catalog_data.append({
                    'Sample #': col,
                    'Pattern': f'P{pattern_num}',
                    'Dose (uC/cm2)': dose,
                    'Polarity': pol
                })
            else:
                print(f"Warning: Polarity mismatch in {col}, expected {polarity}")
        else:
            print(f"Warning: Could not parse column header: {col}")
    
    return pd.DataFrame(catalog_data)

def enhanced_qc_analysis(df: pd.DataFrame, polarity: str) -> Dict:
    """Comprehensive quality control analysis"""
    qc_results = {
        'polarity': polarity,
        'total_samples': len(df.columns) - 1,  # Excluding Mass column
        'total_masses': len(df),
        'mass_range': {'min': df['Mass (u)'].min(), 'max': df['Mass (u)'].max()},
        'sample_stats': {},
        'data_quality': {},
        'outliers': []
    }
    
    # 1. Mass Range Analysis
    masses = df['Mass (u)'].values
    qc_results['mass_gaps'] = []
    for i in range(1, len(masses)):
        if masses[i] - masses[i-1] > 1:
            qc_results['mass_gaps'].append((masses[i-1], masses[i]))
    
    # Check for non-integer masses
    non_integer = masses[masses != np.round(masses)]
    qc_results['non_integer_masses'] = len(non_integer)
    
    # Check for non-increasing masses
    decreasing = np.where(np.diff(masses) <= 0)[0]
    qc_results['decreasing_masses'] = len(decreasing)
    
    # 2. Sample Intensity Validation
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    
    # First pass: calculate basic stats for all samples
    for col in sample_cols:
        intensities = df[col].values
        
        # Basic stats
        total_intensity = np.sum(intensities)
        mean_intensity = np.mean(intensities)
        nonzero_count = np.sum(intensities > 0)
        negative_count = np.sum(intensities < 0)
        
        qc_results['sample_stats'][col] = {
            'total_intensity': total_intensity,
            'mean_intensity': mean_intensity,
            'nonzero_peaks': nonzero_count,
            'negative_intensities': negative_count,
            'max_intensity': np.max(intensities),
            'zero_sum': total_intensity == 0
        }
    
    # Second pass: identify outliers after all samples are processed
    tic_values = [qc_results['sample_stats'][c]['total_intensity'] for c in sample_cols]
    median_tic = np.median(tic_values)
    std_tic = np.std(tic_values)
    
    for col in sample_cols:
        total_intensity = qc_results['sample_stats'][col]['total_intensity']
        if abs(total_intensity - median_tic) > 3 * std_tic:
            qc_results['outliers'].append({
                'sample': col,
                'reason': 'TIC outlier',
                'value': total_intensity,
                'median': median_tic,
                'z_score': (total_intensity - median_tic) / std_tic if std_tic > 0 else 0
            })
    
    # 3. Overall Data Quality Metrics
    all_tics = [stats['total_intensity'] for stats in qc_results['sample_stats'].values()]
    qc_results['data_quality'] = {
        'tic_mean': np.mean(all_tics),
        'tic_std': np.std(all_tics),
        'tic_cv': np.std(all_tics) / np.mean(all_tics) * 100,
        'tic_range': {'min': np.min(all_tics), 'max': np.max(all_tics)},
        'zero_sum_samples': sum(1 for stats in qc_results['sample_stats'].values() if stats['zero_sum']),
        'samples_with_negatives': sum(1 for stats in qc_results['sample_stats'].values() if stats['negative_intensities'] > 0)
    }
    
    return qc_results

def generate_qc_plots(df: pd.DataFrame, polarity: str, output_dir: Path):
    """Generate QC diagnostic plots"""
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    
    # 1. TIC distribution plot
    tics = []
    labels = []
    for col in sample_cols:
        tics.append(np.sum(df[col].values))
        labels.append(col)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(tics)), tics, color='skyblue', alpha=0.7)
    plt.xticks(range(len(tics)), labels, rotation=45, ha='right')
    plt.ylabel('Total Ion Count (TIC)')
    plt.title(f'Sample TIC Distribution - {polarity} Mode')
    plt.tight_layout()
    plt.savefig(output_dir / f'tic_distribution_{polarity.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Intensity heatmap (top 50 most variable masses)
    # Calculate coefficient of variation for each mass
    intensity_matrix = df.set_index('Mass (u)')[sample_cols]
    cvs = intensity_matrix.std(axis=1) / intensity_matrix.mean(axis=1)
    top_variable = cvs.nlargest(50).index
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(intensity_matrix.loc[top_variable], 
                cmap='viridis', 
                cbar_kws={'label': 'Intensity'},
                xticklabels=True, 
                yticklabels=True)
    plt.title(f'Top 50 Most Variable Masses - {polarity} Mode')
    plt.xlabel('Samples')
    plt.ylabel('Mass (u)')
    plt.tight_layout()
    plt.savefig(output_dir / f'intensity_heatmap_{polarity.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mass spectrum overview (mean across all samples)
    mean_spectrum = intensity_matrix.mean(axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.plot(mean_spectrum.index, mean_spectrum.values, linewidth=0.5, alpha=0.7)
    plt.xlabel('Mass (u)')
    plt.ylabel('Mean Intensity')
    plt.title(f'Mean Mass Spectrum - {polarity} Mode')
    plt.xlim(mean_spectrum.index.min(), min(mean_spectrum.index.max(), 200))  # Focus on lower masses
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'mean_spectrum_{polarity.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Execute Phase 1 analysis"""
    print("=== Phase 1: Catalogs & Enhanced QC ===")
    
    # Load datasets
    pos_df = pd.read_csv('out/all_positive_data_renamed.tsv', sep='\t')
    neg_df = pd.read_csv('out/all_negative_data_renamed.tsv', sep='\t')
    
    print(f"Loaded positive data: {pos_df.shape}")
    print(f"Loaded negative data: {neg_df.shape}")
    
    # Create output directories
    qc_dir = Path('qc')
    meta_dir = Path('meta')
    
    # 1. Generate Catalogs
    print("\n1. Generating Catalogs...")
    
    pos_catalog = parse_column_headers(pos_df, 'P')
    neg_catalog = parse_column_headers(neg_df, 'N')
    
    # Save catalogs
    pos_catalog.to_csv(meta_dir / 'pos_catalog.csv', index=False)
    neg_catalog.to_csv(meta_dir / 'neg_catalog.csv', index=False)
    
    print(f"Positive catalog: {len(pos_catalog)} samples")
    print(f"Negative catalog: {len(neg_catalog)} samples")
    print("\nFirst 10 rows of positive catalog:")
    print(pos_catalog.head(10))
    
    # 2. Enhanced QC Analysis
    print("\n2. Enhanced QC Analysis...")
    
    pos_qc = enhanced_qc_analysis(pos_df, 'Positive')
    neg_qc = enhanced_qc_analysis(neg_df, 'Negative')
    
    # Generate QC plots
    generate_qc_plots(pos_df, 'Positive', qc_dir)
    generate_qc_plots(neg_df, 'Negative', qc_dir)
    
    # 3. Generate Validation Summary
    print("\n3. Generating Validation Summary...")
    
    summary_lines = []
    summary_lines.append("=== ToF-SIMS Alucone Dose-Series QC Summary ===")
    summary_lines.append(f"Analysis Date: {pd.Timestamp.now()}")
    summary_lines.append("")
    
    for qc, label in [(pos_qc, "POSITIVE"), (neg_qc, "NEGATIVE")]:
        summary_lines.append(f"=== {label} ION MODE ===")
        summary_lines.append(f"Total samples: {qc['total_samples']}")
        summary_lines.append(f"Total masses: {qc['total_masses']}")
        summary_lines.append(f"Mass range: {qc['mass_range']['min']:.0f} - {qc['mass_range']['max']:.0f} u")
        summary_lines.append(f"Mass gaps (>1 u): {len(qc['mass_gaps'])}")
        summary_lines.append(f"Non-integer masses: {qc['non_integer_masses']}")
        summary_lines.append(f"Non-increasing masses: {qc['decreasing_masses']}")
        summary_lines.append("")
        
        summary_lines.append("Intensity Quality:")
        summary_lines.append(f"  Mean TIC: {qc['data_quality']['tic_mean']:.2e}")
        summary_lines.append(f"  TIC CV%: {qc['data_quality']['tic_cv']:.1f}%")
        summary_lines.append(f"  TIC Range: {qc['data_quality']['tic_range']['min']:.2e} - {qc['data_quality']['tic_range']['max']:.2e}")
        summary_lines.append(f"  Zero-sum samples: {qc['data_quality']['zero_sum_samples']}")
        summary_lines.append(f"  Samples with negatives: {qc['data_quality']['samples_with_negatives']}")
        summary_lines.append("")
        
        if qc['outliers']:
            summary_lines.append(f"Outlier Samples ({len(qc['outliers'])}):")
            for outlier in qc['outliers']:
                summary_lines.append(f"  {outlier['sample']}: {outlier['reason']} (z={outlier['z_score']:.2f})")
        else:
            summary_lines.append("No outlier samples detected.")
        summary_lines.append("")
    
    # Save validation summary
    with open(qc_dir / 'validation_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("\n=== Phase 1 Complete ===")
    print("Generated files:")
    print("  meta/pos_catalog.csv")
    print("  meta/neg_catalog.csv") 
    print("  qc/validation_summary.txt")
    print("  qc/tic_distribution_positive.png")
    print("  qc/tic_distribution_negative.png")
    print("  qc/intensity_heatmap_positive.png")
    print("  qc/intensity_heatmap_negative.png")
    print("  qc/mean_spectrum_positive.png")
    print("  qc/mean_spectrum_negative.png")
    
    return pos_qc, neg_qc

if __name__ == "__main__":
    main()