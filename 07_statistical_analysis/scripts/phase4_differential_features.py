#!/usr/bin/env python3
"""
Phase 4: Differential Features Analysis with Volcano Plots
- Compare extreme dose conditions: 15000 vs 500 (full range) and 10000 vs 2000 (mid-range)
- Statistical testing with FDR correction (Benjamini-Hochberg)
- Generate volcano plots with top differential masses labeled
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

def parse_sample_info(sample_name: str) -> tuple:
    """Extract pattern and dose from sample name"""
    import re
    match = re.match(r'P(\d+)_(\d+)μC-([PN])', sample_name)
    if match:
        pattern = f"P{match.group(1)}"
        dose = int(match.group(2))
        polarity = match.group(3)
        return pattern, dose, polarity
    return None, None, None

def differential_analysis(df: pd.DataFrame, dose1: int, dose2: int, polarity: str) -> pd.DataFrame:
    """Perform differential analysis between two dose conditions"""
    
    # Get sample columns and parse doses
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    
    # Separate samples by dose
    dose1_samples = []
    dose2_samples = []
    
    for col in sample_cols:
        _, dose, _ = parse_sample_info(col)
        if dose == dose1:
            dose1_samples.append(col)
        elif dose == dose2:
            dose2_samples.append(col)
    
    print(f"  Dose {dose1}: {len(dose1_samples)} samples")
    print(f"  Dose {dose2}: {len(dose2_samples)} samples")
    
    if len(dose1_samples) == 0 or len(dose2_samples) == 0:
        print(f"  Warning: Insufficient samples for comparison")
        return pd.DataFrame()
    
    # Prepare results
    results = []
    mass_index = df.set_index('Mass (u)')
    
    for mass in mass_index.index:
        # Get intensities for both groups
        group1_values = mass_index.loc[mass, dose1_samples].values
        group2_values = mass_index.loc[mass, dose2_samples].values
        
        # Remove any non-finite values
        group1_clean = group1_values[np.isfinite(group1_values)]
        group2_clean = group2_values[np.isfinite(group2_values)]
        
        if len(group1_clean) == 0 or len(group2_clean) == 0:
            continue
            
        # Calculate group statistics
        mean1 = np.mean(group1_clean)
        mean2 = np.mean(group2_clean)
        std1 = np.std(group1_clean, ddof=1) if len(group1_clean) > 1 else 0
        std2 = np.std(group2_clean, ddof=1) if len(group2_clean) > 1 else 0
        
        # Log2 fold change (dose2 vs dose1)
        # Add small pseudocount to avoid log(0)
        pseudocount = 1e-10
        log2fc = np.log2((mean2 + pseudocount) / (mean1 + pseudocount))
        
        # Welch's t-test (unequal variances)
        try:
            if len(group1_clean) > 1 and len(group2_clean) > 1:
                t_stat, p_value = ttest_ind(group1_clean, group2_clean, equal_var=False)
            else:
                t_stat, p_value = np.nan, 1.0
        except:
            t_stat, p_value = np.nan, 1.0
        
        results.append({
            'Mass (u)': mass,
            'log2FC': log2fc,
            'p_value': p_value,
            f'mean_{dose1}': mean1,
            f'mean_{dose2}': mean2,
            f'std_{dose1}': std1,
            f'std_{dose2}': std2,
            f'n_{dose1}': len(group1_clean),
            f'n_{dose2}': len(group2_clean),
            't_statistic': t_stat
        })
    
    results_df = pd.DataFrame(results)
    
    # FDR correction (Benjamini-Hochberg)
    if len(results_df) > 0:
        valid_p = ~np.isnan(results_df['p_value'])
        results_df.loc[:, 'q_value'] = np.nan
        
        if np.sum(valid_p) > 0:
            _, q_values, _, _ = multipletests(results_df.loc[valid_p, 'p_value'], 
                                           method='fdr_bh', alpha=0.05)
            results_df.loc[valid_p, 'q_value'] = q_values
        
        # Add significance flags
        results_df['significant'] = results_df['q_value'] < 0.05
    
    return results_df

def create_volcano_plot(diff_results: pd.DataFrame, dose1: int, dose2: int, 
                       polarity: str, method: str, output_path: Path):
    """Create volcano plot with top differential masses labeled"""
    
    if len(diff_results) == 0:
        print(f"  No data for volcano plot")
        return
    
    # Prepare data
    x = diff_results['log2FC'].values
    y = -np.log10(diff_results['q_value'].values + 1e-10)  # Add small value to avoid log(0)
    
    # Remove infinite values
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[finite_mask]
    y_clean = y[finite_mask]
    masses_clean = diff_results['Mass (u)'].values[finite_mask]
    significant = diff_results['significant'].values[finite_mask]
    
    if len(x_clean) == 0:
        print(f"  No finite data for volcano plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all points
    scatter = ax.scatter(x_clean, y_clean, 
                        c=['red' if sig else 'gray' for sig in significant],
                        alpha=0.6, s=20)
    
    # Add significance threshold line
    if np.any(significant):
        y_threshold = -np.log10(0.05)
        ax.axhline(y=y_threshold, color='blue', linestyle='--', alpha=0.5, label='q=0.05')
    
    # Add fold-change threshold lines (±1 log2FC = 2-fold change)
    ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=-1, color='blue', linestyle='--', alpha=0.5)
    
    # Label top masses by |log2FC|
    if len(x_clean) > 0:
        abs_fc = np.abs(x_clean)
        top_fc_indices = np.argsort(abs_fc)[-10:]  # Top 10 by fold change
        
        for idx in top_fc_indices:
            if significant[idx]:  # Only label significant masses
                ax.annotate(f'{int(masses_clean[idx])}', 
                           (x_clean[idx], y_clean[idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    # Label top masses by significance
    if np.any(significant):
        sig_indices = np.where(significant)[0]
        if len(sig_indices) > 0:
            sig_y = y_clean[sig_indices]
            top_sig_indices = sig_indices[np.argsort(sig_y)[-10:]]  # Top 10 by significance
            
            for idx in top_sig_indices:
                ax.annotate(f'{int(masses_clean[idx])}', 
                           (x_clean[idx], y_clean[idx]),
                           xytext=(-5, -5), textcoords='offset points',
                           fontsize=8, alpha=0.7, color='darkred')
    
    # Formatting
    ax.set_xlabel(f'log₂ FC (Dose {dose2} vs {dose1})')
    ax.set_ylabel('-log₁₀ q-value')
    ax.set_title(f'Differential Analysis: {dose2} vs {dose1} μC/cm²\n{polarity} Mode - {method}')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    n_significant = np.sum(significant)
    n_total = len(significant)
    stats_text = f'Significant: {n_significant}/{n_total} ({n_significant/n_total*100:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def interpret_differential_results(diff_results: pd.DataFrame, dose1: int, dose2: int, 
                                 polarity: str) -> dict:
    """Interpret differential results in context of alucone chemistry"""
    
    if len(diff_results) == 0:
        return {}
    
    significant = diff_results[diff_results['significant'] == True]
    
    if len(significant) == 0:
        return {'n_significant': 0, 'interpretations': []}
    
    interpretations = []
    
    # Define mass ranges for interpretation
    mass_families = {
        'low_hydrocarbons': (12, 30),      # C1-C2 fragments
        'mid_hydrocarbons': (31, 60),      # C2-C4 fragments  
        'aromatics': (65, 95),             # Benzene-related
        'metal_oxides': (25, 45),          # Al, AlO related
        'oxygenates': (28, 62),            # CHO, etc.
        'high_mass': (100, 200)            # Larger fragments
    }
    
    # Count changes by family
    family_changes = {}
    
    for family, (min_mass, max_mass) in mass_families.items():
        family_data = significant[
            (significant['Mass (u)'] >= min_mass) & 
            (significant['Mass (u)'] <= max_mass)
        ]
        
        if len(family_data) > 0:
            increasing = len(family_data[family_data['log2FC'] > 0])
            decreasing = len(family_data[family_data['log2FC'] < 0])
            
            family_changes[family] = {
                'total': len(family_data),
                'increasing': increasing,
                'decreasing': decreasing,
                'top_masses': family_data.nlargest(5, 'log2FC')['Mass (u)'].tolist()
            }
    
    # Generate interpretations based on alucone degradation expectations
    for family, changes in family_changes.items():
        if changes['total'] > 0:
            interpretation = f"{family}: {changes['total']} significant masses"
            if changes['increasing'] > changes['decreasing']:
                interpretation += f" (mostly ↑, top: {changes['top_masses'][:3]})"
            else:
                interpretation += f" (mostly ↓, top: {changes['top_masses'][:3]})"
            interpretations.append(interpretation)
    
    # Overall trends
    all_increasing = len(significant[significant['log2FC'] > 0])
    all_decreasing = len(significant[significant['log2FC'] < 0])
    
    if polarity == 'Positive':
        expected_interpretation = "Expected with dose ↑: ↓ oxygenates, ↑ hydrocarbons/aromatics, ↑ Al-O"
    else:
        expected_interpretation = "Expected with dose ↑: ↓ organics, ↑ Al-O condensation, ↑ oxides"
    
    results = {
        'n_significant': len(significant),
        'n_total': len(diff_results),
        'n_increasing': all_increasing,
        'n_decreasing': all_decreasing,
        'family_changes': family_changes,
        'interpretations': interpretations,
        'expected_pattern': expected_interpretation,
        'top_increasing': significant.nlargest(5, 'log2FC')[['Mass (u)', 'log2FC', 'q_value']].to_dict('records'),
        'top_decreasing': significant.nsmallest(5, 'log2FC')[['Mass (u)', 'log2FC', 'q_value']].to_dict('records')
    }
    
    return results

def main():
    """Execute Phase 4 differential analysis"""
    print("=== Phase 4: Differential Features Analysis ===")
    
    results_dir = Path('results')
    contrasts = [(15000, 500), (10000, 2000)]  # Full range and mid-range
    
    interpretation_summary = {}
    
    # Process both normalization methods for both polarities
    for polarity in ['positive', 'negative']:
        for method in ['baseline_TICsqrt', 'robust_PQNsqrtPareto']:
            print(f"\nProcessing {polarity} - {method}...")
            
            # Load normalized data
            data_path = results_dir / polarity / method / 'normalized_data.tsv'
            df = pd.read_csv(data_path, sep='\t')
            
            output_dir = results_dir / polarity / method
            pol_label = 'Positive' if polarity == 'positive' else 'Negative'
            
            method_results = {}
            
            # Analyze each contrast
            for dose2, dose1 in contrasts:
                contrast_name = f"{dose2}_vs_{dose1}"
                print(f"  Analyzing contrast: {dose2} vs {dose1}")
                
                # Perform differential analysis
                diff_results = differential_analysis(df, dose1, dose2, pol_label)
                
                if len(diff_results) > 0:
                    # Save results
                    output_file = output_dir / f'differential_{contrast_name}.tsv'
                    diff_results.to_csv(output_file, sep='\t', index=False)
                    
                    # Create volcano plot
                    plot_path = output_dir / f'volcano_{contrast_name}.png'
                    create_volcano_plot(diff_results, dose1, dose2, pol_label, method, plot_path)
                    
                    # Interpret results
                    interpretation = interpret_differential_results(diff_results, dose1, dose2, pol_label)
                    method_results[contrast_name] = interpretation
                    
                    print(f"    Significant masses: {interpretation.get('n_significant', 0)}/{interpretation.get('n_total', 0)}")
                    print(f"    Increasing: {interpretation.get('n_increasing', 0)}, Decreasing: {interpretation.get('n_decreasing', 0)}")
                else:
                    print(f"    No results for contrast {contrast_name}")
            
            interpretation_summary[f"{polarity}_{method}"] = method_results
    
    # Generate summary report
    print("\nGenerating differential analysis summary...")
    
    summary_lines = []
    summary_lines.append("=== Differential Features Summary ===")
    summary_lines.append(f"Analysis Date: {pd.Timestamp.now()}")
    summary_lines.append("")
    
    for key, method_results in interpretation_summary.items():
        polarity, method = key.split('_', 1)
        method = method.replace('_', '+')
        
        summary_lines.append(f"=== {polarity.upper()} - {method} ===")
        
        for contrast, results in method_results.items():
            dose2, dose1 = contrast.split('_vs_')
            summary_lines.append(f"\nContrast: {dose2} vs {dose1} μC/cm²")
            summary_lines.append(f"  Significant masses: {results.get('n_significant', 0)}/{results.get('n_total', 0)}")
            summary_lines.append(f"  Direction: ↑{results.get('n_increasing', 0)} ↓{results.get('n_decreasing', 0)}")
            
            summary_lines.append(f"  Expected: {results.get('expected_pattern', 'N/A')}")
            
            if results.get('interpretations'):
                summary_lines.append("  Family changes:")
                for interp in results['interpretations']:
                    summary_lines.append(f"    {interp}")
            
            # Top changes
            if results.get('top_increasing'):
                top_inc = results['top_increasing'][0]  # Top increasing
                summary_lines.append(f"  Top increasing: m/z {top_inc['Mass (u)']} (log2FC={top_inc['log2FC']:.2f}, q={top_inc['q_value']:.3f})")
            
            if results.get('top_decreasing'):
                top_dec = results['top_decreasing'][0]  # Top decreasing
                summary_lines.append(f"  Top decreasing: m/z {top_dec['Mass (u)']} (log2FC={top_dec['log2FC']:.2f}, q={top_dec['q_value']:.3f})")
        
        summary_lines.append("")
    
    # Save summary
    with open(Path('qc') / 'differential_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("\n=== Phase 4 Complete ===")
    print("Generated files for each polarity/method combination:")
    print("  - differential_15000_vs_500.tsv")
    print("  - differential_10000_vs_2000.tsv") 
    print("  - volcano_15000_vs_500.png")
    print("  - volcano_10000_vs_2000.png")
    print("  - qc/differential_summary.txt")

if __name__ == "__main__":
    main()