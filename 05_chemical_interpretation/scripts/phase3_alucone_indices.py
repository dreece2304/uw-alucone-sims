#!/usr/bin/env python3
"""
Phase 3: Alucone-Targeted Indices & Dose Trends
- Calculate chemical family indices relevant to alucone degradation
- Analyze dose-response trends with statistical validation  
- Identify threshold doses and monotonic relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

def calculate_alucone_indices(df: pd.DataFrame, polarity: str) -> pd.DataFrame:
    """Calculate alucone-specific chemical indices"""
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    indices_data = []
    
    # Set up mass index for faster lookup
    mass_index = df.set_index('Mass (u)')
    
    def safe_sum_masses(mass_list, sample_col):
        """Safely sum intensities for given masses"""
        total = 0.0
        for mass in mass_list:
            if mass in mass_index.index:
                value = mass_index.loc[mass, sample_col]
                if np.isfinite(value):
                    total += value
        return total
    
    for col in sample_cols:
        indices = {'Sample': col}
        
        if polarity == 'Positive':
            # Aromatic proxy index (aromatic-like / aliphatic-like)
            aromatic_masses = [65, 77, 91, 63, 79]  # Benzene ring fragments
            aliphatic_masses = [41, 43, 55, 57, 67, 69]  # Alkyl fragments
            
            aromatic_sum = safe_sum_masses(aromatic_masses, col)
            aliphatic_sum = safe_sum_masses(aliphatic_masses, col)
            
            indices['AI_sim'] = aromatic_sum / aliphatic_sum if aliphatic_sum > 0 else 0
            
            # Oxygenation index
            oxygenated_masses = [31, 45, 59, 60, 28, 44]  # CHO+, C2H5O+, etc.
            oxygenated_sum = safe_sum_masses(oxygenated_masses, col)
            indices['OI'] = oxygenated_sum / aliphatic_sum if aliphatic_sum > 0 else 0
            
            # Al-O(+aromatic) index  
            al_o_masses = [27, 43, 59]  # Al+, AlO+, related
            indices['AOI_pos'] = safe_sum_masses(al_o_masses, col)
            
            # Metal oxide index (enhanced)
            metal_oxide_masses = [27, 43]  # Al+, AlO+
            indices['Metal_Oxide'] = safe_sum_masses(metal_oxide_masses, col)
            
            # Hydrocarbon degradation index
            hc_degrad_masses = [29, 41, 43, 55]  # Alkyl fragments
            all_organic_masses = list(range(12, 100))  # Approximate organic range
            hc_sum = safe_sum_masses(hc_degrad_masses, col)
            organic_sum = safe_sum_masses(all_organic_masses, col)
            indices['HC_Degrad'] = hc_sum / organic_sum if organic_sum > 0 else 0
            
            # Chain scission proxy
            chain_scission_masses = [15, 29, 43]  # CH3+, C2H5+, C3H7+
            indices['Chain_Scission'] = safe_sum_masses(chain_scission_masses, col)
            
        else:  # Negative mode
            # Oxygenation index (negative)
            oxygenated_masses = [31, 45, 59, 60, 28, 44]  # CHO-, etc.
            indices['OI'] = safe_sum_masses(oxygenated_masses, col)
            
            # Al-O index (negative)
            al_o_masses = [43, 59]  # AlO-, related
            indices['AOI_neg'] = safe_sum_masses(al_o_masses, col)
            
            # H-loss index (dehydrogenation)
            h_loss_masses = [16, 17, 18]  # O-, OH-, H2O-
            indices['H_Loss'] = safe_sum_masses(h_loss_masses, col)
            
            # Oxide formation index
            oxide_masses = [16, 17]  # O-, OH-
            indices['Oxide_Form'] = safe_sum_masses(oxide_masses, col)
            
            # Aluminum-carbon bond proxy (approximate)
            al_c_masses = [39, 55, 71]  # Approximate Al-C fragments
            indices['Al_C_Bond'] = safe_sum_masses(al_c_masses, col)
            
            # Carbonyl/carboxyl index
            carbonyl_masses = [44, 45, 60]  # CO2-, CHO2-, etc.
            indices['Carbonyl'] = safe_sum_masses(carbonyl_masses, col)
        
        indices_data.append(indices)
    
    return pd.DataFrame(indices_data)

def parse_sample_info(sample_name: str) -> tuple:
    """Extract pattern and dose from sample name"""
    # Format: P1_2000μC-P
    import re
    match = re.match(r'P(\d+)_(\d+)μC-([PN])', sample_name)
    if match:
        pattern = f"P{match.group(1)}"
        dose = int(match.group(2))
        return pattern, dose
    return None, None

def analyze_dose_trends(indices_df: pd.DataFrame, polarity: str) -> dict:
    """Analyze dose-response trends with statistical validation"""
    
    # Parse sample info
    indices_df['Pattern'] = indices_df['Sample'].apply(lambda x: parse_sample_info(x)[0])
    indices_df['Dose'] = indices_df['Sample'].apply(lambda x: parse_sample_info(x)[1])
    
    # Get index columns (exclude metadata)
    index_cols = [col for col in indices_df.columns if col not in ['Sample', 'Pattern', 'Dose']]
    
    trend_results = {}
    
    # Aggregate by dose (mean across patterns)
    dose_aggregated = indices_df.groupby('Dose')[index_cols].agg(['mean', 'std', 'count']).reset_index()
    dose_aggregated.columns = ['Dose'] + [f'{col}_{stat}' for col, stat in dose_aggregated.columns[1:]]
    
    # Calculate RSD% (relative standard deviation)
    rsd_data = []
    for dose in indices_df['Dose'].unique():
        dose_data = indices_df[indices_df['Dose'] == dose]
        rsd_row = {'Dose': dose}
        
        for col in index_cols:
            values = dose_data[col].values
            mean_val = np.mean(values)
            std_val = np.std(values)
            rsd_pct = (std_val / mean_val * 100) if mean_val != 0 else np.inf
            rsd_row[f'{col}_RSD%'] = rsd_pct
        
        rsd_data.append(rsd_row)
    
    rsd_df = pd.DataFrame(rsd_data)
    
    # Monotonicity analysis (Spearman correlation with dose)
    monotonicity_results = {}
    
    # Use per-dose means to avoid pseudo-replication
    doses = dose_aggregated['Dose'].values
    
    for col in index_cols:
        dose_means = dose_aggregated[f'{col}_mean'].values
        
        if len(doses) > 2:  # Need at least 3 points for meaningful correlation
            rho, p_value = stats.spearmanr(doses, dose_means)
            
            monotonicity_results[col] = {
                'spearman_rho': rho,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'trend_direction': 'increasing' if rho > 0 else 'decreasing' if rho < 0 else 'none'
            }
            
            # Robust linear slope with 95% CI (effect size)
            if not np.any(np.isnan(dose_means)) and len(doses) > 2:
                try:
                    # Use log10(dose) for better fitting
                    log_doses = np.log10(doses)
                    slope, intercept, r_val, p_val, std_err = stats.linregress(log_doses, dose_means)
                    
                    # 95% CI for slope
                    t_val = stats.t.ppf(0.975, len(doses) - 2)  # 95% CI
                    slope_ci = t_val * std_err
                    
                    monotonicity_results[col]['linear_slope'] = slope
                    monotonicity_results[col]['slope_95ci'] = [slope - slope_ci, slope + slope_ci]
                    monotonicity_results[col]['r_squared'] = r_val**2
                except:
                    monotonicity_results[col]['linear_slope'] = np.nan
                    monotonicity_results[col]['slope_95ci'] = [np.nan, np.nan]
                    monotonicity_results[col]['r_squared'] = np.nan
        else:
            monotonicity_results[col] = {
                'spearman_rho': np.nan,
                'p_value': np.nan,
                'significant': False,
                'trend_direction': 'insufficient_data'
            }
    
    # Threshold detection (change-point analysis)
    threshold_results = {}
    
    for col in index_cols:
        dose_means = dose_aggregated[f'{col}_mean'].values
        
        if len(doses) >= 4:  # Need enough points for change-point detection
            # Simple change-point: test each potential breakpoint
            best_breakpoint = None
            best_improvement = 0
            
            for i in range(1, len(doses) - 1):  # Test breakpoints
                # Split data at breakpoint
                x1, y1 = doses[:i+1], dose_means[:i+1]
                x2, y2 = doses[i:], dose_means[i:]
                
                # Fit linear models to each segment
                if len(x1) >= 2 and len(x2) >= 2:
                    try:
                        _, _, r1, _, _ = stats.linregress(x1, y1)
                        _, _, r2, _, _ = stats.linregress(x2, y2)
                        
                        # Combined R² as improvement metric
                        combined_r2 = (len(x1) * r1**2 + len(x2) * r2**2) / len(doses)
                        
                        # Compare to single linear fit
                        _, _, r_full, _, _ = stats.linregress(doses, dose_means)
                        improvement = combined_r2 - r_full**2
                        
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_breakpoint = doses[i]
                    except:
                        continue
            
            threshold_results[col] = {
                'breakpoint_dose': best_breakpoint,
                'r2_improvement': best_improvement,
                'has_threshold': best_improvement > 0.1  # Arbitrary threshold
            }
        else:
            threshold_results[col] = {
                'breakpoint_dose': None,
                'r2_improvement': 0,
                'has_threshold': False
            }
    
    trend_results = {
        'polarity': polarity,
        'dose_aggregated': dose_aggregated,
        'rsd_summary': rsd_df,
        'monotonicity': monotonicity_results,
        'thresholds': threshold_results,
        'raw_indices': indices_df
    }
    
    return trend_results

def generate_trend_plots(trend_results: dict, output_dir: Path):
    """Generate dose-trend visualization plots"""
    polarity = trend_results['polarity']
    dose_agg = trend_results['dose_aggregated']
    monotonicity = trend_results['monotonicity']
    indices_df = trend_results['raw_indices']
    
    # Get index columns
    index_cols = [col.replace('_mean', '') for col in dose_agg.columns if col.endswith('_mean')]
    
    # 1. Dose-response curves with error bars
    n_indices = len(index_cols)
    n_cols = min(3, n_indices)
    n_rows = int(np.ceil(n_indices / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(index_cols):
        ax = axes[i] if i < len(axes) else plt.subplot(n_rows, n_cols, i+1)
        
        doses = dose_agg['Dose'].values
        means = dose_agg[f'{col}_mean'].values
        stds = dose_agg[f'{col}_std'].values
        
        # Individual points
        individual_data = indices_df.groupby('Dose')[col].apply(list)
        for dose, values in individual_data.items():
            ax.scatter([dose] * len(values), values, alpha=0.6, s=30, color='lightblue')
        
        # Mean with error bars
        ax.errorbar(doses, means, yerr=stds, fmt='o-', capsize=5, capthick=2, 
                   color='darkblue', linewidth=2, markersize=8)
        
        ax.set_xlabel('Dose (μC/cm²)')
        ax.set_ylabel(f'{col} Index')
        
        # Add trend statistics
        mono_stats = monotonicity.get(col, {})
        rho = mono_stats.get('spearman_rho', np.nan)
        p_val = mono_stats.get('p_value', np.nan)
        
        title = f'{col}\nρ={rho:.3f}, p={p_val:.3f}' if not np.isnan(rho) else f'{col}'
        if mono_stats.get('significant', False):
            title += '*'
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for i in range(n_indices, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'dose_response_curves_{polarity.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monotonicity summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spearman correlations
    indices_names = list(monotonicity.keys())
    rho_values = [monotonicity[idx]['spearman_rho'] for idx in indices_names]
    p_values = [monotonicity[idx]['p_value'] for idx in indices_names]
    
    colors = ['red' if p < 0.05 else 'gray' for p in p_values]
    bars = ax1.barh(indices_names, rho_values, color=colors)
    ax1.set_xlabel('Spearman ρ')
    ax1.set_title(f'Dose Monotonicity - {polarity}')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Add significance markers
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        if p_val < 0.05:
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                    '*', ha='left', va='center', fontsize=14, fontweight='bold')
    
    # RSD% summary
    rsd_df = trend_results['rsd_summary']
    rsd_cols = [col for col in rsd_df.columns if col.endswith('_RSD%')]
    
    if rsd_cols:
        rsd_matrix = rsd_df[rsd_cols].values
        rsd_names = [col.replace('_RSD%', '') for col in rsd_cols]
        
        im = ax2.imshow(rsd_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=50)
        ax2.set_xticks(range(len(rsd_names)))
        ax2.set_xticklabels(rsd_names, rotation=45, ha='right')
        ax2.set_yticks(range(len(rsd_df)))
        ax2.set_yticklabels([f"{dose}" for dose in rsd_df['Dose']])
        ax2.set_ylabel('Dose (μC/cm²)')
        ax2.set_title(f'Repeatability (RSD%) - {polarity}')
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='RSD%')
        
        # Add text annotations
        for i in range(len(rsd_df)):
            for j in range(len(rsd_names)):
                value = rsd_matrix[i, j]
                if np.isfinite(value):
                    text_color = 'white' if value > 25 else 'black'
                    ax2.text(j, i, f'{value:.1f}', ha='center', va='center', 
                            color=text_color, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'monotonicity_summary_{polarity.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Execute Phase 3 analysis"""
    print("=== Phase 3: Alucone-Targeted Indices & Dose Trends ===")
    
    results_dir = Path('results')
    
    # Process both normalization methods for both polarities
    for polarity in ['positive', 'negative']:
        for method in ['baseline_TICsqrt', 'robust_PQNsqrtPareto']:
            print(f"\nProcessing {polarity} - {method}...")
            
            # Load normalized data
            data_path = results_dir / polarity / method / 'normalized_data.tsv'
            df = pd.read_csv(data_path, sep='\t')
            
            # Calculate indices
            print("  Calculating alucone indices...")
            pol_label = 'Positive' if polarity == 'positive' else 'Negative'
            indices_df = calculate_alucone_indices(df, pol_label)
            
            # Analyze trends
            print("  Analyzing dose trends...")
            trend_results = analyze_dose_trends(indices_df, pol_label)
            
            # Save results
            output_dir = results_dir / polarity / method
            
            # Save indices per sample
            indices_df.to_csv(output_dir / 'alucone_indices_per_sample.csv', index=False)
            
            # Save dose aggregated data
            trend_results['dose_aggregated'].to_csv(output_dir / 'dose_aggregated_indices.csv', index=False)
            
            # Save RSD summary
            trend_results['rsd_summary'].to_csv(output_dir / 'rsd_summary.csv', index=False)
            
            # Save monotonicity results
            mono_df = pd.DataFrame(trend_results['monotonicity']).T
            mono_df.to_csv(output_dir / 'monotonicity_analysis.csv')
            
            # Save threshold analysis
            thresh_df = pd.DataFrame(trend_results['thresholds']).T
            thresh_df.to_csv(output_dir / 'threshold_analysis.csv')
            
            # Generate plots
            print("  Generating trend plots...")
            generate_trend_plots(trend_results, output_dir)
            
            # Report key findings
            significant_trends = [idx for idx, stats in trend_results['monotonicity'].items() 
                                if stats.get('significant', False)]
            print(f"  Significant dose trends ({len(significant_trends)}): {significant_trends}")
    
    print("\n=== Phase 3 Complete ===")
    print("Generated files for each polarity/method combination:")
    print("  - alucone_indices_per_sample.csv")  
    print("  - dose_aggregated_indices.csv")
    print("  - rsd_summary.csv")
    print("  - monotonicity_analysis.csv")
    print("  - threshold_analysis.csv")
    print("  - dose_response_curves_[polarity].png")
    print("  - monotonicity_summary_[polarity].png")

if __name__ == "__main__":
    main()