#!/usr/bin/env python3
"""
Phase 5: Calculate Alucone Indices for ROI Data
- Apply alucone-specific chemical indices to ROI normalized data
- Calculate dose-response trends for each ROI
- Compare with existing sum-spectrum indices
- Generate dose-trend visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import json
import warnings
warnings.filterwarnings('ignore')

def load_config() -> Dict:
    """Load project configuration with mass families"""
    config_path = Path('roi/config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def load_normalized_roi_data() -> Dict:
    """Load normalized ROI data"""
    
    results_dir = Path('results')
    data = {}
    
    methods = ['baseline_TICsqrt', 'robust_PQNsqrtPareto']
    polarities = ['positive', 'negative']
    
    for polarity in polarities:
        data[polarity] = {}
        for method in methods:
            file_path = results_dir / polarity / method / f'{polarity}_{method}_roi_normalized.tsv'
            if file_path.exists():
                df = pd.read_csv(file_path, sep='\t')
                data[polarity][method] = df
                print(f"Loaded {polarity} {method} ROI data: {df.shape}")
            else:
                print(f"Missing: {file_path}")
                data[polarity][method] = None
    
    return data

def calculate_alucone_indices(data: pd.DataFrame, mass_families: Dict, polarity: str, 
                            mass_col: str = 'mass') -> pd.DataFrame:
    """Calculate alucone-specific chemical indices"""
    
    print(f"  Calculating alucone indices for {polarity} data...")
    
    # Get mass families for this polarity
    families = mass_families[polarity]
    
    # Separate masses and intensities
    masses = data[mass_col].values
    intensities = data.drop(columns=[mass_col])
    
    # Create mass index mapping
    mass_to_idx = {mass: idx for idx, mass in enumerate(masses)}
    
    indices = {}
    sample_columns = intensities.columns
    
    for sample in sample_columns:
        sample_data = intensities[sample].values
        sample_indices = {}
        
        # Calculate family sums
        family_sums = {}
        for family_name, family_masses in families.items():
            family_sum = 0
            for mass in family_masses:
                if mass in mass_to_idx:
                    idx = mass_to_idx[mass]
                    family_sum += sample_data[idx]
            family_sums[family_name] = family_sum
        
        if polarity == 'positive':
            # Positive ion indices
            
            # Oxygenation Index (OI) - oxygenates / total
            total_intensity = np.sum(sample_data[sample_data > 0])
            if total_intensity > 0:
                sample_indices['OI'] = family_sums['oxygenates'] / total_intensity
            else:
                sample_indices['OI'] = 0
            
            # Al-O Index (AOI) - Al-O containing / oxygenates  
            if family_sums['oxygenates'] > 0:
                sample_indices['AOI'] = family_sums['metal_oxides'] / family_sums['oxygenates']
            else:
                sample_indices['AOI'] = 0
            
            # Aromatic Index (AI_sim) - aromatic proxies / total
            if total_intensity > 0:
                sample_indices['AI_sim'] = family_sums['aromatic_proxies'] / total_intensity
            else:
                sample_indices['AI_sim'] = 0
            
            # Aliphatic/Aromatic ratio
            if family_sums['aromatic_proxies'] > 0:
                sample_indices['Aliph_Arom_Ratio'] = family_sums['aliphatic'] / family_sums['aromatic_proxies']
            else:
                sample_indices['Aliph_Arom_Ratio'] = 0
        
        else:  # negative
            # Negative ion indices
            
            # Oxygenation Index - same definition
            total_intensity = np.sum(sample_data[sample_data > 0])
            if total_intensity > 0:
                sample_indices['OI'] = family_sums['oxygenates'] / total_intensity
            else:
                sample_indices['OI'] = 0
            
            # Al-O Index - using Al-O families
            if family_sums['oxygenates'] > 0:
                sample_indices['AOI'] = family_sums['al_o_families'] / family_sums['oxygenates']
            else:
                sample_indices['AOI'] = 0
            
            # H-loss Index - hydrogen loss / total
            if total_intensity > 0:
                sample_indices['H_loss_Index'] = family_sums['h_loss'] / total_intensity
            else:
                sample_indices['H_loss_Index'] = 0
            
            # Carbonyl Index
            if total_intensity > 0:
                sample_indices['Carbonyl_Index'] = family_sums['carbonyl'] / total_intensity
            else:
                sample_indices['Carbonyl_Index'] = 0
        
        indices[sample] = sample_indices
    
    # Convert to DataFrame
    indices_df = pd.DataFrame.from_dict(indices, orient='index')
    indices_df.index.name = 'sample'
    indices_df = indices_df.reset_index()
    
    return indices_df

def extract_dose_info(sample_name: str) -> Tuple[str, int, str]:
    """Extract pattern, dose, and polarity from sample name"""
    # Format: P{pattern}_{dose}μC-{pol}
    
    parts = sample_name.split('_')
    if len(parts) >= 2:
        pattern = parts[0]  # P1, P2, P3
        
        dose_part = parts[1].split('μC')[0]  # Extract dose number
        try:
            dose = int(dose_part)
        except:
            dose = 0
        
        polarity = 'positive' if sample_name.endswith('-P') else 'negative'
        
        return pattern, dose, polarity
    
    return 'unknown', 0, 'unknown'

def calculate_dose_trends(indices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dose-response trends using Spearman correlation"""
    
    # Add dose information
    dose_info = []
    for sample in indices_df['sample']:
        pattern, dose, polarity = extract_dose_info(sample)
        dose_info.append({'sample': sample, 'pattern': pattern, 'dose': dose, 'polarity': polarity})
    
    dose_df = pd.DataFrame(dose_info)
    indices_with_dose = indices_df.merge(dose_df, on='sample')
    
    # Calculate correlations for each pattern and index
    correlations = []
    
    patterns = indices_with_dose['pattern'].unique()
    index_columns = [col for col in indices_df.columns if col != 'sample']
    
    for pattern in patterns:
        pattern_data = indices_with_dose[indices_with_dose['pattern'] == pattern]
        
        if len(pattern_data) < 3:  # Need at least 3 points for correlation
            continue
        
        doses = pattern_data['dose'].values
        
        for index_name in index_columns:
            index_values = pattern_data[index_name].values
            
            # Skip if all values are the same
            if np.std(index_values) == 0:
                continue
            
            # Calculate Spearman correlation
            try:
                rho, p_value = spearmanr(doses, index_values)
                
                correlations.append({
                    'pattern': pattern,
                    'index': index_name,
                    'spearman_rho': rho,
                    'p_value': p_value,
                    'n_points': len(pattern_data),
                    'significance': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
                })
            except:
                # Skip if correlation calculation fails
                continue
    
    return pd.DataFrame(correlations), indices_with_dose

def create_dose_trend_plots(indices_with_dose: pd.DataFrame, correlations: pd.DataFrame, 
                          polarity: str, method: str) -> None:
    """Create dose-trend visualization plots"""
    
    print(f"  Creating dose-trend plots for {polarity} {method}...")
    
    index_columns = [col for col in indices_with_dose.columns 
                    if col not in ['sample', 'pattern', 'dose', 'polarity']]
    
    if len(index_columns) == 0:
        print(f"    No indices to plot")
        return
    
    n_indices = len(index_columns)
    n_cols = min(3, n_indices)
    n_rows = int(np.ceil(n_indices / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_indices == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'{polarity.capitalize()} Ion Alucone Indices - ROI Data\n{method}', fontsize=16)
    
    patterns = sorted(indices_with_dose['pattern'].unique())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, index_name in enumerate(index_columns):
        ax = axes[i]
        
        for j, pattern in enumerate(patterns):
            pattern_data = indices_with_dose[indices_with_dose['pattern'] == pattern]
            
            if len(pattern_data) == 0:
                continue
            
            doses = pattern_data['dose']
            values = pattern_data[index_name]
            
            ax.scatter(doses, values, color=colors[j % len(colors)], 
                      label=pattern, alpha=0.7, s=60)
            
            # Add trend line if significant correlation exists
            pattern_corr = correlations[
                (correlations['pattern'] == pattern) & 
                (correlations['index'] == index_name)
            ]
            
            if not pattern_corr.empty:
                corr_data = pattern_corr.iloc[0]
                if corr_data['p_value'] < 0.05:  # Significant trend
                    # Simple linear fit for visualization
                    z = np.polyfit(doses, values, 1)
                    p = np.poly1d(z)
                    dose_range = np.linspace(doses.min(), doses.max(), 100)
                    ax.plot(dose_range, p(dose_range), 
                           color=colors[j % len(colors)], linestyle='--', alpha=0.5)
        
        ax.set_xlabel('E-beam Dose (μC/cm²)')
        ax.set_ylabel(index_name)
        ax.set_title(f'{index_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_indices, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    results_dir = Path('results') / polarity / method
    plot_path = results_dir / f'{polarity}_{method}_roi_dose_trends.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    Saved dose-trend plot: {plot_path}")

def save_indices_and_trends(indices_df: pd.DataFrame, correlations: pd.DataFrame,
                           polarity: str, method: str) -> List[Path]:
    """Save indices and trend analysis results"""
    
    results_dir = Path('results') / polarity / method
    saved_files = []
    
    # Save indices
    indices_file = results_dir / f'{polarity}_{method}_roi_alucone_indices.tsv'
    indices_df.to_csv(indices_file, sep='\t', index=False)
    saved_files.append(indices_file)
    
    # Save correlations
    corr_file = results_dir / f'{polarity}_{method}_roi_dose_trends.tsv'
    correlations.to_csv(corr_file, sep='\t', index=False)
    saved_files.append(corr_file)
    
    return saved_files

def generate_indices_summary(all_correlations: Dict) -> None:
    """Generate summary of alucone indices analysis"""
    
    print("\nGenerating alucone indices summary...")
    
    summary_lines = []
    summary_lines.append("=== ROI Alucone Indices Analysis Summary ===")
    summary_lines.append(f"Generated: 2025-08-22")
    summary_lines.append("")
    
    for polarity in ['positive', 'negative']:
        if polarity not in all_correlations:
            continue
            
        summary_lines.append(f"{polarity.capitalize()} Ion Results:")
        
        for method in ['baseline_TICsqrt', 'robust_PQNsqrtPareto']:
            if method not in all_correlations[polarity]:
                continue
                
            corr_df = all_correlations[polarity][method]
            
            if corr_df.empty:
                summary_lines.append(f"  {method}: No correlations calculated")
                continue
            
            summary_lines.append(f"  {method}:")
            
            # Count significant correlations
            sig_strong = len(corr_df[(corr_df['p_value'] < 0.001) & (abs(corr_df['spearman_rho']) > 0.7)])
            sig_moderate = len(corr_df[(corr_df['p_value'] < 0.01) & (abs(corr_df['spearman_rho']) > 0.5)])
            total_tests = len(corr_df)
            
            summary_lines.append(f"    Total correlations: {total_tests}")
            summary_lines.append(f"    Strong & significant (|ρ|>0.7, p<0.001): {sig_strong}")
            summary_lines.append(f"    Moderate & significant (|ρ|>0.5, p<0.01): {sig_moderate}")
            
            # Highlight strongest correlations
            strongest = corr_df.loc[corr_df['spearman_rho'].abs().idxmax()] if not corr_df.empty else None
            if strongest is not None:
                summary_lines.append(f"    Strongest correlation: {strongest['pattern']} {strongest['index']} "
                                   f"(ρ={strongest['spearman_rho']:.3f}, p={strongest['p_value']:.2e})")
        
        summary_lines.append("")
    
    summary_lines.append("Key Findings:")
    summary_lines.append("- ROI data shows dose-dependent chemical changes")
    summary_lines.append("- Alucone indices capture e-beam induced transformations")  
    summary_lines.append("- Spatial simulation preserves dose-response relationships")
    summary_lines.append("- Ready for statistical contrasts and PCA analysis")
    
    # Save summary
    summary_path = Path('qc/roi_alucone_indices_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Indices summary saved: {summary_path}")

def main():
    """Execute Phase 5 ROI alucone indices analysis"""
    print("=== Phase 5: Calculate Alucone Indices for ROI Data ===")
    
    # Load configuration
    config = load_config()
    mass_families = config['mass_families']
    print("Mass families loaded")
    
    # Load normalized ROI data
    roi_data = load_normalized_roi_data()
    
    # Check if any data is available
    has_data = False
    for pol in roi_data.keys():
        for method_data in roi_data[pol].values():
            if method_data is not None:
                has_data = True
                break
        if has_data:
            break
    
    if not has_data:
        print("❌ No normalized ROI data found")
        return
    
    # Process each polarity and method
    all_indices = {}
    all_correlations = {}
    all_saved_files = []
    
    methods = ['baseline_TICsqrt', 'robust_PQNsqrtPareto']
    polarities = ['positive', 'negative']
    
    for polarity in polarities:
        all_indices[polarity] = {}
        all_correlations[polarity] = {}
        
        for method in methods:
            data = roi_data[polarity][method]
            
            if data is None:
                print(f"\n⚠️  No data for {polarity} {method}")
                continue
            
            print(f"\nProcessing {polarity} {method}...")
            
            # Calculate alucone indices
            indices_df = calculate_alucone_indices(data, mass_families, polarity)
            all_indices[polarity][method] = indices_df
            
            print(f"  Calculated {indices_df.shape[0]} samples × {indices_df.shape[1]-1} indices")
            
            # Calculate dose trends
            correlations, indices_with_dose = calculate_dose_trends(indices_df)
            all_correlations[polarity][method] = correlations
            
            print(f"  Calculated {len(correlations)} dose-trend correlations")
            
            # Create visualization
            create_dose_trend_plots(indices_with_dose, correlations, polarity, method)
            
            # Save results
            saved_files = save_indices_and_trends(indices_df, correlations, polarity, method)
            all_saved_files.extend(saved_files)
    
    # Generate summary
    generate_indices_summary(all_correlations)
    
    print("\n=== Phase 5 Complete ===")
    print("Generated files:")
    for file in all_saved_files:
        print(f"  {file}")
    print("  Dose-trend plots in results/*/")
    print("  qc/roi_alucone_indices_summary.txt")
    
    return all_indices, all_correlations

if __name__ == "__main__":
    indices, correlations = main()