#!/usr/bin/env python3
"""
Phase 2: Dual Normalization (Baseline & Robust)
- Baseline: TIC normalization → √ transform (PNNL-style)  
- Robust alternative: PQN normalization → √ → Pareto scaling
- Diagnostic comparisons between methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def tic_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """TIC normalization followed by square root transform (PNNL baseline method)"""
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    normalized_df = df.copy()
    
    for col in sample_cols:
        intensities = df[col].values
        tic = np.sum(intensities)
        
        if tic > 0:
            # TIC normalize then square root
            normalized_intensities = intensities / tic
            sqrt_normalized = np.sqrt(normalized_intensities)
            normalized_df[col] = sqrt_normalized
        else:
            normalized_df[col] = intensities  # Keep zeros as zeros
    
    return normalized_df

def pqn_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Probabilistic Quotient Normalization (PQN)"""
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    
    # Calculate reference spectrum (median across all samples per mass)
    intensity_matrix = df.set_index('Mass (u)')[sample_cols].values
    reference_spectrum = np.median(intensity_matrix, axis=1)
    
    # Calculate normalization factors for each sample
    pqn_factors = {}
    normalized_matrix = np.zeros_like(intensity_matrix)
    
    for i, col in enumerate(sample_cols):
        sample_spectrum = intensity_matrix[:, i]
        
        # Calculate quotients (avoid division by zero)
        quotients = []
        for j in range(len(sample_spectrum)):
            if reference_spectrum[j] > 0 and sample_spectrum[j] > 0:
                quotients.append(sample_spectrum[j] / reference_spectrum[j])
        
        if quotients:
            # PQN factor is median of quotients
            pqn_factor = np.median(quotients)
            pqn_factors[col] = pqn_factor
            
            # Apply normalization
            normalized_matrix[:, i] = sample_spectrum / pqn_factor
        else:
            pqn_factors[col] = 1.0
            normalized_matrix[:, i] = sample_spectrum
    
    # Create normalized dataframe
    normalized_df = df.copy()
    for i, col in enumerate(sample_cols):
        normalized_df[col] = normalized_matrix[:, i]
    
    return normalized_df, pqn_factors

def pareto_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto scaling (mean center and divide by sqrt(std))"""
    sample_cols = [col for col in df.columns if col != 'Mass (u)']
    intensity_matrix = df.set_index('Mass (u)')[sample_cols]
    
    # Calculate mean and std for each mass
    means = intensity_matrix.mean(axis=1)
    stds = intensity_matrix.std(axis=1)
    
    # Pareto scaling: (x - mean) / sqrt(std)
    scaled_matrix = intensity_matrix.copy()
    for mass in intensity_matrix.index:
        if stds[mass] > 0:
            scaled_matrix.loc[mass] = (intensity_matrix.loc[mass] - means[mass]) / np.sqrt(stds[mass])
        else:
            scaled_matrix.loc[mass] = intensity_matrix.loc[mass] - means[mass]
    
    # Reconstruct dataframe
    scaled_df = df.copy()
    for col in sample_cols:
        scaled_df[col] = scaled_matrix[col].values
    
    return scaled_df

def robust_normalization_pipeline(df: pd.DataFrame) -> tuple:
    """Complete robust normalization: PQN → √ → Pareto"""
    # Step 1: PQN normalization
    pqn_df, pqn_factors = pqn_normalization(df)
    
    # Step 2: Square root transform
    sample_cols = [col for col in pqn_df.columns if col != 'Mass (u)']
    sqrt_df = pqn_df.copy()
    for col in sample_cols:
        sqrt_df[col] = np.sqrt(np.maximum(pqn_df[col], 0))  # Ensure non-negative before sqrt
    
    # Step 3: Pareto scaling
    robust_df = pareto_scaling(sqrt_df)
    
    return robust_df, pqn_factors

def compare_normalizations(baseline_df: pd.DataFrame, robust_df: pd.DataFrame, 
                         polarity: str, output_dir: Path) -> dict:
    """Compare baseline vs robust normalization methods"""
    sample_cols = [col for col in baseline_df.columns if col != 'Mass (u)']
    
    # Get intensity matrices
    baseline_matrix = baseline_df.set_index('Mass (u)')[sample_cols]
    robust_matrix = robust_df.set_index('Mass (u)')[sample_cols]
    
    # 1. Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Baseline distributions
    baseline_flat = baseline_matrix.values.flatten()
    baseline_flat = baseline_flat[baseline_flat > 0]  # Remove zeros for log scale
    
    axes[0, 0].hist(baseline_flat, bins=50, alpha=0.7, density=True, color='blue')
    axes[0, 0].set_yscale('log')
    axes[0, 0].set_title(f'Baseline (TIC+√) - {polarity}')
    axes[0, 0].set_xlabel('Normalized Intensity')
    
    # Robust distributions  
    robust_flat = robust_matrix.values.flatten()
    robust_flat = robust_flat[np.isfinite(robust_flat)]  # Remove inf/nan
    
    axes[0, 1].hist(robust_flat, bins=50, alpha=0.7, density=True, color='red')
    axes[0, 1].set_title(f'Robust (PQN+√+Pareto) - {polarity}')
    axes[0, 1].set_xlabel('Normalized Intensity')
    
    # Q-Q plots
    if len(baseline_flat) > 0 and len(robust_flat) > 0:
        stats.probplot(baseline_flat, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title(f'Q-Q Plot: Baseline - {polarity}')
        
        stats.probplot(robust_flat, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q Plot: Robust - {polarity}')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'normalization_comparison_{polarity.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation between methods
    # Sample-wise correlations
    sample_correlations = []
    for col in sample_cols:
        baseline_sample = baseline_matrix[col].values
        robust_sample = robust_matrix[col].values
        
        # Remove inf/nan for correlation
        valid_mask = np.isfinite(baseline_sample) & np.isfinite(robust_sample)
        if np.sum(valid_mask) > 10:  # Need enough points for correlation
            corr, _ = stats.pearsonr(baseline_sample[valid_mask], robust_sample[valid_mask])
            sample_correlations.append(corr)
    
    # Overall correlation heatmap (subset of masses for visualization)
    n_masses_plot = min(100, len(baseline_matrix))
    indices_plot = np.linspace(0, len(baseline_matrix)-1, n_masses_plot, dtype=int)
    
    baseline_subset = baseline_matrix.iloc[indices_plot]
    robust_subset = robust_matrix.iloc[indices_plot]
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = np.corrcoef(baseline_subset.T, robust_subset.T)
    n_samples = len(sample_cols)
    cross_correlation = correlation_matrix[:n_samples, n_samples:]
    
    sns.heatmap(cross_correlation, 
                xticklabels=[f'R_{col}' for col in sample_cols],
                yticklabels=[f'B_{col}' for col in sample_cols],
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'})
    plt.title(f'Cross-Method Correlation Matrix - {polarity}')
    plt.xlabel('Robust Method Samples')
    plt.ylabel('Baseline Method Samples')
    plt.tight_layout()
    plt.savefig(output_dir / f'correlation_matrix_{polarity.lower()}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare comparison stats
    comparison_stats = {
        'polarity': polarity,
        'sample_correlations': sample_correlations,
        'mean_correlation': np.mean(sample_correlations) if sample_correlations else np.nan,
        'min_correlation': np.min(sample_correlations) if sample_correlations else np.nan,
        'baseline_stats': {
            'mean': np.mean(baseline_flat) if len(baseline_flat) > 0 else np.nan,
            'std': np.std(baseline_flat) if len(baseline_flat) > 0 else np.nan,
            'skewness': stats.skew(baseline_flat) if len(baseline_flat) > 0 else np.nan
        },
        'robust_stats': {
            'mean': np.mean(robust_flat) if len(robust_flat) > 0 else np.nan,
            'std': np.std(robust_flat) if len(robust_flat) > 0 else np.nan, 
            'skewness': stats.skew(robust_flat) if len(robust_flat) > 0 else np.nan
        }
    }
    
    return comparison_stats

def main():
    """Execute Phase 2 dual normalization"""
    print("=== Phase 2: Dual Normalization ===")
    
    # Load datasets
    pos_df = pd.read_csv('out/all_positive_data_renamed.tsv', sep='\t')
    neg_df = pd.read_csv('out/all_negative_data_renamed.tsv', sep='\t')
    
    # Create output directories
    results_dir = Path('results')
    pos_baseline_dir = results_dir / 'positive' / 'baseline_TICsqrt'
    pos_robust_dir = results_dir / 'positive' / 'robust_PQNsqrtPareto'
    neg_baseline_dir = results_dir / 'negative' / 'baseline_TICsqrt'  
    neg_robust_dir = results_dir / 'negative' / 'robust_PQNsqrtPareto'
    qc_dir = Path('qc')
    
    comparison_results = {}
    
    # Process both polarities
    for df, polarity, baseline_dir, robust_dir in [
        (pos_df, 'Positive', pos_baseline_dir, pos_robust_dir),
        (neg_df, 'Negative', neg_baseline_dir, neg_robust_dir)
    ]:
        print(f"\nProcessing {polarity} ion data...")
        
        # 1. Baseline normalization (TIC + sqrt)
        print("  Applying baseline normalization (TIC + √)...")
        baseline_normalized = tic_normalization(df)
        
        # 2. Robust normalization (PQN + sqrt + Pareto)
        print("  Applying robust normalization (PQN + √ + Pareto)...")
        robust_normalized, pqn_factors = robust_normalization_pipeline(df)
        
        # 3. Save normalized matrices
        baseline_normalized.to_csv(baseline_dir / 'normalized_data.tsv', sep='\t', index=False)
        robust_normalized.to_csv(robust_dir / 'normalized_data.tsv', sep='\t', index=False)
        
        # Save PQN factors
        pqn_df = pd.DataFrame(list(pqn_factors.items()), columns=['Sample', 'PQN_Factor'])
        pqn_df.to_csv(robust_dir / 'pqn_factors.csv', index=False)
        
        # 4. Generate comparison diagnostics
        print("  Generating diagnostic plots...")
        comparison_stats = compare_normalizations(baseline_normalized, robust_normalized, 
                                                polarity, qc_dir)
        comparison_results[polarity] = comparison_stats
        
        print(f"  Mean correlation between methods: {comparison_stats['mean_correlation']:.3f}")
    
    # 5. Generate summary report
    print("\nGenerating normalization summary...")
    summary_lines = []
    summary_lines.append("=== Dual Normalization Summary ===")
    summary_lines.append(f"Analysis Date: {pd.Timestamp.now()}")
    summary_lines.append("")
    
    for polarity, stats in comparison_results.items():
        summary_lines.append(f"=== {polarity.upper()} ION MODE ===")
        summary_lines.append("Method Comparison:")
        summary_lines.append(f"  Mean cross-correlation: {stats['mean_correlation']:.3f}")
        summary_lines.append(f"  Min cross-correlation: {stats['min_correlation']:.3f}")
        summary_lines.append("")
        
        summary_lines.append("Baseline (TIC+√) Statistics:")
        summary_lines.append(f"  Mean: {stats['baseline_stats']['mean']:.4f}")
        summary_lines.append(f"  Std: {stats['baseline_stats']['std']:.4f}")
        summary_lines.append(f"  Skewness: {stats['baseline_stats']['skewness']:.3f}")
        summary_lines.append("")
        
        summary_lines.append("Robust (PQN+√+Pareto) Statistics:")
        summary_lines.append(f"  Mean: {stats['robust_stats']['mean']:.4f}")
        summary_lines.append(f"  Std: {stats['robust_stats']['std']:.4f}")
        summary_lines.append(f"  Skewness: {stats['robust_stats']['skewness']:.3f}")
        summary_lines.append("")
    
    summary_lines.append("Conclusion:")
    if all(stats['mean_correlation'] > 0.7 for stats in comparison_results.values()):
        summary_lines.append("High correlation between methods indicates robust conclusions.")
    else:
        summary_lines.append("Moderate correlation - conclusions should be validated across both methods.")
    
    with open(qc_dir / 'normalization_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print("\n=== Phase 2 Complete ===")
    print("Generated files:")
    print("  results/positive/baseline_TICsqrt/normalized_data.tsv")
    print("  results/positive/robust_PQNsqrtPareto/normalized_data.tsv") 
    print("  results/positive/robust_PQNsqrtPareto/pqn_factors.csv")
    print("  results/negative/baseline_TICsqrt/normalized_data.tsv")
    print("  results/negative/robust_PQNsqrtPareto/normalized_data.tsv")
    print("  results/negative/robust_PQNsqrtPareto/pqn_factors.csv")
    print("  qc/normalization_summary.txt")
    print("  qc/normalization_comparison_positive.png")
    print("  qc/normalization_comparison_negative.png")
    print("  qc/correlation_matrix_positive.png")
    print("  qc/correlation_matrix_negative.png")

if __name__ == "__main__":
    main()