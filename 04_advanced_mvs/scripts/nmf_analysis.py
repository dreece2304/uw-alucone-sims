#!/usr/bin/env python3
"""
NMF Analysis for Phase 4 Advanced MVS
Executes NMF on Phase-2 matrices to obtain chemically interpretable factors
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
import os
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fixed parameters
K_VALUES = [2, 3, 4, 5, 6]
SEEDS = [0, 1, 2]
NMF_PARAMS = {
    'solver': 'mu',  # multiplicative update
    'tol': 1e-4,
    'max_iter': 1000,
    'random_state': None  # will be set per seed
}

def load_inputs_manifest():
    """Load and validate inputs manifest"""
    manifest_path = "04_advanced_mvs/manifests/inputs.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest

def parse_sample_metadata(sample_headers):
    """Parse sample metadata from headers"""
    metadata = []
    pattern = re.compile(r'P(\d+)_(\d+)(?:uC|μC)-([PN])')
    
    for header in sample_headers:
        match = pattern.match(header)
        if match:
            pattern_id = f"P{match.group(1)}"
            dose_uC = int(match.group(2))
            polarity = match.group(3)
            metadata.append({
                'sample_id': header,
                'pattern': pattern_id,
                'dose_uC': dose_uC,
                'polarity': polarity
            })
        else:
            print(f"Warning: Could not parse sample header: {header}")
    
    return pd.DataFrame(metadata)

def load_dataset(file_path, dataset_slug):
    """Load and validate dataset"""
    if not os.path.exists(file_path):
        print(f"Dataset {dataset_slug}: {file_path} not present")
        return None, None, None
        
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    
    # Validate first column name
    if df.index.name != 'Mass (u)':
        print(f"Warning: Expected first column 'Mass (u)', got '{df.index.name}'")
    
    # Parse sample metadata
    sample_metadata = parse_sample_metadata(df.columns.tolist())
    
    # Data matrix (features x samples)
    X = df.values
    mass_mz = df.index.values
    
    print(f"Dataset {dataset_slug}: {X.shape[0]} features x {X.shape[1]} samples")
    
    return X, mass_mz, sample_metadata

def compute_cosine_similarity_matrix(W1, W2):
    """Compute cosine similarity matrix between component sets"""
    # Normalize components
    W1_norm = W1 / (np.linalg.norm(W1, axis=0, keepdims=True) + 1e-12)
    W2_norm = W2 / (np.linalg.norm(W2, axis=0, keepdims=True) + 1e-12)
    
    # Compute similarity matrix
    sim_matrix = W1_norm.T @ W2_norm
    return sim_matrix

def match_components_hungarian(similarities_list):
    """Match components across seeds using Hungarian algorithm"""
    n_components = similarities_list[0].shape[0]
    n_seeds = len(similarities_list)
    
    # For each component, find best matches across all seed pairs
    component_similarities = []
    
    for comp_idx in range(n_components):
        comp_sims = []
        
        # Compare all pairs of seeds
        for i in range(n_seeds):
            for j in range(i+1, n_seeds):
                sim_matrix = similarities_list[j] if i == 0 else compute_cosine_similarity_matrix(
                    np.eye(n_components)[:, [comp_idx]], np.eye(n_components)
                )
                
                # Use Hungarian algorithm to find best assignment
                row_indices, col_indices = linear_sum_assignment(-np.abs(sim_matrix))
                
                # Find similarity for this component
                if comp_idx in row_indices:
                    matched_comp = col_indices[list(row_indices).index(comp_idx)]
                    comp_sims.append(abs(sim_matrix[comp_idx, matched_comp]))
        
        # Use median similarity across all pairs
        component_similarities.append(np.median(comp_sims) if comp_sims else 0.0)
    
    return component_similarities

def run_nmf_single(X, k, seed):
    """Run single NMF with given parameters"""
    nmf_params = NMF_PARAMS.copy()
    nmf_params['n_components'] = k
    nmf_params['random_state'] = seed
    
    model = NMF(**nmf_params)
    W = model.fit_transform(X)  # Components (spectra)
    H = model.components_      # Coefficients (concentrations)
    
    return model, W, H

def compute_reconstruction_metrics(X, X_reconstructed):
    """Compute reconstruction quality metrics"""
    # Frobenius R²
    X_mean = np.mean(X, axis=1, keepdims=True)
    TSS = np.sum((X - X_mean) ** 2)
    RSS = np.sum((X - X_reconstructed) ** 2)
    r_squared = 1 - (RSS / TSS)
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(X.flatten(), X_reconstructed.flatten()))
    
    return r_squared, rmse

def compute_dose_correlations(H, sample_metadata):
    """Compute dose correlations for each component"""
    correlations = []
    
    for comp_idx in range(H.shape[0]):
        coefficients = H[comp_idx, :]
        doses = sample_metadata['dose_uC'].values
        
        # Overall correlation (all 15 samples)
        try:
            rho_all, p_all = spearmanr(doses, coefficients)
            if np.isnan(rho_all):
                rho_all, p_all = 0.0, 1.0
        except:
            rho_all, p_all = 0.0, 1.0
            
        # Dose means correlation (5 points)
        dose_means = []
        coeff_means = []
        unique_doses = sorted(sample_metadata['dose_uC'].unique())
        
        for dose in unique_doses:
            dose_mask = sample_metadata['dose_uC'] == dose
            coeff_mean = np.mean(coefficients[dose_mask])
            dose_means.append(dose)
            coeff_means.append(coeff_mean)
        
        try:
            rho5, p5 = spearmanr(dose_means, coeff_means)
            if np.isnan(rho5):
                rho5, p5 = 0.0, 1.0
        except:
            rho5, p5 = 0.0, 1.0
        
        correlations.append({
            'component': comp_idx,
            'rho_all': float(rho_all),
            'p_all': float(p_all),
            'rho5': float(rho5),
            'p5': float(p5)
        })
    
    return correlations

def create_component_spectra_plot(W, mass_mz, dataset_slug, k):
    """Create component spectra plot"""
    n_components = W.shape[1]
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot spectrum
        ax.plot(mass_mz, W[:, i], 'b-', linewidth=1)
        ax.set_title(f'Component {i+1} Spectrum')
        ax.set_xlabel('Mass (u)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        
        # Add top-20 m/z labels
        top_indices = np.argsort(W[:, i])[-20:]
        for idx in top_indices:
            if W[idx, i] > 0.001:  # Only label significant peaks
                ax.annotate(f'{mass_mz[idx]:.1f}', 
                           xy=(mass_mz[idx], W[idx, i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/nmf_component_spectra__{dataset_slug}__k{k}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_coefficients_heatmap(H, sample_metadata, dataset_slug, k):
    """Create coefficients heatmap"""
    fig, ax = plt.subplots(figsize=(15, max(4, k*0.8)))
    
    # Create sample labels with dose and pattern info
    sample_labels = []
    for _, row in sample_metadata.iterrows():
        label = f"{row['pattern']}_{row['dose_uC']}"
        sample_labels.append(label)
    
    # Create heatmap
    im = ax.imshow(H, aspect='auto', cmap='viridis', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.set_yticks(range(k))
    ax.set_yticklabels([f'Comp {i+1}' for i in range(k)])
    ax.set_xlabel('Samples (Pattern_Dose)')
    ax.set_ylabel('Components')
    ax.set_title(f'NMF Coefficients Heatmap (k={k})')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Coefficient Value')
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/nmf_coefficients_heatmap__{dataset_slug}__k{k}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_quality_curves_plot(metrics_by_seed, dataset_slug, k):
    """Create quality curves plot for this k"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    seeds = list(metrics_by_seed.keys())
    r_squared_values = [metrics_by_seed[seed]['r_squared'] for seed in seeds]
    rmse_values = [metrics_by_seed[seed]['rmse'] for seed in seeds]
    
    # R² plot
    ax1.bar(seeds, r_squared_values, alpha=0.7, color='blue')
    ax1.axhline(np.median(r_squared_values), color='red', linestyle='--', label=f'Median: {np.median(r_squared_values):.3f}')
    ax1.set_xlabel('Seed')
    ax1.set_ylabel('R²')
    ax1.set_title(f'R² across seeds (k={k})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE plot
    ax2.bar(seeds, rmse_values, alpha=0.7, color='orange')
    ax2.axhline(np.median(rmse_values), color='red', linestyle='--', label=f'Median: {np.median(rmse_values):.3f}')
    ax2.set_xlabel('Seed')
    ax2.set_ylabel('RMSE')
    ax2.set_title(f'RMSE across seeds (k={k})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/nmf_quality_curves__{dataset_slug}__k{k}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def analyze_dataset(dataset_path, dataset_slug):
    """Analyze single dataset with NMF"""
    print(f"\n=== Analyzing {dataset_slug} ===")
    
    # Load dataset
    X, mass_mz, sample_metadata = load_dataset(dataset_path, dataset_slug)
    if X is None:
        return None
    
    all_results = {}
    
    # Run NMF for each k
    for k in K_VALUES:
        print(f"Running NMF k={k}...")
        
        k_results = {
            'models': {},
            'W_matrices': {},
            'H_matrices': {},
            'metrics': {},
            'correlations': {}
        }
        
        # Run for each seed
        for seed in SEEDS:
            model, W, H = run_nmf_single(X, k, seed)
            
            # Store results
            k_results['models'][seed] = model
            k_results['W_matrices'][seed] = W
            k_results['H_matrices'][seed] = H
            
            # Compute metrics
            X_reconstructed = W @ H
            r_squared, rmse = compute_reconstruction_metrics(X, X_reconstructed)
            correlations = compute_dose_correlations(H, sample_metadata)
            
            k_results['metrics'][seed] = {
                'r_squared': r_squared,
                'rmse': rmse,
                'loss': model.reconstruction_err_
            }
            k_results['correlations'][seed] = correlations
        
        # Compute component stability
        W_list = [k_results['W_matrices'][seed] for seed in SEEDS]
        similarities = []
        for i in range(len(SEEDS)):
            for j in range(i+1, len(SEEDS)):
                sim_matrix = compute_cosine_similarity_matrix(W_list[i], W_list[j])
                similarities.append(sim_matrix)
        
        # Match components and compute stability
        component_stabilities = []
        for comp_idx in range(k):
            comp_sims = []
            for sim_matrix in similarities:
                # Use Hungarian to find best matching
                row_indices, col_indices = linear_sum_assignment(-np.abs(sim_matrix))
                if comp_idx < len(row_indices):
                    matched_comp = col_indices[comp_idx] if comp_idx < len(col_indices) else comp_idx
                    comp_sims.append(abs(sim_matrix[comp_idx, matched_comp]))
            component_stabilities.append(np.median(comp_sims) if comp_sims else 0.0)
        
        k_results['component_stabilities'] = component_stabilities
        
        # Use median seed for representative results
        median_seed_idx = 1  # Use seed=1 as median
        median_seed = SEEDS[median_seed_idx]
        W_median = k_results['W_matrices'][median_seed]
        H_median = k_results['H_matrices'][median_seed]
        
        # Generate plots
        create_component_spectra_plot(W_median, mass_mz, dataset_slug, k)
        create_coefficients_heatmap(H_median, sample_metadata, dataset_slug, k)
        create_quality_curves_plot(k_results['metrics'], dataset_slug, k)
        
        # Save component matrices
        W_df = pd.DataFrame(W_median, index=mass_mz, columns=[f'Comp_{i+1}' for i in range(k)])
        W_df.index.name = 'Mass (u)'
        W_df.to_csv(f"04_advanced_mvs/logs/nmf_components__{dataset_slug}__k{k}.tsv", sep='\t')
        
        H_df = pd.DataFrame(H_median, 
                           index=[f'Comp_{i+1}' for i in range(k)],
                           columns=sample_metadata['sample_id'])
        H_df.index.name = 'component_id'
        H_df.to_csv(f"04_advanced_mvs/logs/nmf_coefficients__{dataset_slug}__k{k}.tsv", sep='\t')
        
        # Save top m/z for each component
        top_mz_data = []
        for comp_idx in range(k):
            top_indices = np.argsort(W_median[:, comp_idx])[-20:][::-1]  # Top 20, descending
            for rank, idx in enumerate(top_indices):
                top_mz_data.append({
                    'component_id': f'Comp_{comp_idx+1}',
                    'rank': rank + 1,
                    'mz': mass_mz[idx],
                    'weight': W_median[idx, comp_idx]
                })
        
        pd.DataFrame(top_mz_data).to_csv(
            f"04_advanced_mvs/logs/nmf_top_mz__{dataset_slug}__k{k}.csv", index=False)
        
        # Save run log
        run_log = {
            'dataset': dataset_slug,
            'k': k,
            'seeds': SEEDS,
            'shape': X.shape,
            'metrics_by_seed': {
                seed: {
                    'r_squared': float(k_results['metrics'][seed]['r_squared']),
                    'rmse': float(k_results['metrics'][seed]['rmse']),
                    'loss': float(k_results['metrics'][seed]['loss'])
                } for seed in SEEDS
            },
            'component_stabilities': [float(s) for s in component_stabilities],
            'dose_correlations_median_seed': k_results['correlations'][median_seed],
            'stable_fraction': float(np.mean(np.array(component_stabilities) >= 0.90))
        }
        
        with open(f"04_advanced_mvs/logs/nmf_run__{dataset_slug}__k{k}.json", 'w') as f:
            json.dump(run_log, f, indent=2)
        
        all_results[k] = k_results
    
    # Model selection
    selection_results = {}
    for k in K_VALUES:
        metrics_by_seed = all_results[k]['metrics']
        median_r_squared = np.median([metrics_by_seed[seed]['r_squared'] for seed in SEEDS])
        median_rmse = np.median([metrics_by_seed[seed]['rmse'] for seed in SEEDS])
        stable_fraction = np.mean(np.array(all_results[k]['component_stabilities']) >= 0.90)
        
        # Best dose correlation across components
        correlations = all_results[k]['correlations'][SEEDS[1]]  # Use median seed
        best_rho5 = max([abs(c['rho5']) for c in correlations])
        
        selection_results[k] = {
            'median_r_squared': float(median_r_squared),
            'median_rmse': float(median_rmse),
            'stable_fraction': float(stable_fraction),
            'best_abs_rho5': float(best_rho5)
        }
    
    # Choose k* using heuristic
    k_star = None
    best_r_squared = -1
    
    for k in K_VALUES:
        r_squared = selection_results[k]['median_r_squared']
        if r_squared > best_r_squared:
            # Check diminishing returns
            if k_star is None or r_squared - best_r_squared >= 0.02:
                k_star = k
                best_r_squared = r_squared
            elif r_squared - best_r_squared < 0.02:
                # Diminishing returns, use tie-breakers
                if (selection_results[k]['stable_fraction'] > selection_results[k_star]['stable_fraction'] or
                    (selection_results[k]['stable_fraction'] == selection_results[k_star]['stable_fraction'] and
                     selection_results[k]['best_abs_rho5'] > selection_results[k_star]['best_abs_rho5'])):
                    k_star = k
                    best_r_squared = r_squared
    
    # Compute acceptance criteria
    k_star_metrics = selection_results[k_star]
    passed_recon = k_star_metrics['median_r_squared'] >= 0.70
    
    # Check dose correlation for k_star
    k_star_correlations = all_results[k_star]['correlations'][SEEDS[1]]
    passed_dose = any(abs(c['rho5']) >= 0.60 and c['p5'] <= 0.05 for c in k_star_correlations)
    passed_stability = k_star_metrics['stable_fraction'] >= 0.67
    
    # Save model selection results
    model_selection = {
        'dataset': dataset_slug,
        'k_star': k_star,
        'selection_metrics': selection_results,
        'passed_recon': passed_recon,
        'passed_dose': passed_dose,
        'passed_stability': passed_stability,
        'timestamp': datetime.now().isoformat() + 'Z'
    }
    
    with open(f"04_advanced_mvs/logs/nmf_model_selection__{dataset_slug}.json", 'w') as f:
        json.dump(model_selection, f, indent=2)
    
    # Create model selection plot
    k_values = list(selection_results.keys())
    r_squared_values = [selection_results[k]['median_r_squared'] for k in k_values]
    rmse_values = [selection_results[k]['median_rmse'] for k in k_values]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # R² plot
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Components (k)')
    ax1.set_ylabel('R²', color=color1)
    line1 = ax1.plot(k_values, r_squared_values, 'o-', color=color1, linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Mark k*
    k_star_r2 = selection_results[k_star]['median_r_squared']
    ax1.plot(k_star, k_star_r2, 'ro', markersize=12, label=f'k*={k_star}')
    
    # RMSE plot (twin axis)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('RMSE', color=color2)
    line2 = ax2.plot(k_values, rmse_values, 's--', color=color2, linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add legend
    lines = line1 + line2 + [ax1.lines[-1]]  # Include k* marker
    labels = ['R²', 'RMSE', f'k*={k_star}']
    ax1.legend(lines, labels, loc='center right')
    
    plt.title(f'NMF Model Selection: {dataset_slug}')
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/nmf_model_selection__{dataset_slug}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    create_summary_report(dataset_slug, k_star, selection_results, 
                         all_results[k_star]['correlations'][SEEDS[1]],
                         passed_recon, passed_dose, passed_stability)
    
    # Print summary
    print(f"DATASET={dataset_slug}  k*={k_star}  R2={k_star_metrics['median_r_squared']:.3f}  "
          f"stable_frac={k_star_metrics['stable_fraction']:.2f}  "
          f"best|rho5|={k_star_metrics['best_abs_rho5']:.2f}")
    
    return {
        'dataset_slug': dataset_slug,
        'k_star': k_star,
        'results': all_results,
        'selection': model_selection
    }

def create_summary_report(dataset_slug, k_star, selection_results, correlations, 
                         passed_recon, passed_dose, passed_stability):
    """Create markdown summary report"""
    
    # Find best component for dose correlation
    best_comp = max(correlations, key=lambda x: abs(x['rho5']))
    
    # Get top m/z for best component
    top_mz_df = pd.read_csv(f"04_advanced_mvs/logs/nmf_top_mz__{dataset_slug}__k{k_star}.csv")
    best_comp_mz = top_mz_df[top_mz_df['component_id'] == f"Comp_{best_comp['component']+1}"].head(10)
    
    report = f"""# NMF Summary Report: {dataset_slug}

## Model Selection

**Chosen k***: {k_star} components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
"""
    
    for k in sorted(selection_results.keys()):
        metrics = selection_results[k]
        report += f"| {k} | {metrics['median_r_squared']:.3f} | {metrics['median_rmse']:.3f} | {metrics['stable_fraction']:.2f} | {metrics['best_abs_rho5']:.2f} |\n"
    
    report += f"""
## Best Dose-Responsive Component

**Component {best_comp['component']+1}** shows strongest dose correlation:
- ρ₅ (dose means): {best_comp['rho5']:.3f} (p = {best_comp['p5']:.3f})
- ρ (all samples): {best_comp['rho_all']:.3f} (p = {best_comp['p_all']:.3f})

### Top Contributing m/z Values

"""
    
    for _, row in best_comp_mz.iterrows():
        report += f"- **{row['mz']:.1f}** (weight: {row['weight']:.4f})\n"
    
    report += f"""

## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): {selection_results[k_star]['stable_fraction']:.2f}

## Reconstruction Quality

- R² = {selection_results[k_star]['median_r_squared']:.3f}
- RMSE = {selection_results[k_star]['median_rmse']:.3f}

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | {selection_results[k_star]['median_r_squared']:.3f} | {'✅ PASS' if passed_recon else '❌ FAIL'} |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | {abs(best_comp['rho5']):.3f} (p={best_comp['p5']:.3f}) | {'✅ PASS' if passed_dose else '❌ FAIL'} |
| Stability (≥67% stable) | {selection_results[k_star]['stable_fraction']:.1%} | {'✅ PASS' if passed_stability else '❌ FAIL'} |

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f"04_advanced_mvs/reports/NMF_SUMMARY__{dataset_slug}.md", 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("Starting NMF Analysis...")
    
    # Load inputs manifest
    manifest = load_inputs_manifest()
    
    # Define datasets to analyze
    datasets_to_analyze = []
    
    # Primary dataset (required)
    primary_path = manifest['primary_input']
    datasets_to_analyze.append((primary_path, 'combined_robust'))
    
    # Secondary datasets (optional)
    secondary_datasets = [
        ('matrices_pos.robust_pqn_sqrt_pos', 'pos_robust'),
        ('matrices_neg.robust_pqn_sqrt_neg', 'neg_robust')
    ]
    
    for key_path, slug in secondary_datasets:
        keys = key_path.split('.')
        if keys[0] in manifest and keys[1] in manifest[keys[0]]:
            path = manifest[keys[0]][keys[1]]
            datasets_to_analyze.append((path, slug))
    
    # Sensitivity datasets (optional)
    sensitivity_datasets = [
        ('matrices.combined.baseline_tic_sqrt', 'combined_baseline'),
        ('matrices.combined.robust_pqn_sqrt_pareto', 'combined_robust_pareto')
    ]
    
    for key_path, slug in sensitivity_datasets:
        keys = key_path.split('.')
        if keys[0] in manifest and keys[1] in manifest[keys[0]] and keys[2] in manifest[keys[0]][keys[1]]:
            path = manifest[keys[0]][keys[1]][keys[2]]
            datasets_to_analyze.append((path, slug))
    
    # Analyze each dataset
    all_dataset_results = []
    for dataset_path, dataset_slug in datasets_to_analyze:
        result = analyze_dataset(dataset_path, dataset_slug)
        if result:
            all_dataset_results.append(result)
    
    # Create run manifest
    run_manifest = {
        'schema': 'nmf_run_manifest.v1',
        'timestamp': datetime.now().isoformat() + 'Z',
        'git_commit': manifest['metadata']['git_commit'],
        'datasets_processed': [r['dataset_slug'] for r in all_dataset_results],
        'k_values': K_VALUES,
        'seeds': SEEDS,
        'output_files': []
    }
    
    # Collect all output files
    for result in all_dataset_results:
        dataset_slug = result['dataset_slug']
        for k in K_VALUES:
            run_manifest['output_files'].extend([
                f"04_advanced_mvs/figures/nmf_component_spectra__{dataset_slug}__k{k}.png",
                f"04_advanced_mvs/figures/nmf_coefficients_heatmap__{dataset_slug}__k{k}.png",
                f"04_advanced_mvs/figures/nmf_quality_curves__{dataset_slug}__k{k}.png",
                f"04_advanced_mvs/logs/nmf_run__{dataset_slug}__k{k}.json",
                f"04_advanced_mvs/logs/nmf_components__{dataset_slug}__k{k}.tsv",
                f"04_advanced_mvs/logs/nmf_coefficients__{dataset_slug}__k{k}.tsv",
                f"04_advanced_mvs/logs/nmf_top_mz__{dataset_slug}__k{k}.csv"
            ])
        
        run_manifest['output_files'].extend([
            f"04_advanced_mvs/figures/nmf_model_selection__{dataset_slug}.png",
            f"04_advanced_mvs/logs/nmf_model_selection__{dataset_slug}.json",
            f"04_advanced_mvs/reports/NMF_SUMMARY__{dataset_slug}.md"
        ])
    
    # Save run manifest
    with open("04_advanced_mvs/manifests/NMF_RUN_MANIFEST.json", 'w') as f:
        json.dump(run_manifest, f, indent=2)
    
    print(f"\nNMF complete. Artifacts written: manifests/logs/figures/reports. No new directories created.")
    print(f"Processed {len(all_dataset_results)} datasets with k={K_VALUES}")

if __name__ == "__main__":
    main()