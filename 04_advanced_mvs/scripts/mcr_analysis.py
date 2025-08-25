#!/usr/bin/env python3
"""
MCR-ALS Analysis for Phase 4 Advanced MVS
Executes MCR-ALS on Phase-2 matrices using NMF initialization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment, nnls
import os
import re
from datetime import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Fixed parameters
MAX_ITER = 200
TOLERANCE = 1e-5
N_STARTS = 3

def load_inputs_manifest():
    """Load and validate inputs manifest"""
    manifest_path = "04_advanced_mvs/manifests/inputs.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    return manifest

def load_nmf_selection_results():
    """Load NMF model selection results for all datasets"""
    selection_files = [
        "04_advanced_mvs/logs/nmf_model_selection__combined_robust.json",
        "04_advanced_mvs/logs/nmf_model_selection__pos_robust.json",
        "04_advanced_mvs/logs/nmf_model_selection__neg_robust.json",
        "04_advanced_mvs/logs/nmf_model_selection__combined_baseline.json",
        "04_advanced_mvs/logs/nmf_model_selection__combined_robust_pareto.json"
    ]
    
    selections = {}
    for file_path in selection_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                selections[data['dataset']] = data
    
    return selections

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
    
    # Sanity check: values should be non-negative
    if np.any(X < 0):
        print(f"Warning: Dataset {dataset_slug} contains negative values")
    
    print(f"Dataset {dataset_slug}: {X.shape[0]} features x {X.shape[1]} samples")
    
    return X, mass_mz, sample_metadata

def load_nmf_initialization(dataset_slug, k_star):
    """Load NMF components and coefficients for initialization"""
    # Load spectra (W -> S)
    spectra_path = f"04_advanced_mvs/logs/nmf_components__{dataset_slug}__k{k_star}.tsv"
    if not os.path.exists(spectra_path):
        raise FileNotFoundError(f"NMF spectra file not found: {spectra_path}")
    
    spectra_df = pd.read_csv(spectra_path, sep='\t', index_col=0)
    S_init = spectra_df.values  # (n_features, n_components)
    
    # Load coefficients (H -> C)
    coeff_path = f"04_advanced_mvs/logs/nmf_coefficients__{dataset_slug}__k{k_star}.tsv"
    if not os.path.exists(coeff_path):
        raise FileNotFoundError(f"NMF coefficients file not found: {coeff_path}")
    
    coeff_df = pd.read_csv(coeff_path, sep='\t', index_col=0)
    C_init = coeff_df.values  # (n_components, n_samples)
    
    return S_init, C_init

def create_perturbed_initialization(S0, C0, perturbation_factor=0.01):
    """Create perturbed initialization for additional starts"""
    # Add small random noise (≤1% relative magnitude)
    S_pert = S0 * (1 + perturbation_factor * np.random.randn(*S0.shape))
    C_pert = C0 * (1 + perturbation_factor * np.random.randn(*C0.shape))
    
    # Ensure non-negativity
    S_pert = np.maximum(S_pert, 0)
    C_pert = np.maximum(C_pert, 0)
    
    return S_pert, C_pert

def mcr_als_fit(X, S_init, C_init, max_iter=MAX_ITER, tol=TOLERANCE):
    """Perform MCR-ALS fitting with nonnegativity constraints"""
    
    n_features, n_samples = X.shape
    n_components = S_init.shape[1]
    
    # Initialize
    S = S_init.copy()
    C = C_init.copy()
    
    # Track convergence
    residuals = []
    converged = False
    
    with tqdm(range(max_iter), desc="MCR-ALS iterations", unit="iter") as pbar:
        for iteration in pbar:
            S_old = S.copy()
            C_old = C.copy()
            
            # Update C (solve for each sample)
            for sample_idx in range(n_samples):
                C_new_sample, _ = nnls(S, X[:, sample_idx])
                if len(C_new_sample) == n_components:
                    C[:, sample_idx] = C_new_sample
            
            # Update S (solve for each feature)
            for feature_idx in range(n_features):
                S_new_feature, _ = nnls(C.T, X[feature_idx, :])
                if len(S_new_feature) == n_components:
                    S[feature_idx, :] = S_new_feature
        
            # Check for NaN or inf
            if np.any(np.isnan(S)) or np.any(np.isnan(C)) or np.any(np.isinf(S)) or np.any(np.isinf(C)):
                pbar.set_postfix(status="NaN/Inf detected", refresh=True)
                return S_old, C_old, residuals, False
            
            # Compute residual
            X_reconstructed = S @ C
            residual = np.sum((X - X_reconstructed) ** 2)
            residuals.append(residual)
            
            # Update progress bar with current residual
            pbar.set_postfix(residual=f"{residual:.2e}", refresh=True)
            
            # Check convergence
            if iteration > 0:
                relative_change = abs(residuals[-2] - residuals[-1]) / (residuals[-2] + 1e-12)
                if relative_change < tol:
                    converged = True
                    pbar.set_postfix(status=f"Converged! rel_change={relative_change:.2e}", refresh=True)
                    break
    
    if not converged:
        print(f"Did not converge after {max_iter} iterations")
    
    return S, C, residuals, converged

def compute_cosine_similarity_matrix(S1, S2):
    """Compute cosine similarity matrix between component sets"""
    # Normalize components
    S1_norm = S1 / (np.linalg.norm(S1, axis=0, keepdims=True) + 1e-12)
    S2_norm = S2 / (np.linalg.norm(S2, axis=0, keepdims=True) + 1e-12)
    
    # Compute similarity matrix
    sim_matrix = S1_norm.T @ S2_norm
    return sim_matrix

def match_components_hungarian(S_list):
    """Match components across starts using Hungarian algorithm"""
    n_starts = len(S_list)
    n_components = S_list[0].shape[1]
    
    if n_starts < 2:
        return [1.0] * n_components
    
    # Compute all pairwise similarities
    all_similarities = []
    for i in range(n_starts):
        for j in range(i+1, n_starts):
            sim_matrix = compute_cosine_similarity_matrix(S_list[i], S_list[j])
            all_similarities.append(sim_matrix)
    
    # For each component, find median similarity across all pairs
    component_similarities = []
    for comp_idx in range(n_components):
        comp_sims = []
        
        for sim_matrix in all_similarities:
            # Use Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(-np.abs(sim_matrix))
            
            # Find similarity for this component
            if comp_idx < len(row_indices):
                matched_comp = col_indices[comp_idx] if comp_idx < len(col_indices) else comp_idx
                comp_sims.append(abs(sim_matrix[comp_idx, matched_comp]))
        
        # Use median similarity
        component_similarities.append(np.median(comp_sims) if comp_sims else 0.0)
    
    return component_similarities

def compute_reconstruction_metrics(X, X_reconstructed):
    """Compute reconstruction quality metrics"""
    # Frobenius R² (against column-mean baseline)
    X_mean = np.mean(X, axis=1, keepdims=True)
    TSS = np.sum((X - X_mean) ** 2)
    RSS = np.sum((X - X_reconstructed) ** 2)
    r_squared = 1 - (RSS / TSS)
    
    # RMSE
    rmse = np.sqrt(np.mean((X - X_reconstructed) ** 2))
    
    return r_squared, rmse

def compute_dose_correlations(C, sample_metadata):
    """Compute dose correlations for each component"""
    correlations = []
    
    for comp_idx in range(C.shape[0]):
        concentrations = C[comp_idx, :]
        doses = sample_metadata['dose_uC'].values
        
        # Overall correlation (all 15 samples)
        try:
            rho_all, p_all = spearmanr(doses, concentrations)
            if np.isnan(rho_all):
                rho_all, p_all = 0.0, 1.0
        except:
            rho_all, p_all = 0.0, 1.0
            
        # Dose means correlation (5 points)
        dose_means = []
        conc_means = []
        unique_doses = sorted(sample_metadata['dose_uC'].unique())
        
        for dose in unique_doses:
            dose_mask = sample_metadata['dose_uC'] == dose
            conc_mean = np.mean(concentrations[dose_mask])
            dose_means.append(dose)
            conc_means.append(conc_mean)
        
        try:
            rho5, p5 = spearmanr(dose_means, conc_means)
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

def create_spectra_plot(S, mass_mz, dataset_slug, k_star):
    """Create MCR spectra plot"""
    n_components = S.shape[1]
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3*n_components))
    if n_components == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # Plot spectrum
        ax.plot(mass_mz, S[:, i], 'r-', linewidth=1.5, alpha=0.8)
        ax.set_title(f'MCR Component {i+1} Spectrum')
        ax.set_xlabel('Mass (u)')
        ax.set_ylabel('Weight')
        ax.grid(True, alpha=0.3)
        
        # Add top-20 m/z labels
        top_indices = np.argsort(S[:, i])[-20:]
        for idx in top_indices:
            if S[idx, i] > np.max(S[:, i]) * 0.01:  # Only label significant peaks
                ax.annotate(f'{mass_mz[idx]:.1f}', 
                           xy=(mass_mz[idx], S[idx, i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/mcr_spectra__{dataset_slug}__k{k_star}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_concentrations_heatmap(C, sample_metadata, dataset_slug, k_star):
    """Create MCR concentrations heatmap"""
    fig, ax = plt.subplots(figsize=(15, max(4, k_star*0.8)))
    
    # Create sample labels with dose and pattern info
    sample_labels = []
    for _, row in sample_metadata.iterrows():
        label = f"{row['pattern']}_{row['dose_uC']}"
        sample_labels.append(label)
    
    # Create heatmap
    im = ax.imshow(C, aspect='auto', cmap='plasma', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(range(len(sample_labels)))
    ax.set_xticklabels(sample_labels, rotation=45, ha='right')
    ax.set_yticks(range(k_star))
    ax.set_yticklabels([f'Comp {i+1}' for i in range(k_star)])
    ax.set_xlabel('Samples (Pattern_Dose)')
    ax.set_ylabel('Components')
    ax.set_title(f'MCR Concentrations Heatmap (k={k_star})')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Concentration')
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/mcr_concentrations_heatmap__{dataset_slug}__k{k_star}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_reconstruction_fit_plot(X, X_reconstructed, r_squared, rmse, dataset_slug, k_star):
    """Create reconstruction fit plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot: observed vs reconstructed
    ax1.scatter(X.flatten(), X_reconstructed.flatten(), alpha=0.5, s=1)
    ax1.plot([X.min(), X.max()], [X.min(), X.max()], 'r--', linewidth=2)
    ax1.set_xlabel('Observed Intensity')
    ax1.set_ylabel('Reconstructed Intensity')
    ax1.set_title(f'Observed vs Reconstructed\nR² = {r_squared:.3f}, RMSE = {rmse:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # Residuals histogram
    residuals = (X - X_reconstructed).flatten()
    ax2.hist(residuals, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals (Observed - Reconstructed)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Residuals Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/mcr_reconstruction_fit__{dataset_slug}__k{k_star}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def create_dose_trends_plot(C, sample_metadata, correlations, dataset_slug, k_star):
    """Create dose trends plot"""
    n_components = C.shape[0]
    fig, axes = plt.subplots(n_components, 1, figsize=(10, 3*n_components))
    if n_components == 1:
        axes = [axes]
    
    unique_doses = sorted(sample_metadata['dose_uC'].unique())
    
    for i, ax in enumerate(axes):
        concentrations = C[i, :]
        doses = sample_metadata['dose_uC'].values
        
        # Scatter plot all samples
        ax.scatter(doses, concentrations, alpha=0.6, s=50, label='Individual samples')
        
        # Dose means
        dose_means = []
        conc_means = []
        conc_stds = []
        
        for dose in unique_doses:
            dose_mask = sample_metadata['dose_uC'] == dose
            conc_mean = np.mean(concentrations[dose_mask])
            conc_std = np.std(concentrations[dose_mask])
            dose_means.append(dose)
            conc_means.append(conc_mean)
            conc_stds.append(conc_std)
        
        # Plot dose means with error bars
        ax.errorbar(dose_means, conc_means, yerr=conc_stds, 
                   fmt='ro-', linewidth=2, markersize=8, capsize=5, label='Dose means ± SD')
        
        # Annotate correlations
        corr_info = correlations[i]
        ax.set_title(f'Component {i+1}: ρ₅ = {corr_info["rho5"]:.3f} (p = {corr_info["p5"]:.3f})\n'
                    f'ρ_all = {corr_info["rho_all"]:.3f} (p = {corr_info["p_all"]:.3f})')
        ax.set_xlabel('Dose (μC/cm²)')
        ax.set_ylabel('Concentration')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    output_path = f"04_advanced_mvs/figures/mcr_dose_trends__{dataset_slug}__k{k_star}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def analyze_dataset_mcr(dataset_path, dataset_slug, k_star):
    """Analyze single dataset with MCR-ALS"""
    print(f"\n=== MCR-ALS Analysis: {dataset_slug} (k*={k_star}) ===")
    
    # Load dataset
    X, mass_mz, sample_metadata = load_dataset(dataset_path, dataset_slug)
    if X is None:
        return None
    
    # Load NMF initialization
    try:
        S_init, C_init = load_nmf_initialization(dataset_slug, k_star)
        print(f"Loaded NMF initialization: S {S_init.shape}, C {C_init.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None
    
    # Column-normalize S if needed (keep totals comparable)
    S_init_norm = S_init / (np.sum(S_init, axis=0, keepdims=True) + 1e-12)
    
    # Create multiple starts
    np.random.seed(42)  # For reproducible perturbations
    
    starts = []
    # Start 0: Direct NMF initialization
    starts.append((S_init_norm.copy(), C_init.copy(), "direct_nmf"))
    
    # Starts 1-2: Perturbed versions
    for start_idx in range(1, N_STARTS):
        S_pert, C_pert = create_perturbed_initialization(S_init_norm, C_init)
        starts.append((S_pert, C_pert, f"perturbed_{start_idx}"))
    
    # Run MCR-ALS for each start
    results = []
    for start_idx, (S0, C0, start_name) in enumerate(starts):
        print(f"Running MCR-ALS start {start_idx} ({start_name})...")
        
        try:
            S_final, C_final, residuals, converged = mcr_als_fit(X, S0, C0)
            
            # Compute metrics
            X_reconstructed = S_final @ C_final
            r_squared, rmse = compute_reconstruction_metrics(X, X_reconstructed)
            correlations = compute_dose_correlations(C_final, sample_metadata)
            
            results.append({
                'start_idx': start_idx,
                'start_name': start_name,
                'S': S_final,
                'C': C_final,
                'converged': converged,
                'residuals': residuals,
                'r_squared': r_squared,
                'rmse': rmse,
                'correlations': correlations,
                'final_residual': residuals[-1] if residuals else np.inf
            })
            
            print(f"  Converged: {converged}, R²: {r_squared:.3f}, RMSE: {rmse:.4f}")
            
        except Exception as e:
            print(f"  Start {start_idx} failed: {e}")
            results.append({
                'start_idx': start_idx,
                'start_name': start_name,
                'converged': False,
                'error': str(e)
            })
    
    # Filter successful results (include non-converged but reasonable results)
    successful_results = [r for r in results if 'S' in r and r.get('r_squared', 0) > 0.5]
    
    if not successful_results:
        print(f"Error: No reasonable MCR-ALS fits for {dataset_slug}")
        return None
    
    # Prefer converged results, but accept non-converged if they have good metrics
    converged_results = [r for r in successful_results if r.get('converged', False)]
    if converged_results:
        successful_results = converged_results
        print(f"Using converged results ({len(converged_results)} out of {len(results)})")
    else:
        print(f"Using non-converged results with good metrics ({len(successful_results)} out of {len(results)})")
    
    # Choose best result (lowest final residual)
    best_result = min(successful_results, key=lambda r: r['final_residual'])
    print(f"Best result from start {best_result['start_idx']} ({best_result['start_name']})")
    
    # Compute stability across successful starts
    S_list = [r['S'] for r in successful_results]
    component_stabilities = match_components_hungarian(S_list)
    stable_fraction = np.mean(np.array(component_stabilities) >= 0.90)
    
    # Sanity check for suspiciously perfect results
    if (best_result['r_squared'] > 0.999 and 
        any(abs(c['rho5']) > 0.999 for c in best_result['correlations']) and
        stable_fraction < 0.90):
        print(f"WARNING: Suspiciously perfect results for {dataset_slug} with low stability!")
    
    # Use best result for outputs
    S_best = best_result['S']
    C_best = best_result['C']
    
    # Generate plots
    create_spectra_plot(S_best, mass_mz, dataset_slug, k_star)
    create_concentrations_heatmap(C_best, sample_metadata, dataset_slug, k_star)
    
    X_reconstructed_best = S_best @ C_best
    create_reconstruction_fit_plot(X, X_reconstructed_best, 
                                  best_result['r_squared'], best_result['rmse'],
                                  dataset_slug, k_star)
    create_dose_trends_plot(C_best, sample_metadata, best_result['correlations'],
                           dataset_slug, k_star)
    
    # Save spectra and concentrations
    S_df = pd.DataFrame(S_best, index=mass_mz, columns=[f'Comp_{i+1}' for i in range(k_star)])
    S_df.index.name = 'Mass (u)'
    S_df.to_csv(f"04_advanced_mvs/logs/mcr_spectra__{dataset_slug}__k{k_star}.tsv", sep='\t')
    
    C_df = pd.DataFrame(C_best, 
                       index=[f'Comp_{i+1}' for i in range(k_star)],
                       columns=sample_metadata['sample_id'])
    C_df.index.name = 'component_id'
    C_df.to_csv(f"04_advanced_mvs/logs/mcr_concentrations__{dataset_slug}__k{k_star}.tsv", sep='\t')
    
    # Save top m/z for each component
    top_mz_data = []
    for comp_idx in range(k_star):
        top_indices = np.argsort(S_best[:, comp_idx])[-20:][::-1]  # Top 20, descending
        for rank, idx in enumerate(top_indices):
            top_mz_data.append({
                'component_id': f'Comp_{comp_idx+1}',
                'rank': rank + 1,
                'mz': mass_mz[idx],
                'weight': S_best[idx, comp_idx]
            })
    
    pd.DataFrame(top_mz_data).to_csv(
        f"04_advanced_mvs/logs/mcr_top_mz__{dataset_slug}__k{k_star}.csv", index=False)
    
    # Compute acceptance criteria
    passed_recon = best_result['r_squared'] >= 0.70
    passed_dose = any(abs(c['rho5']) >= 0.60 and c['p5'] <= 0.05 for c in best_result['correlations'])
    
    # Stability threshold depends on k*
    if k_star == 2:
        passed_stability = stable_fraction >= 0.90 or np.median(component_stabilities) >= 0.90
    else:
        passed_stability = stable_fraction >= 0.67
    
    # Create run log
    run_log = {
        'dataset': dataset_slug,
        'k_star': k_star,
        'n_starts': len(starts),
        'successful_starts': len(successful_results),
        'best_start': best_result['start_name'],
        'convergence_info': [
            {
                'start_idx': int(r['start_idx']),
                'start_name': r['start_name'],
                'converged': bool(r.get('converged', False)),
                'final_residual': float(r.get('final_residual', 0)) if r.get('final_residual') is not None else None,
                'r_squared': float(r.get('r_squared', 0)) if r.get('r_squared') is not None else None,
                'rmse': float(r.get('rmse', 0)) if r.get('rmse') is not None else None
            }
            for r in results
        ],
        'best_metrics': {
            'r_squared': float(best_result['r_squared']),
            'rmse': float(best_result['rmse']),
            'final_residual': float(best_result['final_residual'])
        },
        'component_stabilities': [float(s) for s in component_stabilities],
        'stable_fraction': float(stable_fraction),
        'dose_correlations': best_result['correlations'],
        'passed_recon': bool(passed_recon),
        'passed_dose': bool(passed_dose),
        'passed_stability': bool(passed_stability),
        'timestamp': datetime.now().isoformat() + 'Z'
    }
    
    with open(f"04_advanced_mvs/logs/mcr_run__{dataset_slug}__k{k_star}.json", 'w') as f:
        json.dump(run_log, f, indent=2)
    
    # Create summary report
    create_mcr_summary_report(dataset_slug, k_star, best_result, component_stabilities, 
                             passed_recon, passed_dose, passed_stability)
    
    # Find best dose correlation
    best_rho5 = max(abs(c['rho5']) for c in best_result['correlations'])
    best_comp = max(best_result['correlations'], key=lambda x: abs(x['rho5']))
    
    # Print summary
    print(f"MCR DATASET={dataset_slug}  k*={k_star}  R2={best_result['r_squared']:.3f}  "
          f"stable_frac={stable_fraction:.2f}  best|rho5|={best_rho5:.2f} "
          f"(comp={best_comp['component']+1}, p={best_comp['p5']:.3f})")
    
    return {
        'dataset_slug': dataset_slug,
        'k_star': k_star,
        'best_result': best_result,
        'component_stabilities': component_stabilities,
        'stable_fraction': stable_fraction,
        'acceptance': {
            'passed_recon': passed_recon,
            'passed_dose': passed_dose,
            'passed_stability': passed_stability
        }
    }

def create_mcr_summary_report(dataset_slug, k_star, best_result, component_stabilities,
                             passed_recon, passed_dose, passed_stability):
    """Create MCR-ALS summary report"""
    
    # Find best component for dose correlation
    best_comp = max(best_result['correlations'], key=lambda x: abs(x['rho5']))
    
    # Get top m/z for best component
    top_mz_df = pd.read_csv(f"04_advanced_mvs/logs/mcr_top_mz__{dataset_slug}__k{k_star}.csv")
    best_comp_mz = top_mz_df[top_mz_df['component_id'] == f"Comp_{best_comp['component']+1}"].head(10)
    
    report = f"""# MCR-ALS Summary Report: {dataset_slug}

## Model Configuration

**Components (k*)**: {k_star}
**Initialization**: NMF solution with {N_STARTS} starts (direct + 2 perturbed)
**Best Start**: {best_result['start_name']}
**Convergence**: {best_result['converged']}

## Reconstruction Quality

- **R²** = {best_result['r_squared']:.3f}
- **RMSE** = {best_result['rmse']:.4f}
- **Final Residual** = {best_result['final_residual']:.2e}

## Best Dose-Responsive Component

**Component {best_comp['component']+1}** shows strongest dose correlation:
- ρ₅ (dose means): {best_comp['rho5']:.3f} (p = {best_comp['p5']:.3f})
- ρ (all samples): {best_comp['rho_all']:.3f} (p = {best_comp['p_all']:.3f})

### Top Contributing m/z Values

"""
    
    for _, row in best_comp_mz.iterrows():
        report += f"- **{row['mz']:.1f}** (weight: {row['weight']:.4f})\n"
    
    stable_frac = np.mean(np.array(component_stabilities) >= 0.90)
    
    report += f"""

## Component Stability

Components show stability across multiple starts:
- **Stable fraction** (≥0.90 similarity): {stable_frac:.2f}
- **Individual stabilities**: {', '.join([f'{s:.2f}' for s in component_stabilities])}

## All Component Dose Correlations

| Component | ρ₅ | p-value | ρ_all | p-value |
|-----------|-----|---------|-------|---------|
"""
    
    for corr in best_result['correlations']:
        report += f"| {corr['component']+1} | {corr['rho5']:.3f} | {corr['p5']:.3f} | {corr['rho_all']:.3f} | {corr['p_all']:.3f} |\n"
    
    report += f"""

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | {best_result['r_squared']:.3f} | {'✅ PASS' if passed_recon else '❌ FAIL'} |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | {abs(best_comp['rho5']):.3f} (p={best_comp['p5']:.3f}) | {'✅ PASS' if passed_dose else '❌ FAIL'} |
| Stability (≥67% stable, or ≥90% for k=2) | {stable_frac:.1%} | {'✅ PASS' if passed_stability else '❌ FAIL'} |

## MCR-ALS vs NMF Comparison

MCR-ALS provides chemically interpretable pure component spectra through:
- Non-negativity constraints on both spectra (S) and concentrations (C)
- Alternating least squares optimization for factor separation
- Initialization from NMF solution ensures reproducible results

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f"04_advanced_mvs/reports/MCR_SUMMARY__{dataset_slug}.md", 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("Starting MCR-ALS Analysis...")
    
    # Load inputs manifest and NMF results
    manifest = load_inputs_manifest()
    nmf_selections = load_nmf_selection_results()
    
    if not nmf_selections:
        print("Error: No NMF model selection results found")
        return
    
    # Define datasets to analyze (same as NMF)
    datasets_to_analyze = []
    
    # Primary dataset (required)
    primary_path = manifest['primary_input']
    if 'combined_robust' in nmf_selections:
        k_star = nmf_selections['combined_robust']['k_star']
        datasets_to_analyze.append((primary_path, 'combined_robust', k_star))
    else:
        print("Error: Primary dataset NMF results not found")
        return
    
    # Secondary datasets (optional)
    secondary_datasets = [
        ('matrices_pos.robust_pqn_sqrt_pos', 'pos_robust'),
        ('matrices_neg.robust_pqn_sqrt_neg', 'neg_robust')
    ]
    
    for key_path, slug in secondary_datasets:
        keys = key_path.split('.')
        if (keys[0] in manifest and keys[1] in manifest[keys[0]] and 
            slug in nmf_selections):
            path = manifest[keys[0]][keys[1]]
            k_star = nmf_selections[slug]['k_star']
            datasets_to_analyze.append((path, slug, k_star))
    
    # Sensitivity datasets (optional)
    sensitivity_datasets = [
        ('matrices.combined.baseline_tic_sqrt', 'combined_baseline'),
        ('matrices.combined.robust_pqn_sqrt_pareto', 'combined_robust_pareto')
    ]
    
    for key_path, slug in sensitivity_datasets:
        keys = key_path.split('.')
        if (keys[0] in manifest and keys[1] in manifest[keys[0]] and 
            keys[2] in manifest[keys[0]][keys[1]] and slug in nmf_selections):
            path = manifest[keys[0]][keys[1]][keys[2]]
            k_star = nmf_selections[slug]['k_star']
            datasets_to_analyze.append((path, slug, k_star))
    
    # Analyze each dataset
    all_mcr_results = []
    print(f"Processing {len(datasets_to_analyze)} datasets...")
    for dataset_path, dataset_slug, k_star in tqdm(datasets_to_analyze, desc="MCR-ALS datasets"):
        result = analyze_dataset_mcr(dataset_path, dataset_slug, k_star)
        if result:
            all_mcr_results.append(result)
    
    # Update run manifest
    run_manifest = {
        'schema': 'mcr_run_manifest.v1',
        'timestamp': datetime.now().isoformat() + 'Z',
        'git_commit': manifest['metadata']['git_commit'],
        'datasets_processed': [r['dataset_slug'] for r in all_mcr_results],
        'initialization': 'NMF_solution_based',
        'n_starts': N_STARTS,
        'max_iter': MAX_ITER,
        'tolerance': TOLERANCE,
        'output_files': []
    }
    
    # Collect all output files
    for result in all_mcr_results:
        dataset_slug = result['dataset_slug']
        k_star = result['k_star']
        run_manifest['output_files'].extend([
            f"04_advanced_mvs/figures/mcr_spectra__{dataset_slug}__k{k_star}.png",
            f"04_advanced_mvs/figures/mcr_concentrations_heatmap__{dataset_slug}__k{k_star}.png",
            f"04_advanced_mvs/figures/mcr_reconstruction_fit__{dataset_slug}__k{k_star}.png",
            f"04_advanced_mvs/figures/mcr_dose_trends__{dataset_slug}__k{k_star}.png",
            f"04_advanced_mvs/logs/mcr_run__{dataset_slug}__k{k_star}.json",
            f"04_advanced_mvs/logs/mcr_spectra__{dataset_slug}__k{k_star}.tsv",
            f"04_advanced_mvs/logs/mcr_concentrations__{dataset_slug}__k{k_star}.tsv",
            f"04_advanced_mvs/logs/mcr_top_mz__{dataset_slug}__k{k_star}.csv",
            f"04_advanced_mvs/reports/MCR_SUMMARY__{dataset_slug}.md"
        ])
    
    # Save/update run manifest
    manifest_path = "04_advanced_mvs/manifests/MCR_RUN_MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(run_manifest, f, indent=2)
    
    print(f"\nMCR-ALS complete. Artifacts written: manifests/logs/figures/reports. No new directories created.")
    print(f"Processed {len(all_mcr_results)} datasets with NMF-initialized MCR-ALS")

if __name__ == "__main__":
    main()