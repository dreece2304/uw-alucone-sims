#!/usr/bin/env python3
"""
NLDR Analysis for Phase 4 Advanced MVS
Apply UMAP and t-SNE to Phase-2 matrices for dose clustering visualization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
import os
import re
from datetime import datetime
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Try to import UMAP (may not be available)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available - will skip UMAP analysis")

# Fixed parameters
RANDOM_SEED = 42
UMAP_PARAMS = {
    'n_neighbors': 5,
    'min_dist': 0.1,
    'metric': 'cosine',
    'random_state': RANDOM_SEED,
    'n_components': 2
}
TSNE_PARAMS = {
    'perplexity': 5,
    'metric': 'cosine', 
    'random_state': RANDOM_SEED,
    'n_components': 2,
    'max_iter': 1000,
    'n_iter_without_progress': 300
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
    
    # Data matrix (samples x features) - transpose for NLDR
    X = df.values.T  # Transpose to get samples as rows
    mass_mz = df.index.values
    
    print(f"Dataset {dataset_slug}: {X.shape[0]} samples x {X.shape[1]} features")
    
    return X, mass_mz, sample_metadata

def compute_trustworthiness_score(X_original, X_embedded, n_neighbors=5):
    """Compute trustworthiness score for embedding quality"""
    try:
        score = trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)
        return float(score)
    except Exception as e:
        print(f"Warning: Could not compute trustworthiness: {e}")
        return 0.0

def run_umap_embedding(X, dataset_slug):
    """Run UMAP embedding"""
    if not UMAP_AVAILABLE:
        return None, {}
    
    print(f"Running UMAP for {dataset_slug}...")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run UMAP
        reducer = umap.UMAP(**UMAP_PARAMS)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Compute trustworthiness
        trust_score = compute_trustworthiness_score(X_scaled, X_umap)
        
        metrics = {
            'trustworthiness': trust_score,
            'n_neighbors': UMAP_PARAMS['n_neighbors'],
            'min_dist': UMAP_PARAMS['min_dist'],
            'metric': UMAP_PARAMS['metric']
        }
        
        print(f"  UMAP trustworthiness: {trust_score:.3f}")
        
        return X_umap, metrics
        
    except Exception as e:
        print(f"Error running UMAP: {e}")
        return None, {}

def run_tsne_embedding(X, dataset_slug):
    """Run t-SNE embedding"""
    print(f"Running t-SNE for {dataset_slug}...")
    
    try:
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Adjust perplexity if needed
        n_samples = X.shape[0]
        perplexity = min(TSNE_PARAMS['perplexity'], (n_samples - 1) // 3)
        
        tsne_params = TSNE_PARAMS.copy()
        tsne_params['perplexity'] = perplexity
        
        # Run t-SNE
        reducer = TSNE(**tsne_params)
        X_tsne = reducer.fit_transform(X_scaled)
        
        # Compute trustworthiness
        trust_score = compute_trustworthiness_score(X_scaled, X_tsne)
        
        metrics = {
            'trustworthiness': trust_score,
            'perplexity': perplexity,
            'metric': tsne_params['metric'],
            'kl_divergence': float(reducer.kl_divergence_) if hasattr(reducer, 'kl_divergence_') else None
        }
        
        print(f"  t-SNE trustworthiness: {trust_score:.3f}")
        
        return X_tsne, metrics
        
    except Exception as e:
        print(f"Error running t-SNE: {e}")
        return None, {}

def create_embedding_plot(X_embedded, sample_metadata, method, dataset_slug):
    """Create embedding scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors for doses (gradient from blue to red)
    unique_doses = sorted(sample_metadata['dose_uC'].unique())
    dose_colors = plt.cm.viridis(np.linspace(0, 1, len(unique_doses)))
    dose_color_map = dict(zip(unique_doses, dose_colors))
    
    # Define shapes for patterns
    pattern_markers = {'P1': 'o', 'P2': 's', 'P3': '^'}
    
    # Plot points
    for _, row in sample_metadata.iterrows():
        idx = sample_metadata.index.get_loc(_)
        x, y = X_embedded[idx]
        
        dose = row['dose_uC']
        pattern = row['pattern']
        
        ax.scatter(x, y, 
                  c=[dose_color_map[dose]], 
                  marker=pattern_markers.get(pattern, 'o'),
                  s=100, 
                  alpha=0.8,
                  edgecolors='black',
                  linewidth=0.5,
                  label=f"{pattern}_{dose}" if _ < 3 else "")
    
    # Colorbar for dose
    sm = plt.cm.ScalarMappable(cmap='viridis', 
                              norm=plt.Normalize(vmin=min(unique_doses), vmax=max(unique_doses)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Dose (μC/cm²)')
    
    # Legend for patterns
    legend_elements = []
    for pattern, marker in pattern_markers.items():
        legend_elements.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                        linestyle='None', markersize=8, label=pattern))
    ax.legend(handles=legend_elements, title='Pattern', loc='upper right')
    
    # Labels and title
    ax.set_xlabel(f'{method} 1')
    ax.set_ylabel(f'{method} 2')
    ax.set_title(f'{method} Embedding: {dataset_slug}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    method_lower = method.lower().replace('-', '')
    output_path = f"04_advanced_mvs/figures/nldr_{method_lower}__{dataset_slug}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def assess_dose_progression(X_embedded, sample_metadata):
    """Assess whether dose forms ordered trajectory"""
    # Compute mean position for each dose
    dose_positions = {}
    unique_doses = sorted(sample_metadata['dose_uC'].unique())
    
    for dose in unique_doses:
        dose_mask = sample_metadata['dose_uC'] == dose
        mean_pos = np.mean(X_embedded[dose_mask.values], axis=0)
        dose_positions[dose] = mean_pos
    
    # Compute distances between consecutive doses
    consecutive_distances = []
    for i in range(len(unique_doses) - 1):
        dose1, dose2 = unique_doses[i], unique_doses[i + 1]
        dist = np.linalg.norm(dose_positions[dose2] - dose_positions[dose1])
        consecutive_distances.append(dist)
    
    # Assess if doses form a reasonably ordered path
    mean_consecutive_dist = np.mean(consecutive_distances)
    
    # Also compute spread within each dose
    within_dose_spreads = []
    for dose in unique_doses:
        dose_mask = sample_metadata['dose_uC'] == dose
        dose_points = X_embedded[dose_mask.values]
        if len(dose_points) > 1:
            spread = np.mean([np.linalg.norm(p - dose_positions[dose]) for p in dose_points])
            within_dose_spreads.append(spread)
    
    mean_within_spread = np.mean(within_dose_spreads) if within_dose_spreads else 0.0
    
    # Simple heuristic: if consecutive doses are well-separated compared to within-dose spread
    separation_ratio = mean_consecutive_dist / (mean_within_spread + 1e-6)
    
    if separation_ratio > 2.0:
        trend_assessment = "clear"
    elif separation_ratio > 1.2:
        trend_assessment = "moderate"  
    else:
        trend_assessment = "weak"
    
    return trend_assessment, {
        'consecutive_dist_mean': float(mean_consecutive_dist),
        'within_dose_spread_mean': float(mean_within_spread),
        'separation_ratio': float(separation_ratio)
    }

def assess_pattern_overlap(X_embedded, sample_metadata):
    """Assess overlap between patterns (P1, P2, P3)"""
    patterns = ['P1', 'P2', 'P3']
    pattern_centers = {}
    
    for pattern in patterns:
        pattern_mask = sample_metadata['pattern'] == pattern
        if pattern_mask.sum() > 0:
            center = np.mean(X_embedded[pattern_mask.values], axis=0)
            pattern_centers[pattern] = center
    
    # Compute inter-pattern distances
    inter_distances = []
    pattern_pairs = []
    for i, p1 in enumerate(patterns):
        for p2 in patterns[i+1:]:
            if p1 in pattern_centers and p2 in pattern_centers:
                dist = np.linalg.norm(pattern_centers[p2] - pattern_centers[p1])
                inter_distances.append(dist)
                pattern_pairs.append((p1, p2))
    
    mean_inter_dist = np.mean(inter_distances) if inter_distances else 0.0
    
    # Compute intra-pattern spreads
    intra_spreads = {}
    for pattern in patterns:
        pattern_mask = sample_metadata['pattern'] == pattern
        pattern_points = X_embedded[pattern_mask.values]
        if len(pattern_points) > 1 and pattern in pattern_centers:
            spread = np.mean([np.linalg.norm(p - pattern_centers[pattern]) for p in pattern_points])
            intra_spreads[pattern] = spread
    
    mean_intra_spread = np.mean(list(intra_spreads.values())) if intra_spreads else 0.0
    
    # Assessment
    overlap_ratio = mean_intra_spread / (mean_inter_dist + 1e-6)
    
    if overlap_ratio > 0.8:
        overlap_assessment = "high"
    elif overlap_ratio > 0.4:
        overlap_assessment = "moderate"
    else:
        overlap_assessment = "low"
    
    return overlap_assessment, {
        'inter_pattern_dist_mean': float(mean_inter_dist),
        'intra_pattern_spread_mean': float(mean_intra_spread),
        'overlap_ratio': float(overlap_ratio)
    }

def analyze_dataset_nldr(dataset_path, dataset_slug):
    """Analyze single dataset with NLDR"""
    print(f"\n=== NLDR Analysis: {dataset_slug} ===")
    
    # Load dataset
    X, mass_mz, sample_metadata = load_dataset(dataset_path, dataset_slug)
    if X is None:
        return None
    
    results = {
        'dataset': dataset_slug,
        'n_samples': X.shape[0],
        'n_features': X.shape[1]
    }
    
    umap_trust = 0.0
    tsne_trust = 0.0
    
    # Run UMAP
    if UMAP_AVAILABLE:
        X_umap, umap_metrics = run_umap_embedding(X, dataset_slug)
        if X_umap is not None:
            # Create plot
            create_embedding_plot(X_umap, sample_metadata, 'UMAP', dataset_slug)
            
            # Assess clustering
            dose_trend, dose_metrics = assess_dose_progression(X_umap, sample_metadata)
            pattern_overlap, pattern_metrics = assess_pattern_overlap(X_umap, sample_metadata)
            
            results['umap'] = {
                'metrics': umap_metrics,
                'dose_trend': dose_trend,
                'dose_metrics': dose_metrics,
                'pattern_overlap': pattern_overlap,
                'pattern_metrics': pattern_metrics
            }
            
            umap_trust = umap_metrics.get('trustworthiness', 0.0)
    
    # Run t-SNE
    X_tsne, tsne_metrics = run_tsne_embedding(X, dataset_slug)
    if X_tsne is not None:
        # Create plot
        create_embedding_plot(X_tsne, sample_metadata, 't-SNE', dataset_slug)
        
        # Assess clustering
        dose_trend, dose_metrics = assess_dose_progression(X_tsne, sample_metadata)
        pattern_overlap, pattern_metrics = assess_pattern_overlap(X_tsne, sample_metadata)
        
        results['tsne'] = {
            'metrics': tsne_metrics,
            'dose_trend': dose_trend,
            'dose_metrics': dose_metrics,
            'pattern_overlap': pattern_overlap,
            'pattern_metrics': pattern_metrics
        }
        
        tsne_trust = tsne_metrics.get('trustworthiness', 0.0)
    
    # Save metrics
    results['timestamp'] = datetime.now().isoformat() + 'Z'
    
    metrics_path = f"04_advanced_mvs/logs/nldr_metrics__{dataset_slug}.json"
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary report
    create_nldr_summary_report(dataset_slug, results)
    
    # Determine overall trend assessment
    trend_assessments = []
    if 'umap' in results:
        trend_assessments.append(results['umap']['dose_trend'])
    if 'tsne' in results:
        trend_assessments.append(results['tsne']['dose_trend'])
    
    if trend_assessments:
        # Use the best assessment
        if 'clear' in trend_assessments:
            overall_trend = 'clear'
        elif 'moderate' in trend_assessments:
            overall_trend = 'moderate'
        else:
            overall_trend = 'weak'
    else:
        overall_trend = 'unknown'
    
    # Print summary
    print(f"NLDR DATASET={dataset_slug}  UMAP_trust={umap_trust:.3f}  tSNE_trust={tsne_trust:.3f}  trend={overall_trend}")
    
    # Check trustworthiness warnings
    if umap_trust > 0 and umap_trust < 0.80:
        print(f"Warning: UMAP trustworthiness ({umap_trust:.3f}) below 0.80")
    if tsne_trust > 0 and tsne_trust < 0.80:
        print(f"Warning: t-SNE trustworthiness ({tsne_trust:.3f}) below 0.80")
    
    return results

def create_nldr_summary_report(dataset_slug, results):
    """Create NLDR summary report"""
    
    report = f"""# NLDR Summary Report: {dataset_slug}

## Dataset Overview

- **Samples**: {results['n_samples']}
- **Features**: {results['n_features']}
- **Methods**: {', '.join(['UMAP' if 'umap' in results else '', 't-SNE' if 'tsne' in results else '']).strip(', ')}

"""
    
    if 'umap' in results:
        umap_data = results['umap']
        trust = umap_data['metrics']['trustworthiness']
        report += f"""## UMAP Results

**Trustworthiness**: {trust:.3f} {'✅' if trust >= 0.80 else '⚠️'}

### Dose Progression Analysis
- **Trend Assessment**: {umap_data['dose_trend']}
- **Separation Ratio**: {umap_data['dose_metrics']['separation_ratio']:.2f}
- **Consecutive Dose Distance**: {umap_data['dose_metrics']['consecutive_dist_mean']:.3f}
- **Within-Dose Spread**: {umap_data['dose_metrics']['within_dose_spread_mean']:.3f}

### Pattern Overlap Analysis  
- **Overlap Assessment**: {umap_data['pattern_overlap']}
- **Inter-Pattern Distance**: {umap_data['pattern_metrics']['inter_pattern_dist_mean']:.3f}
- **Intra-Pattern Spread**: {umap_data['pattern_metrics']['intra_pattern_spread_mean']:.3f}

"""
    
    if 'tsne' in results:
        tsne_data = results['tsne']
        trust = tsne_data['metrics']['trustworthiness']
        report += f"""## t-SNE Results

**Trustworthiness**: {trust:.3f} {'✅' if trust >= 0.80 else '⚠️'}

### Dose Progression Analysis
- **Trend Assessment**: {tsne_data['dose_trend']}
- **Separation Ratio**: {tsne_data['dose_metrics']['separation_ratio']:.2f}
- **Consecutive Dose Distance**: {tsne_data['dose_metrics']['consecutive_dist_mean']:.3f}
- **Within-Dose Spread**: {tsne_data['dose_metrics']['within_dose_spread_mean']:.3f}

### Pattern Overlap Analysis
- **Overlap Assessment**: {tsne_data['pattern_overlap']}  
- **Inter-Pattern Distance**: {tsne_data['pattern_metrics']['inter_pattern_dist_mean']:.3f}
- **Intra-Pattern Spread**: {tsne_data['pattern_metrics']['intra_pattern_spread_mean']:.3f}

"""
    
    report += f"""## Qualitative Assessment

### Dose Ordering
"""
    
    # Add qualitative discussion based on results
    dose_trends = []
    if 'umap' in results:
        dose_trends.append(results['umap']['dose_trend'])
    if 'tsne' in results:
        dose_trends.append(results['tsne']['dose_trend'])
    
    if 'clear' in dose_trends:
        report += "The dose progression forms **clear ordered trajectories** in the embedding space. Higher doses are well-separated from lower doses, indicating strong dose-dependent chemical changes.\n\n"
    elif 'moderate' in dose_trends:
        report += "The dose progression shows **moderate ordering** in the embedding space. Some dose-dependent structure is visible but with some overlap between dose levels.\n\n"
    else:
        report += "The dose progression shows **weak ordering** in the embedding space. Limited dose-dependent clustering is observed.\n\n"
    
    report += "### Pattern Separation\n"
    
    pattern_overlaps = []
    if 'umap' in results:
        pattern_overlaps.append(results['umap']['pattern_overlap'])
    if 'tsne' in results:
        pattern_overlaps.append(results['tsne']['pattern_overlap'])
    
    if 'low' in pattern_overlaps:
        report += "The experimental patterns (P1, P2, P3) show **good separation**, indicating reproducible differences between replicates.\n\n"
    elif 'moderate' in pattern_overlaps:
        report += "The experimental patterns (P1, P2, P3) show **moderate overlap**, with some clustering by pattern but also some mixing.\n\n"
    else:
        report += "The experimental patterns (P1, P2, P3) show **high overlap**, suggesting limited reproducibility differences between patterns.\n\n"
    
    report += f"""## Methodology

- **UMAP Parameters**: n_neighbors=5, min_dist=0.1, metric=cosine
- **t-SNE Parameters**: perplexity=5, metric=cosine
- **Preprocessing**: StandardScaler normalization
- **Random Seed**: {RANDOM_SEED} (for reproducibility)

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    with open(f"04_advanced_mvs/reports/NLDR_SUMMARY__{dataset_slug}.md", 'w') as f:
        f.write(report)

def main():
    """Main execution function"""
    print("Starting NLDR Analysis...")
    
    # Load inputs manifest
    manifest = load_inputs_manifest()
    
    # Define datasets to analyze
    datasets_to_analyze = []
    
    # Primary dataset (required)
    primary_path = manifest['primary_input']
    datasets_to_analyze.append((primary_path, 'combined_robust'))
    
    # Optional datasets
    optional_datasets = [
        ('matrices_pos.robust_pqn_sqrt_pos', 'pos_robust'),
        ('matrices_neg.robust_pqn_sqrt_neg', 'neg_robust'),
        ('matrices.combined.baseline_tic_sqrt', 'combined_baseline'),
        ('matrices.combined.robust_pqn_sqrt_pareto', 'combined_robust_pareto')
    ]
    
    for key_path, slug in optional_datasets:
        keys = key_path.split('.')
        try:
            if len(keys) == 2:
                # matrices_pos/matrices_neg case
                if keys[0] in manifest and keys[1] in manifest[keys[0]]:
                    path = manifest[keys[0]][keys[1]]
                    datasets_to_analyze.append((path, slug))
            elif len(keys) == 3:
                # matrices.combined case
                if (keys[0] in manifest and keys[1] in manifest[keys[0]] and 
                    keys[2] in manifest[keys[0]][keys[1]]):
                    path = manifest[keys[0]][keys[1]][keys[2]]
                    datasets_to_analyze.append((path, slug))
        except KeyError:
            print(f"Optional dataset {slug} not found in manifest - skipping")
    
    # Check primary dataset exists
    if not os.path.exists(primary_path):
        print(f"Error: Primary dataset not found: {primary_path}")
        return
    
    # Analyze each dataset
    all_nldr_results = []
    processed_datasets = []
    
    for dataset_path, dataset_slug in datasets_to_analyze:
        result = analyze_dataset_nldr(dataset_path, dataset_slug)
        if result:
            all_nldr_results.append(result)
            processed_datasets.append(dataset_slug)
    
    if not all_nldr_results:
        print("Error: No datasets could be processed")
        return
    
    # Create run manifest
    run_manifest = {
        'schema': 'nldr_run_manifest.v1',
        'timestamp': datetime.now().isoformat() + 'Z',
        'git_commit': manifest['metadata']['git_commit'],
        'datasets_processed': processed_datasets,
        'methods': ['UMAP' if UMAP_AVAILABLE else None, 't-SNE'],
        'methods': [m for m in ['UMAP' if UMAP_AVAILABLE else None, 't-SNE'] if m],
        'parameters': {
            'umap': UMAP_PARAMS if UMAP_AVAILABLE else None,
            'tsne': TSNE_PARAMS,
            'random_seed': RANDOM_SEED
        },
        'output_files': []
    }
    
    # Collect all output files
    for dataset_slug in processed_datasets:
        if UMAP_AVAILABLE:
            run_manifest['output_files'].append(f"04_advanced_mvs/figures/nldr_umap__{dataset_slug}.png")
        run_manifest['output_files'].append(f"04_advanced_mvs/figures/nldr_tsne__{dataset_slug}.png")
        run_manifest['output_files'].append(f"04_advanced_mvs/logs/nldr_metrics__{dataset_slug}.json")
        run_manifest['output_files'].append(f"04_advanced_mvs/reports/NLDR_SUMMARY__{dataset_slug}.md")
    
    # Save run manifest
    with open("04_advanced_mvs/manifests/NLDR_RUN_MANIFEST.json", 'w') as f:
        json.dump(run_manifest, f, indent=2)
    
    print(f"\nNLDR complete. Artifacts written: manifests/logs/figures/reports. No new directories created.")
    print(f"Processed {len(processed_datasets)} datasets with {', '.join(run_manifest['methods'])}")

if __name__ == "__main__":
    main()