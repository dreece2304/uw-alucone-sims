#!/usr/bin/env python3
"""
Phase 5: Non-Negative Matrix Factorization Analysis
==================================================

Implementation of NMF analysis following literature best practices:
- Gardner et al. (2022): NMF for chemically interpretable components in ToF-SIMS
- Non-negative constraints provide more realistic chemical factors
- Multiple initialization strategies and factorization ranks

Non-Negative Matrix Factorization (NMF):
- Decomposes data into non-negative factors W and H: X ≈ WH^T
- W: sample loadings (non-negative), H: feature loadings (non-negative)
- More chemically interpretable than PCA (no negative intensities)
- Useful for spectral unmixing and source separation

Features:
- Multiple NMF algorithms (multiplicative update, coordinate descent)
- Component number optimization (2-10 factors)
- Comparison with PCA results  
- Chemical interpretability assessment
- Dose-response analysis of NMF components

Author: Claude Code Assistant
Date: 2025-08-23
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
import pickle
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class NMFAnalysis:
    """Non-Negative Matrix Factorization analysis for ToF-SIMS data."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.norm_dir = self.base_dir / '02_preprocessing' / 'normalized'
        self.pca_dir = self.base_dir / '03_pca_analysis' / 'baseline'
        self.nmf_dir = self.base_dir / '04_advanced_mvs' / 'nmf'
        self.meta_dir = self.base_dir / '01_raw_data' / 'metadata'
        
        # Create output directory
        self.nmf_dir.mkdir(parents=True, exist_ok=True)
        
        # Load preprocessing results and PCA results  
        self.load_preprocessing_stats()
        self.load_pca_results()
        
        # NMF results storage
        self.nmf_results = {}
        
        # NMF algorithms to test
        self.nmf_algorithms = {
            'multiplicative': {'solver': 'mu', 'description': 'Multiplicative Update'},
            'coordinate_descent': {'solver': 'cd', 'description': 'Coordinate Descent'}
        }
        
        # Component range to test
        self.component_range = range(2, 11)  # 2-10 components
        
    def load_preprocessing_stats(self):
        """Load preprocessing statistics to get recommended methods."""
        print("Loading preprocessing statistics...")
        
        with open(self.base_dir / '02_preprocessing' / 'phase2_preprocessing_stats.json', 'r') as f:
            self.prep_stats = json.load(f)
        
        self.recommended_methods = {
            'positive': self.prep_stats['positive']['recommended_method'],
            'negative': self.prep_stats['negative']['recommended_method']
        }
        
        print(f"✓ Recommended methods: {self.recommended_methods}")
    
    def load_pca_results(self):
        """Load PCA results for comparison."""
        print("Loading PCA results for comparison...")
        
        self.pca_results = {}
        for polarity in ['positive', 'negative']:
            # Load PCA summary
            with open(self.pca_dir / f'{polarity}_pca_summary.json', 'r') as f:
                pca_summary = json.load(f)
            
            # Load PCA scores and loadings
            scores_df = pd.read_csv(self.pca_dir / f'{polarity}_pca_scores.tsv', sep='\t', index_col=0)
            loadings_df = pd.read_csv(self.pca_dir / f'{polarity}_pca_loadings.tsv', sep='\t', index_col=0)
            
            self.pca_results[polarity] = {
                'summary': pca_summary,
                'scores': scores_df,
                'loadings': loadings_df
            }
        
        print("✓ PCA results loaded for comparison")
    
    def load_preprocessed_data(self, polarity, method_name):
        """Load specific preprocessed dataset."""
        data_file = self.norm_dir / polarity / f"{method_name}_normalized.tsv"
        meta_file = self.norm_dir / polarity / f"{method_name}_metadata.json"
        
        # Load data
        df = pd.read_csv(data_file, sep='\t', index_col=0)
        
        # Load metadata
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Separate mass data from sample info
        mass_cols = [col for col in df.columns if col.startswith('m')]
        sample_cols = [col for col in df.columns if not col.startswith('m')]
        
        data_matrix = df[mass_cols].values
        sample_info = df[sample_cols]
        masses = [float(col[1:]) for col in mass_cols]  # Remove 'm' prefix
        
        return {
            'data': data_matrix,
            'masses': np.array(masses),
            'samples': sample_info,
            'metadata': metadata
        }
    
    def prepare_data_for_nmf(self, data_matrix):
        """Prepare data for NMF (ensure non-negativity)."""
        # NMF requires non-negative data
        # Since preprocessing may have mean-centered data (negative values),
        # we need to shift to non-negative range
        
        # Shift data to make all values non-negative
        min_val = np.min(data_matrix)
        if min_val < 0:
            shifted_data = data_matrix - min_val
            print(f"  Shifted data by {-min_val:.3f} to ensure non-negativity")
        else:
            shifted_data = data_matrix
            print("  Data already non-negative")
        
        # Additional scaling to improve NMF convergence
        max_val = np.max(shifted_data)
        if max_val > 0:
            scaled_data = shifted_data / max_val
        else:
            scaled_data = shifted_data
        
        return scaled_data, {'min_shift': min_val, 'max_scale': max_val}
    
    def optimize_nmf_components(self, data, algorithm='multiplicative', max_components=10):
        """Optimize number of NMF components using reconstruction error."""
        print(f"  Optimizing NMF components using {algorithm}...")
        
        reconstruction_errors = []
        frobenius_norms = []
        component_numbers = list(range(2, max_components + 1))
        
        for n_comp in component_numbers:
            # Fit NMF
            nmf_model = NMF(
                n_components=n_comp,
                solver=self.nmf_algorithms[algorithm]['solver'],
                random_state=RANDOM_STATE,
                max_iter=1000,
                alpha_W=0.01,  # L1 regularization for sparsity
                alpha_H=0.01
            )
            
            try:
                W = nmf_model.fit_transform(data)
                H = nmf_model.components_
                
                # Calculate reconstruction error
                reconstruction = np.dot(W, H)
                mse = mean_squared_error(data, reconstruction)
                reconstruction_errors.append(mse)
                
                # Calculate Frobenius norm (NMF objective)
                frobenius_norms.append(nmf_model.reconstruction_err_)
                
            except Exception as e:
                print(f"    NMF failed for {n_comp} components: {e}")
                reconstruction_errors.append(np.inf)
                frobenius_norms.append(np.inf)
        
        # Find optimal number using elbow method
        if len(reconstruction_errors) > 0 and not all(np.isinf(reconstruction_errors)):
            # Calculate rate of change in reconstruction error
            changes = np.diff(reconstruction_errors)
            if len(changes) > 1:
                # Find elbow point (where improvement diminishes)
                elbow_idx = np.argmax(changes[:-1] - changes[1:])
                optimal_components = component_numbers[elbow_idx + 1]
            else:
                optimal_components = component_numbers[0]
        else:
            optimal_components = 3  # Default fallback
        
        return {
            'component_numbers': component_numbers,
            'reconstruction_errors': reconstruction_errors,
            'frobenius_norms': frobenius_norms,
            'optimal_components': optimal_components
        }
    
    def run_nmf_analysis(self, data, n_components, algorithm='multiplicative'):
        """Run NMF analysis with specified parameters."""
        nmf_model = NMF(
            n_components=n_components,
            solver=self.nmf_algorithms[algorithm]['solver'],
            random_state=RANDOM_STATE,
            max_iter=1000,
            alpha_W=0.01,  # L1 regularization
            alpha_H=0.01,
            init='nndsvd'  # Better initialization
        )
        
        try:
            # Fit NMF
            W = nmf_model.fit_transform(data)  # Sample loadings
            H = nmf_model.components_        # Feature loadings
            
            # Calculate reconstruction quality
            reconstruction = np.dot(W, H)
            reconstruction_error = mean_squared_error(data, reconstruction)
            
            # Calculate explained variance (approximation)
            total_var = np.var(data)
            residual_var = np.var(data - reconstruction)
            explained_variance_ratio = 1 - (residual_var / total_var) if total_var > 0 else 0
            
            # Component sparsity (percentage of near-zero values)
            sparsity_W = np.sum(W < 0.01) / W.size
            sparsity_H = np.sum(H < 0.01) / H.size
            
            return {
                'model': nmf_model,
                'W': W,  # Sample loadings (15 samples x n_components)
                'H': H,  # Feature loadings (n_components x n_masses)
                'reconstruction_error': reconstruction_error,
                'explained_variance_ratio': explained_variance_ratio,
                'n_components': n_components,
                'sparsity_W': sparsity_W,
                'sparsity_H': sparsity_H,
                'algorithm': algorithm
            }
            
        except Exception as e:
            print(f"    NMF failed: {e}")
            return None
    
    def analyze_dose_response_nmf(self, nmf_result, sample_info):
        """Analyze dose-response relationships in NMF components."""
        W = nmf_result['W']  # Sample loadings
        doses = sample_info['dose'].values
        
        dose_correlations = {}
        for i in range(W.shape[1]):
            component_scores = W[:, i]
            
            # Spearman correlation with dose
            corr, p_val = stats.spearmanr(component_scores, doses)
            
            dose_correlations[f'NMF{i+1}'] = {
                'rho': corr,
                'p': p_val,
                'mean_loading': np.mean(component_scores),
                'std_loading': np.std(component_scores)
            }
        
        return dose_correlations
    
    def compare_nmf_with_pca(self, polarity, nmf_results):
        """Compare NMF results with PCA for the same polarity."""
        pca_data = self.pca_results[polarity]
        
        comparison = {
            'pca_dose_correlation_pc1': float(pca_data['summary']['dose_correlations']['PC1']['rho']),
            'pca_explained_variance_total': sum(pca_data['summary']['explained_variance_first_5'][:3])
        }
        
        # Find best NMF algorithm and component number
        best_nmf = None
        best_score = -np.inf
        
        for alg_name, alg_results in nmf_results.items():
            for n_comp, result in alg_results['results'].items():
                if result is not None:
                    # Score based on dose correlation + explained variance
                    max_dose_corr = max([abs(corr['rho']) for corr in result['dose_correlations'].values()])
                    score = max_dose_corr + result['explained_variance_ratio']
                    
                    if score > best_score:
                        best_score = score
                        best_nmf = {
                            'algorithm': alg_name,
                            'n_components': n_comp,
                            'result': result
                        }
        
        if best_nmf:
            best_dose_corr = max([abs(corr['rho']) for corr in best_nmf['result']['dose_correlations'].values()])
            comparison.update({
                'best_nmf_algorithm': best_nmf['algorithm'],
                'best_nmf_n_components': best_nmf['n_components'],
                'best_nmf_dose_correlation': best_dose_corr,
                'best_nmf_explained_variance': best_nmf['result']['explained_variance_ratio'],
                'nmf_vs_pca_correlation_improvement': best_dose_corr - abs(comparison['pca_dose_correlation_pc1']),
                'nmf_vs_pca_sparsity_W': best_nmf['result']['sparsity_W'],
                'nmf_vs_pca_sparsity_H': best_nmf['result']['sparsity_H']
            })
        
        return comparison
    
    def run_complete_nmf_analysis(self, polarity):
        """Run comprehensive NMF analysis for one polarity."""
        print(f"\\n{'='*60}")
        print(f"NMF ANALYSIS - {polarity.upper()}")
        print(f"{'='*60}")
        
        # Load recommended preprocessed data
        method_name = self.recommended_methods[polarity]
        print(f"Using recommended method: {method_name}")
        
        dataset = self.load_preprocessed_data(polarity, method_name)
        data_matrix = dataset['data']
        masses = dataset['masses']
        sample_info = dataset['samples']
        
        print(f"Data shape: {data_matrix.shape}")
        print(f"Mass range: {masses.min():.1f} - {masses.max():.1f} u")
        
        # Prepare data for NMF
        nmf_data, scaling_info = self.prepare_data_for_nmf(data_matrix)
        
        # Test different NMF algorithms
        nmf_results = {}
        
        for alg_name, alg_config in self.nmf_algorithms.items():
            print(f"\\nTesting {alg_config['description']} algorithm...")
            
            # Optimize number of components
            optimization_result = self.optimize_nmf_components(nmf_data, alg_name, max_components=8)
            optimal_n = optimization_result['optimal_components']
            
            print(f"  Optimal components: {optimal_n}")
            
            # Test multiple component numbers around optimum
            test_components = [max(2, optimal_n-1), optimal_n, min(8, optimal_n+1)]
            
            alg_results = {
                'optimization': optimization_result,
                'results': {}
            }
            
            for n_comp in test_components:
                print(f"  Running NMF with {n_comp} components...")
                
                nmf_result = self.run_nmf_analysis(nmf_data, n_comp, alg_name)
                
                if nmf_result is not None:
                    # Analyze dose response
                    dose_correlations = self.analyze_dose_response_nmf(nmf_result, sample_info)
                    nmf_result['dose_correlations'] = dose_correlations
                    
                    # Store masses for later use
                    nmf_result['masses'] = masses
                    nmf_result['scaling_info'] = scaling_info
                    
                    print(f"    Reconstruction error: {nmf_result['reconstruction_error']:.4f}")
                    print(f"    Explained variance: {nmf_result['explained_variance_ratio']:.1%}")
                    
                    best_corr = max([abs(corr['rho']) for corr in dose_correlations.values()])
                    print(f"    Best dose correlation: {best_corr:.3f}")
                
                alg_results['results'][n_comp] = nmf_result
            
            nmf_results[alg_name] = alg_results
        
        # Compare with PCA
        pca_comparison = self.compare_nmf_with_pca(polarity, nmf_results)
        
        # Store results
        self.nmf_results[polarity] = {
            'method_name': method_name,
            'nmf_algorithms': nmf_results,
            'pca_comparison': pca_comparison,
            'dataset_info': {
                'n_samples': len(sample_info),
                'n_masses': len(masses),
                'mass_range': [float(masses.min()), float(masses.max())],
                'sample_info': sample_info,
                'scaling_info': scaling_info
            }
        }
        
        print(f"\\n✓ NMF analysis complete for {polarity}")
        return self.nmf_results[polarity]
    
    def create_nmf_visualizations(self, polarity):
        """Create comprehensive NMF visualizations."""
        print(f"\\nCreating NMF visualizations for {polarity}...")
        
        result = self.nmf_results[polarity]
        sample_info = result['dataset_info']['sample_info']
        doses = sample_info['dose'].values
        
        # Create comparison plot: PCA vs NMF algorithms
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{polarity.title()} Ion: NMF vs PCA Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Plot PCA (reference)
        pca_scores = self.pca_results[polarity]['scores'][[col for col in self.pca_results[polarity]['scores'].columns if col.startswith('PC')]].iloc[:, :2]
        
        scatter0 = axes[0].scatter(pca_scores.iloc[:, 0], pca_scores.iloc[:, 1], c=doses, cmap='viridis', s=60, alpha=0.8)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('PCA Reference')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter0, ax=axes[0], label='Dose (μC/cm²)')
        
        # Plot best NMF results for each algorithm
        plot_idx = 1
        for alg_name, alg_data in result['nmf_algorithms'].items():
            # Find best result for this algorithm
            best_result = None
            best_score = -np.inf
            
            for n_comp, nmf_res in alg_data['results'].items():
                if nmf_res is not None:
                    max_corr = max([abs(corr['rho']) for corr in nmf_res['dose_correlations'].values()])
                    score = max_corr + nmf_res['explained_variance_ratio']
                    if score > best_score:
                        best_score = score
                        best_result = nmf_res
            
            if best_result is not None and plot_idx < len(axes):
                W = best_result['W']
                scatter = axes[plot_idx].scatter(W[:, 0], W[:, 1] if W.shape[1] > 1 else np.zeros_like(W[:, 0]), 
                                               c=doses, cmap='viridis', s=60, alpha=0.8)
                axes[plot_idx].set_xlabel('NMF1')
                axes[plot_idx].set_ylabel('NMF2' if W.shape[1] > 1 else 'Zero')
                axes[plot_idx].set_title(f'{alg_name.title()} NMF\\n({best_result["n_components"]} components)')
                axes[plot_idx].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[plot_idx], label='Dose (μC/cm²)')
                plot_idx += 1
        
        # Component optimization plot
        if plot_idx < len(axes):
            ax_opt = axes[plot_idx]
            for alg_name, alg_data in result['nmf_algorithms'].items():
                opt_data = alg_data['optimization']
                comp_nums = opt_data['component_numbers']
                errors = opt_data['reconstruction_errors']
                
                # Only plot if we have valid errors
                valid_mask = ~np.isinf(errors)
                if np.any(valid_mask):
                    ax_opt.plot(np.array(comp_nums)[valid_mask], np.array(errors)[valid_mask], 
                               'o-', label=alg_name, alpha=0.8)
            
            ax_opt.set_xlabel('Number of Components')
            ax_opt.set_ylabel('Reconstruction Error')
            ax_opt.set_title('Component Optimization')
            ax_opt.legend()
            ax_opt.grid(True, alpha=0.3)
            plot_idx += 1
        
        # Comparison metrics plot
        if plot_idx < len(axes):
            ax_comp = axes[plot_idx]
            
            # Comparison data
            pca_corr = abs(result['pca_comparison']['pca_dose_correlation_pc1'])
            nmf_corr = result['pca_comparison'].get('best_nmf_dose_correlation', 0)
            
            methods = ['PCA', 'Best NMF']
            correlations = [pca_corr, nmf_corr]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax_comp.bar(methods, correlations, color=colors, alpha=0.8)
            ax_comp.set_ylabel('|Dose Correlation|')
            ax_comp.set_title('PCA vs NMF Performance')
            ax_comp.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, corr in zip(bars, correlations):
                height = bar.get_height()
                ax_comp.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.nmf_dir / f'{polarity}_nmf_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ NMF analysis plot saved: {polarity}_nmf_analysis.png")
    
    def save_nmf_results(self, polarity):
        """Save NMF results and analysis."""
        print(f"\\nSaving NMF results for {polarity}...")
        
        result = self.nmf_results[polarity]
        
        # Find and save best NMF result
        best_result = self.find_best_nmf_result(polarity)
        
        if best_result:
            # Save NMF loadings (W matrix - sample loadings)
            nmf_scores_df = pd.DataFrame(
                best_result['W'],
                columns=[f'NMF{i+1}' for i in range(best_result['W'].shape[1])],
                index=result['dataset_info']['sample_info'].index
            )
            nmf_scores_df = pd.concat([nmf_scores_df, result['dataset_info']['sample_info']], axis=1)
            nmf_scores_df.to_csv(self.nmf_dir / f'{polarity}_nmf_scores.tsv', sep='\t')
            
            # Save NMF components (H matrix - feature loadings)
            nmf_loadings_df = pd.DataFrame(
                best_result['H'],
                columns=[f'm{mass:.3f}' for mass in best_result['masses']],
                index=[f'NMF{i+1}' for i in range(best_result['H'].shape[0])]
            )
            nmf_loadings_df.to_csv(self.nmf_dir / f'{polarity}_nmf_loadings.tsv', sep='\t')
        
        # Save comprehensive summary
        summary = {
            'method_used': result['method_name'],
            'pca_vs_nmf_comparison': result['pca_comparison'],
            'algorithm_performance': {
                alg_name: {
                    'optimal_components': alg_data['optimization']['optimal_components'],
                    'best_reconstruction_error': min([r['reconstruction_error'] for r in alg_data['results'].values() if r is not None], default=np.inf)
                }
                for alg_name, alg_data in result['nmf_algorithms'].items()
            },
            'dataset_info': {k: v for k, v in result['dataset_info'].items() if k != 'sample_info'}  # Exclude sample_info for JSON
        }
        
        with open(self.nmf_dir / f'{polarity}_nmf_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ NMF results saved to {self.nmf_dir}")
    
    def find_best_nmf_result(self, polarity):
        """Find the best NMF result based on combined criteria."""
        result = self.nmf_results[polarity]
        
        best_result = None
        best_score = -np.inf
        
        for alg_name, alg_data in result['nmf_algorithms'].items():
            for n_comp, nmf_res in alg_data['results'].items():
                if nmf_res is not None:
                    # Combined score: dose correlation + explained variance - reconstruction error
                    max_dose_corr = max([abs(corr['rho']) for corr in nmf_res['dose_correlations'].values()])
                    score = max_dose_corr + nmf_res['explained_variance_ratio'] - nmf_res['reconstruction_error']
                    
                    if score > best_score:
                        best_score = score
                        best_result = nmf_res
        
        return best_result
    
    def run_complete_analysis(self):
        """Run complete NMF analysis."""
        print("="*70)
        print("ToF-SIMS NON-NEGATIVE MATRIX FACTORIZATION ANALYSIS")
        print("="*70)
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run NMF analysis for both polarities
            for polarity in ['positive', 'negative']:
                # Core NMF analysis
                self.run_complete_nmf_analysis(polarity)
                
                # Create visualizations
                self.create_nmf_visualizations(polarity)
                
                # Save results
                self.save_nmf_results(polarity)
            
            print("\\n" + "="*70)
            print("✅ PHASE 5 NON-NEGATIVE MATRIX FACTORIZATION COMPLETE")
            print("="*70)
            
            # Summary
            for polarity in ['positive', 'negative']:
                result = self.nmf_results[polarity]
                pca_comp = result['pca_comparison']
                
                print(f"\\n🔬 {polarity.upper()} RESULTS:")
                print(f"   Method: {result['method_name']}")
                if 'best_nmf_algorithm' in pca_comp:
                    print(f"   Best NMF: {pca_comp['best_nmf_algorithm']} ({pca_comp['best_nmf_n_components']} components)")
                    print(f"   PCA dose correlation: {pca_comp['pca_dose_correlation_pc1']:.3f}")
                    print(f"   NMF dose correlation: {pca_comp['best_nmf_dose_correlation']:.3f}")
                    print(f"   Improvement: {pca_comp['nmf_vs_pca_correlation_improvement']:+.3f}")
                    print(f"   Sparsity: W={pca_comp['nmf_vs_pca_sparsity_W']:.1%}, H={pca_comp['nmf_vs_pca_sparsity_H']:.1%}")
            
            print(f"\\n📁 All results saved to: {self.nmf_dir}")
            
        except Exception as e:
            print(f"❌ ERROR in NMF analysis: {str(e)}")
            raise

def main():
    """Main execution function."""
    analyzer = NMFAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()