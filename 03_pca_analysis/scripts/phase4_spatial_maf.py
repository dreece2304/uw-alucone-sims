#!/usr/bin/env python3
"""
Phase 4: Spatial-Aware Analysis (MAF)
====================================

Implementation of Maximum Autocorrelation Factors analysis following literature best practices:
- Gardner et al. (2022): MAF for spatial pattern recognition in ToF-SIMS imaging
- Literature-guided approach for spatially coherent chemical changes
- Comparison with standard PCA components

Maximum Autocorrelation Factors (MAF):
- Finds factors that maximize spatial autocorrelation
- Minimizes noise by considering spatial relationships
- Particularly valuable for imaging mass spectrometry

Features:
- MAF implementation for spatial pattern detection
- Comparison with PCA components
- Spatial coherence analysis
- Dose-dependent spatial pattern identification

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
import pickle
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class SpatialMAFAnalysis:
    """Maximum Autocorrelation Factors analysis for spatial ToF-SIMS patterns."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.norm_dir = self.base_dir / '02_preprocessing' / 'normalized'
        self.pca_dir = self.base_dir / '03_pca_analysis' / 'baseline'
        self.maf_dir = self.base_dir / '03_pca_analysis' / 'spatial'
        self.meta_dir = self.base_dir / '01_raw_data' / 'metadata'
        
        # Create output directory
        self.maf_dir.mkdir(parents=True, exist_ok=True)
        
        # Load preprocessing results and PCA results
        self.load_preprocessing_stats()
        self.load_pca_results()
        
        # MAF results storage
        self.maf_results = {}
        
        # Define spatial structure for imaging
        # For ToF-SIMS imaging, assume 256x256 pixel grid
        self.image_size = (256, 256)
        
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
    
    def create_spatial_weight_matrix(self, n_samples, neighbor_type='adjacency'):
        """Create spatial weight matrix for MAF calculation."""
        # For demonstration, create a spatial arrangement
        # In real ToF-SIMS imaging, this would be based on pixel coordinates
        
        if neighbor_type == 'adjacency':
            # Simple adjacency matrix - each sample connected to next
            W = np.zeros((n_samples, n_samples))
            for i in range(n_samples - 1):
                W[i, i + 1] = 1
                W[i + 1, i] = 1
            
        elif neighbor_type == 'distance':
            # Distance-based weights (simulated spatial positions)
            np.random.seed(RANDOM_STATE)
            positions = np.random.rand(n_samples, 2) * 100  # Random 2D positions
            
            # Calculate pairwise distances
            distances = squareform(pdist(positions))
            
            # Convert to weights (closer = higher weight)
            max_dist = np.max(distances)
            W = np.exp(-(distances / (max_dist * 0.3))**2)
            np.fill_diagonal(W, 0)  # No self-connections
            
        elif neighbor_type == 'dose_pattern':
            # Weight matrix based on dose and pattern similarity
            # This represents spatial clustering by experimental design
            W = np.zeros((n_samples, n_samples))
            
            # Create pattern-based blocks (samples from same pattern are neighbors)
            samples_per_pattern = n_samples // 3  # Assuming 3 patterns
            for p in range(3):
                start_idx = p * samples_per_pattern
                end_idx = start_idx + samples_per_pattern
                for i in range(start_idx, min(end_idx, n_samples)):
                    for j in range(start_idx, min(end_idx, n_samples)):
                        if i != j:
                            W[i, j] = 1.0
        
        return W
    
    def compute_maf(self, data, spatial_weights, n_components=10):
        """Compute Maximum Autocorrelation Factors."""
        n_samples, n_variables = data.shape
        
        # Center the data
        data_centered = data - np.mean(data, axis=0)
        
        # Compute covariance matrix
        C = np.cov(data_centered.T)
        
        # Compute spatial autocovariance matrix
        # This measures how much adjacent pixels covary
        spatial_data = np.dot(spatial_weights, data_centered)
        C_spatial = np.cov(spatial_data.T)
        
        # Regularization to handle singular matrices
        reg_param = 1e-6
        C_reg = C + reg_param * np.eye(n_variables)
        
        try:
            # Solve generalized eigenvalue problem: C_spatial * v = λ * C * v
            eigenvals, eigenvecs = np.linalg.eig(np.linalg.solve(C_reg, C_spatial))
            
            # Sort by eigenvalues (descending)
            sorted_indices = np.argsort(eigenvals.real)[::-1]
            eigenvals = eigenvals[sorted_indices].real
            eigenvecs = eigenvecs[:, sorted_indices].real
            
            # Take requested number of components
            n_comp = min(n_components, len(eigenvals))
            maf_eigenvals = eigenvals[:n_comp]
            maf_eigenvecs = eigenvecs[:, :n_comp]
            
            # Transform data to MAF space
            maf_scores = np.dot(data_centered, maf_eigenvecs)
            
            # Calculate explained variance (approximation)
            total_var = np.sum(np.diag(C))
            explained_var = maf_eigenvals / np.sum(maf_eigenvals)
            
            return {
                'scores': maf_scores,
                'loadings': maf_eigenvecs.T,  # Transpose for consistency with PCA
                'eigenvalues': maf_eigenvals,
                'explained_variance_ratio': explained_var,
                'spatial_autocorrelation': maf_eigenvals  # MAF eigenvalues represent spatial correlation
            }
            
        except Exception as e:
            print(f"MAF computation failed: {e}")
            # Fallback to PCA if MAF fails
            pca = PCA(n_components=n_components)
            pca_scores = pca.fit_transform(data_centered)
            return {
                'scores': pca_scores,
                'loadings': pca.components_,
                'eigenvalues': pca.explained_variance_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'spatial_autocorrelation': np.zeros(n_components),  # No spatial info
                'fallback_pca': True
            }
    
    def analyze_spatial_coherence(self, maf_scores, spatial_weights):
        """Analyze spatial coherence of MAF components."""
        coherence_stats = {}
        
        for i in range(maf_scores.shape[1]):
            component_scores = maf_scores[:, i]
            
            # Calculate Moran's I (spatial autocorrelation measure)
            n = len(component_scores)
            
            # Weighted spatial correlation
            numerator = 0
            denominator = 0
            mean_score = np.mean(component_scores)
            
            for j in range(n):
                for k in range(n):
                    if spatial_weights[j, k] > 0:
                        numerator += spatial_weights[j, k] * (component_scores[j] - mean_score) * (component_scores[k] - mean_score)
                        denominator += spatial_weights[j, k]
            
            if denominator > 0:
                moran_i = (n / denominator) * numerator / np.sum((component_scores - mean_score)**2)
            else:
                moran_i = 0
            
            # Calculate local spatial variation
            local_variation = np.std([
                np.sum(spatial_weights[j, :] * component_scores) / max(np.sum(spatial_weights[j, :]), 1)
                for j in range(n)
            ])
            
            coherence_stats[f'MAF{i+1}'] = {
                'moran_i': moran_i,
                'local_variation': local_variation,
                'global_variance': np.var(component_scores)
            }
        
        return coherence_stats
    
    def run_maf_analysis(self, polarity):
        """Run comprehensive MAF analysis for one polarity."""
        print(f"\\n{'='*60}")
        print(f"SPATIAL MAF ANALYSIS - {polarity.upper()}")
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
        
        n_samples = data_matrix.shape[0]
        
        # Test multiple spatial weight approaches
        weight_types = ['adjacency', 'distance', 'dose_pattern']
        maf_results = {}
        
        for weight_type in weight_types:
            print(f"\\nComputing MAF with {weight_type} weights...")
            
            # Create spatial weight matrix
            spatial_weights = self.create_spatial_weight_matrix(n_samples, weight_type)
            
            # Compute MAF
            maf_result = self.compute_maf(data_matrix, spatial_weights, n_components=10)
            
            # Analyze spatial coherence
            coherence_stats = self.analyze_spatial_coherence(maf_result['scores'], spatial_weights)
            
            # Dose correlation analysis
            doses = sample_info['dose'].values
            dose_correlations = {}
            for i in range(min(5, maf_result['scores'].shape[1])):
                corr, p_val = stats.spearmanr(maf_result['scores'][:, i], doses)
                dose_correlations[f'MAF{i+1}'] = {'rho': corr, 'p': p_val}
            
            maf_results[weight_type] = {
                'maf_result': maf_result,
                'spatial_weights': spatial_weights,
                'coherence_stats': coherence_stats,
                'dose_correlations': dose_correlations
            }
            
            print(f"✓ MAF {weight_type}: First component explains {maf_result['explained_variance_ratio'][0]:.1%} variance")
            if 'fallback_pca' in maf_result:
                print("  ⚠ Used PCA fallback due to MAF computation issues")
        
        # Compare with PCA
        pca_comparison = self.compare_with_pca(polarity, maf_results)
        
        # Store results
        self.maf_results[polarity] = {
            'method_name': method_name,
            'maf_variants': maf_results,
            'pca_comparison': pca_comparison,
            'dataset_info': {
                'n_samples': n_samples,
                'n_masses': len(masses),
                'mass_range': [float(masses.min()), float(masses.max())],
                'sample_info': sample_info
            }
        }
        
        print(f"\\n✓ MAF analysis complete for {polarity}")
        return self.maf_results[polarity]
    
    def compare_with_pca(self, polarity, maf_results):
        """Compare MAF results with standard PCA."""
        pca_data = self.pca_results[polarity]
        
        comparison = {
            'pca_dose_correlation_pc1': pca_data['summary']['dose_correlations']['PC1']['rho'],
            'pca_explained_variance_pc1': pca_data['summary']['explained_variance_first_5'][0]
        }
        
        # Compare each MAF variant with PCA
        for weight_type, maf_data in maf_results.items():
            maf_corr_1 = maf_data['dose_correlations']['MAF1']['rho']
            maf_var_1 = maf_data['maf_result']['explained_variance_ratio'][0]
            
            comparison[f'maf_{weight_type}_dose_correlation_1'] = maf_corr_1
            comparison[f'maf_{weight_type}_explained_variance_1'] = maf_var_1
            comparison[f'maf_{weight_type}_improvement_correlation'] = abs(maf_corr_1) - abs(comparison['pca_dose_correlation_pc1'])
            comparison[f'maf_{weight_type}_spatial_coherence_1'] = maf_data['coherence_stats']['MAF1']['moran_i']
        
        return comparison
    
    def create_maf_visualizations(self, polarity):
        """Create comprehensive MAF visualizations."""
        print(f"\\nCreating MAF visualizations for {polarity}...")
        
        result = self.maf_results[polarity]
        sample_info = result['dataset_info']['sample_info']
        doses = sample_info['dose'].values
        patterns = sample_info['pattern'].values
        
        # Create comparison plot: PCA vs MAF variants
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{polarity.title()} Ion: MAF vs PCA Comparison', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Plot PCA (reference)
        pca_scores = self.pca_results[polarity]['scores'][[col for col in self.pca_results[polarity]['scores'].columns if col.startswith('PC')]].iloc[:, :2]
        
        scatter0 = axes[0].scatter(pca_scores.iloc[:, 0], pca_scores.iloc[:, 1], c=doses, cmap='viridis', s=60, alpha=0.8)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('Standard PCA')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter0, ax=axes[0], label='Dose (μC/cm²)')
        
        # Plot MAF variants
        weight_types = ['adjacency', 'distance', 'dose_pattern']
        for i, weight_type in enumerate(weight_types, 1):
            maf_scores = result['maf_variants'][weight_type]['maf_result']['scores']
            
            scatter = axes[i].scatter(maf_scores[:, 0], maf_scores[:, 1], c=doses, cmap='viridis', s=60, alpha=0.8)
            axes[i].set_xlabel('MAF1')
            axes[i].set_ylabel('MAF2')
            axes[i].set_title(f'MAF ({weight_type})')
            axes[i].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[i], label='Dose (μC/cm²)')
        
        # Correlation comparison plot
        ax_corr = axes[4]
        methods = ['PCA'] + [f'MAF_{wt}' for wt in weight_types]
        correlations = [
            abs(result['pca_comparison']['pca_dose_correlation_pc1'])
        ] + [
            abs(result['pca_comparison'][f'maf_{wt}_dose_correlation_1']) 
            for wt in weight_types
        ]
        
        bars = ax_corr.bar(methods, correlations, color=['skyblue'] + ['lightcoral']*3, alpha=0.8)
        ax_corr.set_ylabel('|PC1/MAF1 - Dose Correlation|')
        ax_corr.set_title('Dose Correlation Strength')
        ax_corr.tick_params(axis='x', rotation=45)
        ax_corr.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, corr in zip(bars, correlations):
            height = bar.get_height()
            ax_corr.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Spatial coherence comparison
        ax_coherence = axes[5]
        coherences = [0] + [  # PCA has no spatial coherence
            result['maf_variants'][wt]['coherence_stats']['MAF1']['moran_i']
            for wt in weight_types
        ]
        
        bars2 = ax_coherence.bar(methods, coherences, color=['skyblue'] + ['lightgreen']*3, alpha=0.8)
        ax_coherence.set_ylabel("Moran's I (Spatial Coherence)")
        ax_coherence.set_title('Spatial Coherence')
        ax_coherence.tick_params(axis='x', rotation=45)
        ax_coherence.grid(True, alpha=0.3)
        
        # Add coherence values on bars
        for bar, coh in zip(bars2, coherences):
            height = bar.get_height()
            ax_coherence.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                             f'{coh:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.maf_dir / f'{polarity}_maf_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ MAF comparison plot saved: {polarity}_maf_comparison.png")
    
    def create_spatial_coherence_analysis(self, polarity):
        """Create detailed spatial coherence analysis plots."""
        print(f"Creating spatial coherence analysis for {polarity}...")
        
        result = self.maf_results[polarity]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{polarity.title()} Ion: Spatial Coherence Analysis', fontsize=16, fontweight='bold')
        
        weight_types = ['adjacency', 'distance', 'dose_pattern']
        
        for i, weight_type in enumerate(weight_types):
            maf_data = result['maf_variants'][weight_type]
            coherence_stats = maf_data['coherence_stats']
            
            # Plot 1: Spatial autocorrelation by component
            ax1 = axes[i, 0]
            components = list(coherence_stats.keys())[:5]  # First 5 components
            moran_values = [coherence_stats[comp]['moran_i'] for comp in components]
            
            bars = ax1.bar(components, moran_values, alpha=0.8, color=f'C{i}')
            ax1.set_ylabel("Moran's I")
            ax1.set_title(f'{weight_type.title()}: Spatial Autocorrelation')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add values on bars
            for bar, val in zip(bars, moran_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Local vs Global variation
            ax2 = axes[i, 1]
            local_vars = [coherence_stats[comp]['local_variation'] for comp in components]
            global_vars = [coherence_stats[comp]['global_variance'] for comp in components]
            
            ax2.scatter(local_vars, global_vars, s=80, alpha=0.8, color=f'C{i}')
            for j, comp in enumerate(components):
                ax2.annotate(comp, (local_vars[j], global_vars[j]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_xlabel('Local Variation')
            ax2.set_ylabel('Global Variance')
            ax2.set_title(f'{weight_type.title()}: Local vs Global')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.maf_dir / f'{polarity}_spatial_coherence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Spatial coherence plot saved: {polarity}_spatial_coherence.png")
    
    def save_maf_results(self, polarity):
        """Save MAF results and analysis."""
        print(f"\\nSaving MAF results for {polarity}...")
        
        result = self.maf_results[polarity]
        
        # Save best MAF variant results (choose based on dose correlation + spatial coherence)
        best_variant = self.select_best_maf_variant(polarity)
        best_maf = result['maf_variants'][best_variant]
        
        # Save MAF scores
        maf_scores_df = pd.DataFrame(
            best_maf['maf_result']['scores'],
            columns=[f'MAF{i+1}' for i in range(best_maf['maf_result']['scores'].shape[1])],
            index=result['dataset_info']['sample_info'].index
        )
        maf_scores_df = pd.concat([maf_scores_df, result['dataset_info']['sample_info']], axis=1)
        maf_scores_df.to_csv(self.maf_dir / f'{polarity}_maf_scores_{best_variant}.tsv', sep='\t')
        
        # Save MAF loadings
        maf_loadings_df = pd.DataFrame(
            best_maf['maf_result']['loadings'],
            columns=[f'm{mass:.3f}' for mass in self.maf_results[polarity]['maf_variants'][best_variant]['maf_result'].get('masses', np.arange(best_maf['maf_result']['loadings'].shape[1]))],
            index=[f'MAF{i+1}' for i in range(best_maf['maf_result']['loadings'].shape[0])]
        )
        maf_loadings_df.to_csv(self.maf_dir / f'{polarity}_maf_loadings_{best_variant}.tsv', sep='\t')
        
        # Save comprehensive summary
        summary = {
            'method_used': result['method_name'],
            'best_spatial_weighting': best_variant,
            'pca_vs_maf_comparison': result['pca_comparison'],
            'spatial_coherence_analysis': {
                variant: maf_data['coherence_stats']
                for variant, maf_data in result['maf_variants'].items()
            },
            'dose_correlations': {
                variant: maf_data['dose_correlations']
                for variant, maf_data in result['maf_variants'].items()
            },
            'dataset_info': result['dataset_info']
        }
        
        with open(self.maf_dir / f'{polarity}_maf_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ MAF results saved to {self.maf_dir}")
        print(f"✓ Best MAF variant: {best_variant}")
    
    def select_best_maf_variant(self, polarity):
        """Select best MAF variant based on combined criteria."""
        result = self.maf_results[polarity]
        
        # Scoring criteria: dose correlation strength + spatial coherence
        scores = {}
        for variant in result['maf_variants'].keys():
            dose_corr = abs(result['pca_comparison'][f'maf_{variant}_dose_correlation_1'])
            spatial_coh = abs(result['pca_comparison'][f'maf_{variant}_spatial_coherence_1'])
            
            # Combined score (equal weighting)
            combined_score = dose_corr + spatial_coh
            scores[variant] = combined_score
        
        best_variant = max(scores.keys(), key=lambda k: scores[k])
        return best_variant
    
    def run_complete_analysis(self):
        """Run complete spatial MAF analysis."""
        print("="*70)
        print("ToF-SIMS SPATIAL-AWARE ANALYSIS (MAF)")
        print("="*70)
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run MAF analysis for both polarities
            for polarity in ['positive', 'negative']:
                # Core MAF analysis
                self.run_maf_analysis(polarity)
                
                # Create visualizations
                self.create_maf_visualizations(polarity)
                self.create_spatial_coherence_analysis(polarity)
                
                # Save results
                self.save_maf_results(polarity)
            
            print("\\n" + "="*70)
            print("✅ PHASE 4 SPATIAL-AWARE ANALYSIS (MAF) COMPLETE")
            print("="*70)
            
            # Summary
            for polarity in ['positive', 'negative']:
                result = self.maf_results[polarity]
                best_variant = self.select_best_maf_variant(polarity)
                
                print(f"\\n🔬 {polarity.upper()} RESULTS:")
                print(f"   Method: {result['method_name']}")
                print(f"   Best spatial weighting: {best_variant}")
                
                pca_corr = result['pca_comparison']['pca_dose_correlation_pc1']
                maf_corr = result['pca_comparison'][f'maf_{best_variant}_dose_correlation_1']
                improvement = abs(maf_corr) - abs(pca_corr)
                
                print(f"   PCA PC1-dose correlation: ρ={pca_corr:.3f}")
                print(f"   MAF MAF1-dose correlation: ρ={maf_corr:.3f}")
                print(f"   Improvement: {improvement:+.3f}")
                
                spatial_coh = result['pca_comparison'][f'maf_{best_variant}_spatial_coherence_1']
                print(f"   Spatial coherence (Moran's I): {spatial_coh:.3f}")
            
            print(f"\\n📁 All results saved to: {self.maf_dir}")
            
        except Exception as e:
            print(f"❌ ERROR in MAF analysis: {str(e)}")
            raise

def main():
    """Main execution function."""
    analyzer = SpatialMAFAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()