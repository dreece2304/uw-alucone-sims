#!/usr/bin/env python3
"""
Phase 3: Standard PCA Analysis
=============================

Implementation of comprehensive PCA analysis following literature best practices:
- Graham & Castner (2012): Score plots, loading analysis, biplot visualization
- Zhou et al. (2024): Automated reporting with 95% confidence ellipses  
- Gardner et al. (2022): Component selection and chemical interpretation
- Literature-guided approach with scree plots and Hotelling's T² outlier detection

Features:
- Component selection via scree plot (capture >90% variance)
- Score plots with 95% confidence ellipses
- Loading analysis for chemical interpretation
- Biplot generation for integrated visualization
- Outlier detection using Hotelling's T²

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
from scipy.stats import chi2
import pickle
warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class StandardPCAAnalysis:
    """Comprehensive standard PCA analysis for ToF-SIMS data."""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.norm_dir = self.base_dir / '02_preprocessing' / 'normalized'
        self.pca_dir = self.base_dir / '03_pca_analysis' / 'baseline'
        self.meta_dir = self.base_dir / '01_raw_data' / 'metadata'
        
        # Create output directory
        self.pca_dir.mkdir(parents=True, exist_ok=True)
        
        # Load preprocessing results
        self.load_preprocessing_stats()
        
        # PCA results storage
        self.pca_results = {}
        
    def load_preprocessing_stats(self):
        """Load preprocessing statistics to identify best methods."""
        print("Loading preprocessing statistics...")
        
        with open(self.base_dir / '02_preprocessing' / 'phase2_preprocessing_stats.json', 'r') as f:
            self.prep_stats = json.load(f)
        
        # Get recommended methods
        self.recommended_methods = {
            'positive': self.prep_stats['positive']['recommended_method'],
            'negative': self.prep_stats['negative']['recommended_method']
        }
        
        print(f"✓ Recommended methods: {self.recommended_methods}")
    
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
    
    def select_components_scree(self, pca_model, variance_threshold=0.90):
        """Select components using scree plot to capture >90% variance."""
        explained_var = pca_model.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        # Find number of components for 90% variance
        n_components_90 = np.where(cumulative_var >= variance_threshold)[0][0] + 1
        
        return n_components_90, explained_var, cumulative_var
    
    def calculate_confidence_ellipses(self, scores, groups, confidence=0.95):
        """Calculate 95% confidence ellipses for groups in PC space."""
        ellipses = {}
        unique_groups = np.unique(groups)
        
        chi2_val = chi2.ppf(confidence, 2)  # 2 DOF for 2D ellipse
        
        for group in unique_groups:
            group_mask = groups == group
            group_scores = scores[group_mask]
            
            if len(group_scores) < 2:
                continue
            
            # Calculate mean and covariance
            mean = np.mean(group_scores, axis=0)
            cov = np.cov(group_scores.T)
            
            # Eigendecomposition for ellipse parameters
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Ellipse parameters
            width = 2 * np.sqrt(chi2_val * eigenvals[0])
            height = 2 * np.sqrt(chi2_val * eigenvals[1])
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            
            ellipses[group] = {
                'center': mean,
                'width': width,
                'height': height,
                'angle': angle
            }
        
        return ellipses
    
    def detect_outliers_hotelling(self, scores, alpha=0.05):
        """Detect outliers using Hotelling's T² test."""
        n_samples, n_components = scores.shape
        
        # Calculate T² statistics
        mean_scores = np.mean(scores, axis=0)
        cov_scores = np.cov(scores.T)
        
        try:
            inv_cov = np.linalg.inv(cov_scores)
        except:
            inv_cov = np.linalg.pinv(cov_scores)
        
        t_squared = []
        for i in range(n_samples):
            diff = scores[i] - mean_scores
            t2 = n_samples * np.dot(diff, np.dot(inv_cov, diff))
            t_squared.append(t2)
        
        # Critical value for outlier detection
        f_critical = stats.f.ppf(1 - alpha, n_components, n_samples - n_components)
        t2_critical = (n_samples - 1) * n_components / (n_samples - n_components) * f_critical
        
        outliers = np.array(t_squared) > t2_critical
        
        return np.array(t_squared), outliers, t2_critical
    
    def run_pca_analysis(self, polarity):
        """Run comprehensive PCA analysis for one polarity."""
        print(f"\\n{'='*60}")
        print(f"STANDARD PCA ANALYSIS - {polarity.upper()}")
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
        
        # Fit PCA with all components first
        pca_full = PCA()
        scores_full = pca_full.fit_transform(data_matrix)
        
        # Component selection via scree plot
        n_components_90, explained_var, cumulative_var = self.select_components_scree(pca_full)
        print(f"\\nComponents for 90% variance: {n_components_90}")
        print(f"Explained variance (first 5): {explained_var[:5]}")
        
        # Fit PCA with selected components (minimum 3 for analysis)
        pca = PCA(n_components=min(max(n_components_90, 3), 10))  # Min 3, Max 10 for visualization
        scores = pca.fit_transform(data_matrix)
        loadings = pca.components_.T
        
        # Dose correlation analysis
        doses = sample_info['dose'].values
        pc1_dose_corr, pc1_p = stats.spearmanr(scores[:, 0], doses)
        pc2_dose_corr, pc2_p = stats.spearmanr(scores[:, 1], doses)
        
        print(f"\\nDose correlations:")
        print(f"PC1-dose: ρ={pc1_dose_corr:.3f}, p={pc1_p:.2e}")
        print(f"PC2-dose: ρ={pc2_dose_corr:.3f}, p={pc2_p:.2e}")
        
        # Pattern effect analysis
        patterns = sample_info['pattern'].values
        pattern_separation = self.analyze_pattern_effects(scores, patterns)
        
        # Calculate confidence ellipses
        dose_groups = sample_info['dose'].values
        ellipses_dose = self.calculate_confidence_ellipses(scores[:, :2], dose_groups)
        ellipses_pattern = self.calculate_confidence_ellipses(scores[:, :2], patterns)
        
        # Outlier detection
        t_squared, outliers, t2_critical = self.detect_outliers_hotelling(scores[:, :min(5, scores.shape[1])])
        n_outliers = np.sum(outliers)
        print(f"\\nOutlier detection: {n_outliers}/{len(outliers)} samples flagged")
        
        # Store results
        self.pca_results[polarity] = {
            'method_name': method_name,
            'pca_model': pca,
            'scores': scores,
            'loadings': loadings,
            'masses': masses,
            'sample_info': sample_info,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'n_components_90': n_components_90,
            'dose_correlations': {
                'PC1': {'rho': pc1_dose_corr, 'p': pc1_p},
                'PC2': {'rho': pc2_dose_corr, 'p': pc2_p}
            },
            'pattern_effects': pattern_separation,
            'confidence_ellipses': {'dose': ellipses_dose, 'pattern': ellipses_pattern},
            'outliers': {'t_squared': t_squared, 'outlier_flags': outliers, 'critical': t2_critical}
        }
        
        print(f"✓ PCA analysis complete for {polarity}")
        return self.pca_results[polarity]
    
    def analyze_pattern_effects(self, scores, patterns):
        """Analyze separation between patterns in PC space."""
        pattern_stats = {}
        unique_patterns = np.unique(patterns)
        
        for pattern in unique_patterns:
            mask = patterns == pattern
            pattern_scores = scores[mask]
            
            pattern_stats[pattern] = {
                'mean_pc1': np.mean(pattern_scores[:, 0]),
                'mean_pc2': np.mean(pattern_scores[:, 1]),
                'std_pc1': np.std(pattern_scores[:, 0]),
                'std_pc2': np.std(pattern_scores[:, 1]),
                'n_samples': np.sum(mask)
            }
        
        # Calculate pattern separation (distance between centroids)
        separations = {}
        for i, p1 in enumerate(unique_patterns):
            for p2 in unique_patterns[i+1:]:
                dist = np.sqrt(
                    (pattern_stats[p1]['mean_pc1'] - pattern_stats[p2]['mean_pc1'])**2 + 
                    (pattern_stats[p1]['mean_pc2'] - pattern_stats[p2]['mean_pc2'])**2
                )
                separations[f"{p1}_vs_{p2}"] = dist
        
        return {'pattern_stats': pattern_stats, 'separations': separations}
    
    def create_scree_plot(self, polarity):
        """Create scree plot for component selection."""
        result = self.pca_results[polarity]
        explained_var = result['explained_variance']
        cumulative_var = result['cumulative_variance']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'{polarity.title()} Ion PCA - Component Selection', fontsize=14, fontweight='bold')
        
        # Individual explained variance
        components = np.arange(1, len(explained_var[:20]) + 1)
        ax1.bar(components, explained_var[:20], alpha=0.7, color='skyblue')
        ax1.axhline(y=0.05, color='red', linestyle='--', label='5% threshold')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Explained Variance Ratio')
        ax1.set_title('Scree Plot - Individual Variance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax2.plot(components, cumulative_var[:20], 'o-', color='darkblue', linewidth=2)
        ax2.axhline(y=0.90, color='red', linestyle='--', label='90% threshold')
        ax2.axvline(x=result['n_components_90'], color='green', linestyle='--', 
                   label=f'{result["n_components_90"]} components')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Explained Variance')
        ax2.set_title('Cumulative Variance Explained')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.pca_dir / f'{polarity}_scree_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scree plot saved: {polarity}_scree_plot.png")
    
    def create_score_plots(self, polarity):
        """Create comprehensive PCA score plots with confidence ellipses."""
        result = self.pca_results[polarity]
        scores = result['scores']
        sample_info = result['sample_info']
        ellipses_dose = result['confidence_ellipses']['dose']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{polarity.title()} Ion PCA - Score Plots', fontsize=16, fontweight='bold')
        
        # PC1 vs PC2 colored by dose
        doses = sample_info['dose'].values
        scatter1 = ax1.scatter(scores[:, 0], scores[:, 1], c=doses, cmap='viridis', 
                              s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Add confidence ellipses for doses
        from matplotlib.patches import Ellipse
        colors_dose = plt.cm.viridis(np.linspace(0, 1, len(ellipses_dose)))
        for i, (dose, ellipse_data) in enumerate(ellipses_dose.items()):
            ellipse = Ellipse(ellipse_data['center'], ellipse_data['width'], ellipse_data['height'],
                             angle=ellipse_data['angle'], alpha=0.3, color=colors_dose[i])
            ax1.add_patch(ellipse)
        
        ax1.set_xlabel(f'PC1 ({result["explained_variance"][0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
        ax1.set_title('Scores by Dose with 95% Confidence Ellipses')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Dose (μC/cm²)')
        
        # PC1 vs PC2 colored by pattern
        patterns = sample_info['pattern'].values
        pattern_colors = {'P1': 'red', 'P2': 'blue', 'P3': 'green'}
        for pattern in np.unique(patterns):
            mask = patterns == pattern
            ax2.scatter(scores[mask, 0], scores[mask, 1], c=pattern_colors.get(pattern, 'gray'),
                       label=pattern, s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel(f'PC1 ({result["explained_variance"][0]:.1%} variance)')
        ax2.set_ylabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
        ax2.set_title('Scores by Pattern')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # PC2 vs PC3 if available
        if scores.shape[1] > 2:
            scatter3 = ax3.scatter(scores[:, 1], scores[:, 2], c=doses, cmap='viridis', 
                                  s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            ax3.set_xlabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
            ax3.set_ylabel(f'PC3 ({result["explained_variance"][2]:.1%} variance)')
            ax3.set_title('PC2 vs PC3 by Dose')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax3, label='Dose (μC/cm²)')
        
        # Outlier detection plot
        t_squared = result['outliers']['t_squared']
        outlier_flags = result['outliers']['outlier_flags']
        colors = ['red' if flag else 'blue' for flag in outlier_flags]
        
        ax4.scatter(range(len(t_squared)), t_squared, c=colors, s=60, alpha=0.8)
        ax4.axhline(y=result['outliers']['critical'], color='red', linestyle='--', 
                   label=f'Critical value ({result["outliers"]["critical"]:.2f})')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel("Hotelling's T²")
        ax4.set_title(f'Outlier Detection ({np.sum(outlier_flags)} outliers)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.pca_dir / f'{polarity}_score_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Score plots saved: {polarity}_score_plots.png")
    
    def create_loading_plots(self, polarity):
        """Create loading plots for chemical interpretation."""
        result = self.pca_results[polarity]
        loadings = result['loadings']
        masses = result['masses']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{polarity.title()} Ion PCA - Loading Analysis', fontsize=16, fontweight='bold')
        axes = axes.flatten()
        
        # Plot first 4 PCs loadings
        for i in range(min(4, loadings.shape[1])):
            ax = axes[i]
            
            # Loading plot
            ax.plot(masses, loadings[:, i], alpha=0.7, linewidth=1)
            ax.fill_between(masses, loadings[:, i], alpha=0.3)
            
            # Highlight significant loadings
            threshold = 2 * np.std(loadings[:, i])
            significant_mask = np.abs(loadings[:, i]) > threshold
            if np.any(significant_mask):
                ax.scatter(masses[significant_mask], loadings[:, i][significant_mask], 
                          c='red', s=30, zorder=5, alpha=0.8)
            
            ax.set_xlabel('Mass (u)')
            ax.set_ylabel(f'PC{i+1} Loading')
            ax.set_title(f'PC{i+1} ({result["explained_variance"][i]:.1%} variance)')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.pca_dir / f'{polarity}_loading_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loading plots saved: {polarity}_loading_plots.png")
    
    def create_biplots(self, polarity):
        """Create PCA biplots combining scores and loadings."""
        result = self.pca_results[polarity]
        scores = result['scores']
        loadings = result['loadings']
        masses = result['masses']
        sample_info = result['sample_info']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'{polarity.title()} Ion PCA - Biplots', fontsize=16, fontweight='bold')
        
        # Biplot 1: PC1 vs PC2
        # Plot scores
        doses = sample_info['dose'].values
        scatter = ax1.scatter(scores[:, 0], scores[:, 1], c=doses, cmap='viridis', 
                             s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Plot significant loadings as arrows
        loading_scale = 0.8 * np.max(np.abs(scores[:, :2]))
        threshold = 2 * np.std(loadings[:, 0])  # Threshold for significant loadings
        
        significant_indices = np.where(
            (np.abs(loadings[:, 0]) > threshold) | 
            (np.abs(loadings[:, 1]) > threshold)
        )[0][:20]  # Show top 20 most significant
        
        for idx in significant_indices:
            ax1.arrow(0, 0, loadings[idx, 0] * loading_scale, loadings[idx, 1] * loading_scale,
                     head_width=loading_scale*0.05, head_length=loading_scale*0.05, 
                     fc='red', ec='red', alpha=0.6)
            ax1.text(loadings[idx, 0] * loading_scale * 1.1, 
                    loadings[idx, 1] * loading_scale * 1.1,
                    f'{masses[idx]:.0f}', fontsize=8, alpha=0.8)
        
        ax1.set_xlabel(f'PC1 ({result["explained_variance"][0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
        ax1.set_title('PC1 vs PC2 Biplot')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Dose (μC/cm²)')
        
        # Biplot 2: PC2 vs PC3 if available
        if scores.shape[1] > 2:
            scatter2 = ax2.scatter(scores[:, 1], scores[:, 2], c=doses, cmap='viridis', 
                                  s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
            
            # Loading arrows for PC2 vs PC3
            threshold_2 = 2 * np.std(loadings[:, 1])
            significant_indices_2 = np.where(
                (np.abs(loadings[:, 1]) > threshold_2) | 
                (np.abs(loadings[:, 2]) > threshold_2)
            )[0][:20]
            
            for idx in significant_indices_2:
                ax2.arrow(0, 0, loadings[idx, 1] * loading_scale, loadings[idx, 2] * loading_scale,
                         head_width=loading_scale*0.05, head_length=loading_scale*0.05,
                         fc='red', ec='red', alpha=0.6)
                ax2.text(loadings[idx, 1] * loading_scale * 1.1,
                        loadings[idx, 2] * loading_scale * 1.1,
                        f'{masses[idx]:.0f}', fontsize=8, alpha=0.8)
            
            ax2.set_xlabel(f'PC2 ({result["explained_variance"][1]:.1%} variance)')
            ax2.set_ylabel(f'PC3 ({result["explained_variance"][2]:.1%} variance)')
            ax2.set_title('PC2 vs PC3 Biplot')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='Dose (μC/cm²)')
        
        plt.tight_layout()
        plt.savefig(self.pca_dir / f'{polarity}_biplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Biplots saved: {polarity}_biplots.png")
    
    def save_pca_results(self, polarity):
        """Save PCA results and generate reports."""
        print(f"\\nSaving PCA results for {polarity}...")
        
        result = self.pca_results[polarity]
        
        # Save scores
        scores_df = pd.DataFrame(
            result['scores'], 
            columns=[f'PC{i+1}' for i in range(result['scores'].shape[1])],
            index=result['sample_info'].index
        )
        scores_df = pd.concat([scores_df, result['sample_info']], axis=1)
        scores_df.to_csv(self.pca_dir / f'{polarity}_pca_scores.tsv', sep='\t')
        
        # Save loadings
        loadings_df = pd.DataFrame(
            result['loadings'],
            index=[f'm{mass:.3f}' for mass in result['masses']],
            columns=[f'PC{i+1}' for i in range(result['loadings'].shape[1])]
        )
        loadings_df.to_csv(self.pca_dir / f'{polarity}_pca_loadings.tsv', sep='\t')
        
        # Save model and results
        with open(self.pca_dir / f'{polarity}_pca_model.pkl', 'wb') as f:
            pickle.dump(result['pca_model'], f)
        
        # Save summary statistics
        summary = {
            'method_used': result['method_name'],
            'n_components_90_variance': result['n_components_90'],
            'explained_variance_first_5': result['explained_variance'][:5].tolist(),
            'cumulative_variance_first_5': result['cumulative_variance'][:5].tolist(),
            'dose_correlations': result['dose_correlations'],
            'n_outliers': int(np.sum(result['outliers']['outlier_flags'])),
            'pattern_separations': result['pattern_effects']['separations']
        }
        
        with open(self.pca_dir / f'{polarity}_pca_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✓ Results saved to {self.pca_dir}")
    
    def run_complete_analysis(self):
        """Run complete standard PCA analysis."""
        print("="*70)
        print("ToF-SIMS STANDARD PCA ANALYSIS")
        print("="*70)
        print(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Run PCA for both polarities
            for polarity in ['positive', 'negative']:
                # Core PCA analysis
                self.run_pca_analysis(polarity)
                
                # Create visualizations
                self.create_scree_plot(polarity)
                self.create_score_plots(polarity)
                self.create_loading_plots(polarity)
                self.create_biplots(polarity)
                
                # Save results
                self.save_pca_results(polarity)
            
            print("\\n" + "="*70)
            print("✅ PHASE 3 STANDARD PCA ANALYSIS COMPLETE")
            print("="*70)
            
            # Summary
            for polarity in ['positive', 'negative']:
                result = self.pca_results[polarity]
                print(f"\\n🔬 {polarity.upper()} RESULTS:")
                print(f"   Method: {result['method_name']}")
                print(f"   Components (90% var): {result['n_components_90']}")
                print(f"   PC1-dose correlation: ρ={result['dose_correlations']['PC1']['rho']:.3f}")
                print(f"   Outliers detected: {np.sum(result['outliers']['outlier_flags'])}")
            
            print(f"\\n📁 All results saved to: {self.pca_dir}")
            
        except Exception as e:
            print(f"❌ ERROR in PCA analysis: {str(e)}")
            raise

def main():
    """Main execution function."""
    analyzer = StandardPCAAnalysis()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()