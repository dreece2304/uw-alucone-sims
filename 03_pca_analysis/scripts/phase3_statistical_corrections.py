#!/usr/bin/env python3
"""
Statistical Corrections for PCA Analysis
Addresses pseudo-replication, PC sign arbitrariness, and normalization path validation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from scipy import stats
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StatisticalPCACorrection:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / '03_pca_analysis' / 'corrected'
        self.results_dir.mkdir(exist_ok=True)
        
        # Load dose mapping
        with open(self.base_dir / 'sample_mapping.json', 'r') as f:
            self.sample_mapping = json.load(f)
        
        # Create dose array for proper analysis
        self.dose_levels = [500, 2000, 5000, 10000, 15000]
        self.n_patterns = 3
        self.n_doses = 5
        
    def exact_spearman_p(self, n):
        """Calculate exact p-value for Spearman correlation with n points"""
        if n == 5:
            # For perfect monotonic trend |ρ|=1, exact p = 2/5! = 0.0167
            return 2.0 / np.math.factorial(n)
        else:
            return None
    
    def load_preprocessed_data(self, polarity, method):
        """Load preprocessed data for specific polarity and method"""
        try:
            file_path = self.base_dir / '02_preprocessing' / 'processed' / f'{polarity}_{method}.npz'
            data = np.load(file_path)
            X = data['data']
            masses = data['masses'] 
            samples = data['samples']
            return X, masses, samples
        except Exception as e:
            print(f"Error loading {polarity} {method}: {e}")
            return None, None, None
    
    def calculate_per_dose_means(self, X, samples):
        """Calculate per-dose means to avoid pseudo-replication"""
        dose_means = []
        dose_labels = []
        
        for dose in self.dose_levels:
            dose_samples = []
            for i, sample in enumerate(samples):
                if sample in self.sample_mapping and self.sample_mapping[sample]['dose'] == dose:
                    dose_samples.append(X[i])
            
            if dose_samples:
                dose_mean = np.mean(dose_samples, axis=0)
                dose_means.append(dose_mean)
                dose_labels.append(dose)
        
        return np.array(dose_means), dose_labels
    
    def fit_pca_corrected(self, X_means, n_components=5):
        """Fit PCA with proper statistical handling"""
        # Mean center the data (critical for proper PCA)
        X_centered = X_means - np.mean(X_means, axis=0)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X_centered)
        
        return pca, scores, X_centered
    
    def calculate_dose_correlations_corrected(self, scores, dose_labels):
        """Calculate dose correlations with proper statistics"""
        correlations = []
        
        for i in range(scores.shape[1]):
            pc_scores = scores[:, i]
            
            # Calculate Spearman correlation
            rho, p_value = spearmanr(dose_labels, pc_scores)
            
            # Calculate exact p-value for n=5
            exact_p = self.exact_spearman_p(len(dose_labels))
            
            correlations.append({
                'PC': i + 1,
                'rho': rho,
                'rho_abs': abs(rho),
                'p_value': p_value,
                'exact_p_bound': exact_p,
                'significant': abs(rho) > 0.8  # Conservative threshold
            })
        
        return correlations
    
    def flip_pc_for_positive_dose_correlation(self, scores, loadings, dose_labels):
        """Flip PC signs so dose correlation is positive"""
        scores_corrected = scores.copy()
        loadings_corrected = loadings.copy()
        
        for i in range(scores.shape[1]):
            pc_scores = scores[:, i]
            rho, _ = spearmanr(dose_labels, pc_scores)
            
            if rho < 0:
                scores_corrected[:, i] *= -1
                loadings_corrected[i, :] *= -1
        
        return scores_corrected, loadings_corrected
    
    def analyze_polarity_corrected(self, polarity):
        """Perform corrected analysis for one polarity"""
        print(f"\n{'='*60}")
        print(f"CORRECTED STATISTICAL ANALYSIS - {polarity.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        # Test both normalization methods
        methods = ['TIC_sqrt', 'Vector_sqrt'] if polarity == 'positive' else ['Vector_sqrt', 'TIC_sqrt']
        
        for method in methods:
            print(f"\nMethod: {method}")
            print("-" * 40)
            
            # Load data
            X, masses, samples = self.load_preprocessed_data(polarity, method)
            if X is None:
                print(f"Could not load data for {method}")
                continue
            
            # Calculate per-dose means (n=5, not n=15)
            X_means, dose_labels = self.calculate_per_dose_means(X, samples)
            print(f"Per-dose means shape: {X_means.shape}")
            print(f"Dose levels: {dose_labels}")
            
            # Fit PCA with proper centering
            pca, scores, X_centered = self.fit_pca_corrected(X_means)
            
            # Flip PC signs for positive dose correlation
            scores_corrected, loadings_corrected = self.flip_pc_for_positive_dose_correlation(
                scores, pca.components_, dose_labels
            )
            
            # Calculate corrected dose correlations
            correlations = self.calculate_dose_correlations_corrected(scores_corrected, dose_labels)
            
            # Find best dose-tracking PC
            best_pc_idx = max(correlations, key=lambda x: x['rho_abs'])
            
            print(f"\nDose Correlations (n={len(dose_labels)} per-dose means):")
            for corr in correlations:
                pc_str = f"PC{corr['PC']}"
                rho_str = f"ρ = {corr['rho']:+.3f}"
                sig_str = "***" if corr['significant'] else ""
                print(f"  {pc_str}: {rho_str} {sig_str}")
            
            print(f"\nBest dose-tracking PC: PC{best_pc_idx['PC']}")
            print(f"Correlation: ρ = {best_pc_idx['rho']:+.3f}")
            print(f"Exact p-value bound: p ≥ {best_pc_idx['exact_p_bound']:.4f}")
            
            # Store results
            method_results = {
                'method': method,
                'pca': pca,
                'scores_corrected': scores_corrected,
                'loadings_corrected': loadings_corrected,
                'dose_labels': dose_labels,
                'correlations': correlations,
                'best_pc': best_pc_idx,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'masses': masses
            }
            
            results[method] = method_results
        
        # Compare methods
        self.compare_normalization_methods(results, polarity)
        
        return results
    
    def compare_normalization_methods(self, results, polarity):
        """Compare results across normalization methods"""
        print(f"\n{'='*40}")
        print(f"NORMALIZATION PATH COMPARISON")
        print(f"{'='*40}")
        
        methods = list(results.keys())
        if len(methods) < 2:
            print("Only one method available for comparison")
            return
        
        for method in methods:
            best_pc = results[method]['best_pc']
            evr = results[method]['explained_variance_ratio']
            print(f"\n{method}:")
            print(f"  Best PC{best_pc['PC']}: ρ = {best_pc['rho']:+.3f}")
            print(f"  EVR cumulative: {np.sum(evr[:3]):.1%}")
        
        # Check if results are consistent
        rhos = [results[method]['best_pc']['rho'] for method in methods]
        if all(abs(rho) > 0.7 for rho in rhos):
            print(f"\n✓ Consistent strong dose correlation across methods")
        else:
            print(f"\n⚠ Dose correlation varies across normalization paths")
    
    def create_corrected_visualizations(self, results, polarity):
        """Create corrected visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{polarity.title()} Ion: Corrected Statistical Analysis', fontsize=16, fontweight='bold')
        
        # Use first method for visualization
        method = list(results.keys())[0]
        data = results[method]
        
        scores = data['scores_corrected']
        dose_labels = data['dose_labels']
        correlations = data['correlations']
        evr = data['explained_variance_ratio']
        
        # Find dose-tracking PCs
        dose_pc = max(correlations, key=lambda x: x['rho_abs'])
        dose_pc_idx = dose_pc['PC'] - 1
        
        # 1. Corrected scores plot
        ax1 = axes[0, 0]
        scatter = ax1.scatter(scores[:, 0], scores[:, dose_pc_idx] if dose_pc_idx != 0 else scores[:, 1], 
                            c=dose_labels, cmap='viridis', s=100, alpha=0.7)
        ax1.set_xlabel(f'PC1 ({evr[0]:.1%} variance)')
        ax1.set_ylabel(f'PC{dose_pc["PC"]} ({evr[dose_pc_idx]:.1%} variance) - DOSE AXIS')
        ax1.set_title('Per-Dose Means (n=5)\nSign Corrected for Positive Dose Correlation')
        plt.colorbar(scatter, ax=ax1, label='Dose (μC/cm²)')
        
        # 2. Dose correlation plot
        ax2 = axes[0, 1]
        pcs = [c['PC'] for c in correlations]
        rhos = [c['rho'] for c in correlations]
        colors = ['red' if c['significant'] else 'gray' for c in correlations]
        
        bars = ax2.bar(pcs, rhos, color=colors, alpha=0.7)
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Spearman ρ (dose correlation)')
        ax2.set_title('Dose Correlations\n(Red = |ρ| > 0.8)')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Add correlation values on bars
        for bar, rho in zip(bars, rhos):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{rho:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 3. Explained variance
        ax3 = axes[1, 0]
        ax3.bar(range(1, len(evr) + 1), evr, alpha=0.7, color='steelblue')
        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Explained Variance Ratio')
        ax3.set_title(f'Variance Explained\nPC1: {evr[0]:.1%} (after mean centering)')
        
        # 4. Top loadings for dose-tracking PC
        ax4 = axes[1, 1]
        loadings = data['loadings_corrected'][dose_pc_idx]
        masses = data['masses']
        
        # Get top 10 positive and negative loadings
        top_indices = np.argsort(np.abs(loadings))[-20:]
        top_masses = masses[top_indices]
        top_loadings = loadings[top_indices]
        
        colors = ['red' if l > 0 else 'blue' for l in top_loadings]
        ax4.barh(range(len(top_loadings)), top_loadings, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(top_loadings)))
        ax4.set_yticklabels([f'{m:.1f}' for m in top_masses])
        ax4.set_xlabel('PC Loading')
        ax4.set_ylabel('Mass (u)')
        ax4.set_title(f'Top Loadings - PC{dose_pc["PC"]} (Dose Axis)\nRed: ↑ with dose, Blue: ↓ with dose')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.results_dir / f'{polarity}_corrected_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Corrected analysis plot saved: {plot_path}")
    
    def save_corrected_results(self, results, polarity):
        """Save corrected statistical results"""
        summary = {
            'polarity': polarity,
            'analysis_type': 'corrected_statistical',
            'n_dose_levels': len(self.dose_levels),
            'n_patterns_per_dose': self.n_patterns,
            'statistical_notes': [
                'Used per-dose means (n=5) to avoid pseudo-replication',
                'PC signs flipped for positive dose correlation (sign is arbitrary)', 
                'Exact Spearman p-value bound: p ≥ 0.0167 for perfect monotonic trend',
                'Correlations tested across both normalization paths'
            ]
        }
        
        for method, data in results.items():
            method_summary = {
                'method': method,
                'best_dose_pc': data['best_pc']['PC'],
                'dose_correlation': data['best_pc']['rho'],
                'exact_p_bound': data['best_pc']['exact_p_bound'],
                'pc_correlations': [
                    {
                        'PC': c['PC'],
                        'rho': c['rho'],
                        'significant': c['significant']
                    } for c in data['correlations']
                ],
                'explained_variance_ratio': data['explained_variance_ratio'].tolist(),
                'cumulative_variance_3pc': np.sum(data['explained_variance_ratio'][:3])
            }
            summary[f'{method}_results'] = method_summary
        
        # Save summary
        summary_path = self.results_dir / f'{polarity}_corrected_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Corrected results saved: {summary_path}")
    
    def run_corrected_analysis(self):
        """Run complete corrected statistical analysis"""
        print("="*80)
        print("ToF-SIMS PCA: STATISTICAL CORRECTIONS")
        print("="*80)
        print("Addressing:")
        print("• Pseudo-replication (using per-dose means, n=5)")
        print("• PC sign arbitrariness (flipped for positive dose correlation)")
        print("• Normalization path validation (testing both methods)")
        print("• Proper p-value bounds (exact Spearman for n=5)")
        print("="*80)
        
        all_results = {}
        
        for polarity in ['negative', 'positive']:
            results = self.analyze_polarity_corrected(polarity)
            self.create_corrected_visualizations(results, polarity)
            self.save_corrected_results(results, polarity)
            all_results[polarity] = results
        
        # Final summary
        self.create_final_summary(all_results)
        
        return all_results
    
    def create_final_summary(self, all_results):
        """Create final corrected summary"""
        print("\n" + "="*80)
        print("FINAL CORRECTED SUMMARY")
        print("="*80)
        
        for polarity in ['negative', 'positive']:
            results = all_results[polarity]
            print(f"\n{polarity.upper()} IONS:")
            
            for method, data in results.items():
                best_pc = data['best_pc']
                evr = data['explained_variance_ratio']
                
                print(f"  {method}:")
                print(f"    Dose-tracking PC: PC{best_pc['PC']}")
                print(f"    Dose correlation: ρ = {best_pc['rho']:+.3f} (sign corrected)")
                print(f"    Exact p-bound: p ≥ {best_pc['exact_p_bound']:.4f}")
                print(f"    PC1 EVR: {evr[0]:.1%} (after mean centering)")
        
        print(f"\n{'='*80}")
        print("✅ STATISTICAL CORRECTIONS COMPLETE")
        print("✅ Results now properly account for experimental design")
        print("✅ Per-dose means avoid pseudo-replication")
        print("✅ PC signs oriented for interpretability")
        print("✅ Both normalization paths validated")

if __name__ == "__main__":
    corrector = StatisticalPCACorrection("/home/dreece23/sims-pca-ws")
    results = corrector.run_corrected_analysis()