import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats import pearsonr

def load(tsv):
    df = pd.read_csv(tsv, sep="\t")
    mass = df.iloc[:,0].values
    X = df.iloc[:,1:].astype(float).values
    cols = df.columns[1:]
    return mass, X, cols

def savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def compute_samplewise_correlations(baseline_data, robust_data):
    """Compute samplewise Pearson correlations between baseline and robust matrices."""
    sample_correlations = []
    
    for i in range(baseline_data.shape[1]):  # For each sample
        baseline_sample = baseline_data[:, i]
        robust_sample = robust_data[:, i]
        
        # Remove any rows where either value is NaN or infinite
        valid_mask = np.isfinite(baseline_sample) & np.isfinite(robust_sample)
        if valid_mask.sum() < 10:  # Need at least 10 points for correlation
            continue
            
        try:
            r, _ = pearsonr(baseline_sample[valid_mask], robust_sample[valid_mask])
            if np.isfinite(r):
                sample_correlations.append(r)
        except Exception:
            continue
    
    return np.array(sample_correlations)

def main():
    raw_pos = "out/all_positive_data_renamed.tsv"
    raw_neg = "out/all_negative_data_renamed.tsv"
    base = "02_preprocessing/matrices/baseline_tic_sqrt.tsv"
    rob  = "02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv"

    # Original QC plots
    pos = pd.read_csv(raw_pos, sep="\t"); neg = pd.read_csv(raw_neg, sep="\t")
    tic_pos = pos.iloc[:,1:].astype(float).sum(axis=0).values
    tic_neg = neg.iloc[:,1:].astype(float).sum(axis=0).values
    plt.figure(); plt.hist(tic_pos, bins=15, alpha=0.6, label="POS TIC")
    plt.hist(tic_neg, bins=15, alpha=0.6, label="NEG TIC")
    plt.legend(); plt.xlabel("TIC"); plt.ylabel("Count")
    savefig("02_preprocessing/qc/tic_hist.png")

    _, Xb, _ = load(base); _, Xr, _ = load(rob)
    plt.figure(); plt.hist(Xb.ravel(), bins=100, alpha=0.6, density=True, label="baseline √")
    plt.hist(Xr.ravel(), bins=100, alpha=0.6, density=True, label="robust PQN√Pareto")
    plt.legend(); plt.xlabel("intensity"); plt.ylabel("density")
    savefig("02_preprocessing/qc/intensity_density.png")

    mb = Xb.mean(axis=0); mr = Xr.mean(axis=0)
    plt.figure(); plt.scatter(mb, mr, s=10, alpha=0.6)
    plt.xlabel("baseline mean int."); plt.ylabel("robust mean int.")
    savefig("02_preprocessing/qc/baseline_vs_robust_scatter.png")
    
    # New QC: Samplewise correlation histogram
    sample_correlations = compute_samplewise_correlations(Xb, Xr)
    
    if len(sample_correlations) > 0:
        plt.figure()
        plt.hist(sample_correlations, bins=12, alpha=0.7, edgecolor='black')
        plt.xlabel("Samplewise Pearson r (baseline vs robust)")
        plt.ylabel("Count")
        plt.title(f"Method Agreement Distribution (n={len(sample_correlations)})")
        
        # Add vertical lines for median and mean
        median_r = np.median(sample_correlations)
        mean_r = np.mean(sample_correlations)
        plt.axvline(median_r, color='red', linestyle='--', alpha=0.8, label=f'Median: {median_r:.3f}')
        plt.axvline(mean_r, color='orange', linestyle='--', alpha=0.8, label=f'Mean: {mean_r:.3f}')
        plt.legend()
        
        savefig("02_preprocessing/qc/method_agreement_samplewise_hist.png")
        
        # Save stats to text file
        stats = {
            "samplewise_correlation_median": float(median_r),
            "samplewise_correlation_mean": float(mean_r),
            "samplewise_correlation_count": int(len(sample_correlations)),
            "samplewise_correlation_min": float(np.min(sample_correlations)),
            "samplewise_correlation_max": float(np.max(sample_correlations))
        }
        
        stats_path = Path("02_preprocessing/qc/method_agreement_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Also create a simple text version
        text_stats_path = Path("02_preprocessing/qc/method_agreement_stats.txt")
        with open(text_stats_path, 'w') as f:
            f.write("Method Agreement Statistics (Baseline vs Robust)\n")
            f.write("=" * 50 + "\n")
            f.write(f"Samplewise Pearson r median: {median_r:.4f}\n")
            f.write(f"Samplewise Pearson r mean:   {mean_r:.4f}\n")
            f.write(f"Number of samples:           {len(sample_correlations)}\n")
            f.write(f"Range: [{np.min(sample_correlations):.4f}, {np.max(sample_correlations):.4f}]\n")
        
        print(f"QC written. Samplewise correlations - median: {median_r:.4f}, mean: {mean_r:.4f}")
    else:
        print("QC written. Warning: No valid samplewise correlations computed.")
if __name__ == "__main__":
    main()