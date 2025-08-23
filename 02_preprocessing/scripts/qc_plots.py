import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

def main():
    raw_pos = "out/all_positive_data_renamed.tsv"
    raw_neg = "out/all_negative_data_renamed.tsv"
    base = "02_preprocessing/matrices/baseline_tic_sqrt.tsv"
    rob  = "02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv"

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
    print("QC written.")
if __name__ == "__main__":
    main()