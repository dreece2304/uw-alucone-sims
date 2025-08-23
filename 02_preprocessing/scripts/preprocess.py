import json, hashlib, os, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "02_preprocessing"

def tic_normalize(df):
    X = df.iloc[:,1:].astype(float).values
    tic = X.sum(axis=0, keepdims=True)
    tic[tic==0] = 1.0
    return X / tic, tic.ravel()

def pqn_normalize(df):
    X = df.iloc[:,1:].astype(float).values
    tic = X.sum(axis=0, keepdims=True)
    tic[tic==0] = 1.0
    Xn = X / tic
    ref = np.median(Xn, axis=1, keepdims=True)
    ref[ref==0] = 1.0
    ratios = Xn / ref
    factors = np.median(ratios, axis=0, keepdims=True)
    factors[factors==0] = 1.0
    Xp = Xn / factors
    return Xp, tic.ravel(), factors.ravel()

def sqrt_transform(X):
    return np.sqrt(np.maximum(X, 0.0))

def pareto_scale(X):
    # Feature stdev (ddof=1); guard zeros
    std = X.std(axis=1, ddof=1, keepdims=True)
    std[std==0] = 1.0
    return X / np.sqrt(std)

def write_matrix(df_like, X, out_path):
    out = pd.DataFrame(X, columns=df_like.columns[1:])
    out.insert(0, df_like.columns[0], df_like.iloc[:,0].values)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, sep="\t", index=False)

def main():
    pos_path = ROOT / "out" / "all_positive_data_renamed.tsv"
    neg_path = ROOT / "out" / "all_negative_data_renamed.tsv"
    pos = pd.read_csv(pos_path, sep="\t")
    neg = pd.read_csv(neg_path, sep="\t")
    assert pos.columns[0] == "Mass (u)" and neg.columns[0] == "Mass (u)", "First header must be 'Mass (u)'"
    # Extract sample base names (remove -P/-N suffix) and verify alignment
    pos_samples = [col[:-2] if col.endswith('-P') else col for col in pos.columns[1:]]
    neg_samples = [col[:-2] if col.endswith('-N') else col for col in neg.columns[1:]]
    assert pos_samples == neg_samples, f"Sample base names mismatch: {pos_samples} vs {neg_samples}"
    samples = list(pos.columns[1:])  # Keep original pos column names for output

    # ---- Baseline: TIC → sqrt ----
    Xp_base, tic_pos = tic_normalize(pos)
    Xn_base, tic_neg = tic_normalize(neg)
    Xp_base = sqrt_transform(Xp_base)
    Xn_base = sqrt_transform(Xn_base)

    base_pos_df = pd.DataFrame(Xp_base, columns=samples); base_pos_df.insert(0,"Mass (u)", pos.iloc[:,0].values)
    base_neg_df = pd.DataFrame(Xn_base, columns=samples); base_neg_df.insert(0,"Mass (u)", neg.iloc[:,0].values)
    base_merged = pd.concat([base_pos_df, base_neg_df], axis=0, ignore_index=True)
    write_matrix(base_merged, base_merged.iloc[:,1:].values, OUT_DIR / "matrices" / "baseline_tic_sqrt.tsv")
    
    # Write per-polarity baseline matrices
    write_matrix(base_pos_df, base_pos_df.iloc[:,1:].values, OUT_DIR / "matrices_pos" / "baseline_tic_sqrt_pos.tsv")
    write_matrix(base_neg_df, base_neg_df.iloc[:,1:].values, OUT_DIR / "matrices_neg" / "baseline_tic_sqrt_neg.tsv")

    # ---- Robust: PQN → sqrt → Pareto ----
    Xp_rob, tic_pos2, fac_pos = pqn_normalize(pos)
    Xn_rob, tic_neg2, fac_neg = pqn_normalize(neg)
    Xp_rob = pareto_scale(sqrt_transform(Xp_rob))
    Xn_rob = pareto_scale(sqrt_transform(Xn_rob))

    rob_pos_df = pd.DataFrame(Xp_rob, columns=samples); rob_pos_df.insert(0,"Mass (u)", pos.iloc[:,0].values)
    rob_neg_df = pd.DataFrame(Xn_rob, columns=samples); rob_neg_df.insert(0,"Mass (u)", neg.iloc[:,0].values)
    rob_merged = pd.concat([rob_pos_df, rob_neg_df], axis=0, ignore_index=True)
    write_matrix(rob_merged, rob_merged.iloc[:,1:].values, OUT_DIR / "matrices" / "robust_pqn_sqrt_pareto.tsv")
    
    # Write per-polarity robust matrices
    write_matrix(rob_pos_df, rob_pos_df.iloc[:,1:].values, OUT_DIR / "matrices_pos" / "robust_pqn_sqrt_pareto_pos.tsv")
    write_matrix(rob_neg_df, rob_neg_df.iloc[:,1:].values, OUT_DIR / "matrices_neg" / "robust_pqn_sqrt_pareto_neg.tsv")

    # ---- Diagnostics: scaling factors & TIC ----
    diag = pd.DataFrame({
        "sample": samples,
        "tic_pos": tic_pos,
        "tic_neg": tic_neg,
        "pqn_factor_pos": fac_pos,
        "pqn_factor_neg": fac_neg
    })
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    diag.to_csv(OUT_DIR / "logs" / "scaling_factors.csv", index=False)

    # ---- Run metadata ----
    def sha256(p: Path):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
        return h.hexdigest()

    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "inputs": {
            "positive": str(pos_path), "positive_sha256": sha256(pos_path),
            "negative": str(neg_path), "negative_sha256": sha256(neg_path)
        },
        "outputs": {
            "baseline_tic_sqrt": "02_preprocessing/matrices/baseline_tic_sqrt.tsv",
            "robust_pqn_sqrt_pareto": "02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv",
            "baseline_tic_sqrt_pos": "02_preprocessing/matrices_pos/baseline_tic_sqrt_pos.tsv",
            "baseline_tic_sqrt_neg": "02_preprocessing/matrices_neg/baseline_tic_sqrt_neg.tsv",
            "robust_pqn_sqrt_pareto_pos": "02_preprocessing/matrices_pos/robust_pqn_sqrt_pareto_pos.tsv",
            "robust_pqn_sqrt_pareto_neg": "02_preprocessing/matrices_neg/robust_pqn_sqrt_pareto_neg.tsv",
            "scaling_factors": "02_preprocessing/logs/scaling_factors.csv"
        },
        "samples": samples,
        "notes": "Centering is NOT persisted; only inside PCA later."
    }
    (OUT_DIR / "logs" / "preprocess_run.json").write_text(json.dumps(meta, indent=2))
    print("Wrote matrices and logs.")
if __name__ == "__main__":
    main()