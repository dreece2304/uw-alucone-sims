# Phase 02 — Preprocessing

**Pipelines**
- Baseline: TIC normalization → square-root transform
- Robust: PQN normalization → square-root → Pareto scaling (feature-wise)

**Inputs**
- out/all_positive_data_renamed.tsv
- out/all_negative_data_renamed.tsv
- meta/pos_catalog.csv, meta/neg_catalog.csv

**Outputs**
- 02_preprocessing/matrices/baseline_tic_sqrt.tsv
- 02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv
- QC: 02_preprocessing/qc/{tic_hist.png,intensity_density.png,baseline_vs_robust_scatter.png}
- Logs: 02_preprocessing/logs/{scaling_factors.csv,preprocess_run.json}
- Handoff: 02_preprocessing/exports/HANDOFF.json

**Notes**
- No centering saved; PCA will center internally.
- Sample columns are identical across matrices and match Phase-1 TSVs.
- Positive/negative masses are concatenated by rows to preserve all features.