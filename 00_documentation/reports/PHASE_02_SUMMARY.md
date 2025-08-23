# Phase 02 — Preprocessing

**Pipelines**
- Baseline: TIC normalization → square-root transform
- Robust: PQN normalization → square-root → Pareto scaling (feature-wise)

**Inputs**
- out/all_positive_data_renamed.tsv
- out/all_negative_data_renamed.tsv
- meta/pos_catalog.csv, meta/neg_catalog.csv

**Outputs**
- Combined matrices:
  - 02_preprocessing/matrices/baseline_tic_sqrt.tsv
  - 02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv
- Per-polarity matrices:
  - 02_preprocessing/matrices_pos/{baseline_tic_sqrt_pos.tsv,robust_pqn_sqrt_pareto_pos.tsv}
  - 02_preprocessing/matrices_neg/{baseline_tic_sqrt_neg.tsv,robust_pqn_sqrt_pareto_neg.tsv}
- QC: 02_preprocessing/qc/{tic_hist.png,intensity_density.png,baseline_vs_robust_scatter.png,method_agreement_samplewise_hist.png}
- Logs: 02_preprocessing/logs/{scaling_factors.csv,preprocess_run.json,validation.json}
- Handoff: 02_preprocessing/exports/HANDOFF.json

**Notes**
- No centering saved; PCA will center internally.
- Sample columns are identical across matrices and match Phase-1 TSVs.
- Positive/negative masses are concatenated by rows to preserve all features.

## Validation

**Schema/metrics**: PASS
- All matrices validated successfully (1,853 rows, 15 samples)
- Agreement metrics meet 0.60 threshold:
  - Samplewise correlation (mean): 0.6023
  - Mean-spectrum correlation: 0.6689

**Folder hygiene**: PASS
- All required directories and files present
- No disallowed files or patterns detected

**Key Metrics**
- Samplewise Pearson r (median): 0.5613
- Samplewise Pearson r (mean): 0.6023  
- Mean-spectrum Pearson r: 0.6689
- Range: [0.5293, 0.7761]

**Enhanced QC**
- New samplewise correlation histogram: method_agreement_samplewise_hist.png
- Method agreement statistics: method_agreement_stats.{json,txt}