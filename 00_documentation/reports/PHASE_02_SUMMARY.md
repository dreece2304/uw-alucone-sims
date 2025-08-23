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

## Validation (Phase-2)

**Policy**: balanced (median ≥ 0.55 & mean-spectrum ≥ 0.65)

**Agreement Metrics**: 
- Median samplewise r: 0.5613 ≥ 0.55 ✓
- Mean samplewise r: 0.6023  
- Mean-spectrum r: 0.6689 ≥ 0.65 ✓

**Flagged Samples**: 12 MARGINAL (0.50 ≤ r < 0.60), 0 LOW (r < 0.50)
- Range: [0.5293, 0.7761]
- Only 3 samples achieve OK status (500μC dose)

**PQN Correlations**:
- vs pqn_factor_pos: r=0.7112, p=0.0030 (significant positive)
- vs pqn_factor_neg: r=-0.9336, p<0.0001 (highly significant negative)
- *Interpretation*: Low agreement stems from differential ionization efficiency responses, not random variation

**Sensitivity Conclusion**:
- Current (PQN→√→Pareto): median r=0.5613, mean-spectrum r=0.6689
- Variant (PQN→√, no Pareto): median r=0.9974, mean-spectrum r=0.8837
- Deltas: +0.4361, +0.2148 respectively
- **Recommendation**: Adopt PQN→√ (no Pareto) as robust path for Phase-3

**Per-polarity Handoff**: HANDOFF.json includes matrices_pos, matrices_neg, and matrices_variant (no-Pareto); defaults unchanged pending Phase-3 policy

**Hygiene**: PASS

**Artifacts Referenced**:
- 02_preprocessing/logs/samplewise_agreement.csv
- 02_preprocessing/logs/samplewise_pqn_crosstab.csv  
- 02_preprocessing/logs/robust_variant_metrics.json
- 02_preprocessing/qc/method_agreement_samplewise_hist.png
- 02_preprocessing/qc/method_agreement_samplewise_hist_variant.png