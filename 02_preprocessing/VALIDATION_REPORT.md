# Phase 2 Preprocessing Validation Report

- **policy**: balanced
- **thresholds**: median_samplewise_r ≥ 0.55, mean_spectrum_r ≥ 0.65
- **ok**: True

## Summary

- **Rows**: 1853 (baseline), 1853 (robust)
- **Samples**: 15

## Agreement Metrics

- **Samplewise correlation (median)**: 0.5613
- **Samplewise correlation (mean)**: 0.6023
- **Mean-spectrum correlation**: 0.6689

Balanced policy thresholds: median_samplewise_r ≥ 0.55, mean_spectrum_r ≥ 0.65

## Issues

None detected.

## File Information

### Baseline Matrix
- Path: `02_preprocessing/matrices/baseline_tic_sqrt.tsv`
- Size: 539,410 bytes
- SHA256: `c9d5e13fc7d4ef15866f820d36eb390e448d9228fa63a8ce3cefd08c6937938d`

### Robust Matrix
- Path: `02_preprocessing/matrices/robust_pqn_sqrt_pareto.tsv`
- Size: 508,257 bytes
- SHA256: `1967597a14bf619d9603445832aa5b9bd627d502c985517eecefa7f57f5820e1`
