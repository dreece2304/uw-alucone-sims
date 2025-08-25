# NMF Summary Report: neg_robust

## Model Selection

**Chosen k***: 2 components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
| 2 | 0.999 | 0.003 | 1.00 | 1.00 |
| 3 | 0.999 | 0.002 | 1.00 | 1.00 |
| 4 | 1.000 | 0.002 | 1.00 | 1.00 |
| 5 | 1.000 | 0.001 | 1.00 | 1.00 |
| 6 | 1.000 | 0.001 | 1.00 | 1.00 |

## Best Dose-Responsive Component

**Component 1** shows strongest dose correlation:
- ρ₅ (dose means): -1.000 (p = 0.000)
- ρ (all samples): -0.971 (p = 0.000)

### Top Contributing m/z Values

- **16.0** (weight: 1.6151)
- **1.0** (weight: 1.3207)
- **25.0** (weight: 1.1003)
- **17.0** (weight: 1.0260)
- **24.0** (weight: 0.8625)
- **60.0** (weight: 0.7480)
- **77.0** (weight: 0.6797)
- **76.0** (weight: 0.6449)
- **19.0** (weight: 0.6365)
- **13.0** (weight: 0.6151)


## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): 1.00

## Reconstruction Quality

- R² = 0.999
- RMSE = 0.003

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.999 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable) | 100.0% | ✅ PASS |

---

*Generated: 2025-08-24 17:08:30*
