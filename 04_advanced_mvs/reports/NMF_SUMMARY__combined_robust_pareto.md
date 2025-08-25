# NMF Summary Report: combined_robust_pareto

## Model Selection

**Chosen k***: 6 components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
| 2 | 0.947 | 0.026 | 1.00 | 0.90 |
| 3 | 0.967 | 0.021 | 1.00 | 1.00 |
| 4 | 0.976 | 0.018 | 1.00 | 1.00 |
| 5 | 0.985 | 0.014 | 1.00 | 1.00 |
| 6 | 0.990 | 0.011 | 1.00 | 1.00 |

## Best Dose-Responsive Component

**Component 2** shows strongest dose correlation:
- ρ₅ (dose means): -1.000 (p = 0.000)
- ρ (all samples): -0.960 (p = 0.000)

### Top Contributing m/z Values

- **16.0** (weight: 0.9865)
- **1.0** (weight: 0.9409)
- **25.0** (weight: 0.8284)
- **17.0** (weight: 0.8129)
- **24.0** (weight: 0.7048)
- **60.0** (weight: 0.6512)
- **13.0** (weight: 0.6283)
- **77.0** (weight: 0.6282)
- **76.0** (weight: 0.6139)
- **19.0** (weight: 0.6122)


## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): 1.00

## Reconstruction Quality

- R² = 0.990
- RMSE = 0.011

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.990 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable) | 100.0% | ✅ PASS |

---

*Generated: 2025-08-24 17:08:48*
