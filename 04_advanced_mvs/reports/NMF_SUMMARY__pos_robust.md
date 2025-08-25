# NMF Summary Report: pos_robust

## Model Selection

**Chosen k***: 6 components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
| 2 | 0.903 | 0.003 | 1.00 | 0.00 |
| 3 | 0.954 | 0.002 | 1.00 | 0.90 |
| 4 | 0.991 | 0.001 | 1.00 | 0.90 |
| 5 | 0.994 | 0.001 | 1.00 | 0.90 |
| 6 | 0.996 | 0.001 | 1.00 | 1.00 |

## Best Dose-Responsive Component

**Component 6** shows strongest dose correlation:
- ρ₅ (dose means): -1.000 (p = 0.000)
- ρ (all samples): -0.906 (p = 0.000)

### Top Contributing m/z Values

- **28.0** (weight: 0.1433)
- **27.0** (weight: 0.1145)
- **45.0** (weight: 0.1061)
- **74.0** (weight: 0.0666)
- **63.0** (weight: 0.0641)
- **86.0** (weight: 0.0628)
- **98.0** (weight: 0.0619)
- **1.0** (weight: 0.0603)
- **50.0** (weight: 0.0587)
- **62.0** (weight: 0.0582)


## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): 1.00

## Reconstruction Quality

- R² = 0.996
- RMSE = 0.001

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.996 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable) | 100.0% | ✅ PASS |

---

*Generated: 2025-08-24 17:08:21*
