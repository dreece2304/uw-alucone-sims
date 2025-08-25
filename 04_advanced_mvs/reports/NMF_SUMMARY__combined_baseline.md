# NMF Summary Report: combined_baseline

## Model Selection

**Chosen k***: 5 components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
| 2 | 0.896 | 0.003 | 1.00 | 0.60 |
| 3 | 0.969 | 0.002 | 1.00 | 1.00 |
| 4 | 0.988 | 0.001 | 1.00 | 1.00 |
| 5 | 0.995 | 0.001 | 1.00 | 1.00 |
| 6 | 0.996 | 0.001 | 1.00 | 1.00 |

## Best Dose-Responsive Component

**Component 3** shows strongest dose correlation:
- ρ₅ (dose means): 1.000 (p = 0.000)
- ρ (all samples): 0.982 (p = 0.000)

### Top Contributing m/z Values

- **49.0** (weight: 0.1418)
- **25.0** (weight: 0.1387)
- **41.0** (weight: 0.1212)
- **77.0** (weight: 0.1116)
- **45.0** (weight: 0.1102)
- **91.0** (weight: 0.1090)
- **115.0** (weight: 0.1052)
- **73.0** (weight: 0.1043)
- **39.0** (weight: 0.1042)
- **65.0** (weight: 0.0959)


## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): 1.00

## Reconstruction Quality

- R² = 0.995
- RMSE = 0.001

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.995 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable) | 100.0% | ✅ PASS |

---

*Generated: 2025-08-24 17:08:39*
