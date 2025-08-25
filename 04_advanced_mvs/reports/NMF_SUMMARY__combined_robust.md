# NMF Summary Report: combined_robust

## Model Selection

**Chosen k***: 2 components

### Selection Metrics by k

| k | R² | RMSE | Stable Fraction | Best |ρ₅| |
|---|----|----|----------------|---------|
| 2 | 0.998 | 0.003 | 1.00 | 1.00 |
| 3 | 0.999 | 0.002 | 1.00 | 1.00 |
| 4 | 0.999 | 0.002 | 1.00 | 1.00 |
| 5 | 1.000 | 0.001 | 1.00 | 1.00 |
| 6 | 1.000 | 0.001 | 1.00 | 1.00 |

## Best Dose-Responsive Component

**Component 1** shows strongest dose correlation:
- ρ₅ (dose means): -1.000 (p = 0.000)
- ρ (all samples): -0.982 (p = 0.000)

### Top Contributing m/z Values

- **16.0** (weight: 1.6053)
- **1.0** (weight: 1.3151)
- **25.0** (weight: 1.0942)
- **17.0** (weight: 1.0207)
- **24.0** (weight: 0.8570)
- **60.0** (weight: 0.7421)
- **77.0** (weight: 0.6752)
- **76.0** (weight: 0.6398)
- **19.0** (weight: 0.6333)
- **13.0** (weight: 0.6119)


## Component Stability

Components show stability across random seeds:
- Stable fraction (≥0.90 similarity): 1.00

## Reconstruction Quality

- R² = 0.998
- RMSE = 0.003

## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.998 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable) | 100.0% | ✅ PASS |

---

*Generated: 2025-08-24 17:08:13*
