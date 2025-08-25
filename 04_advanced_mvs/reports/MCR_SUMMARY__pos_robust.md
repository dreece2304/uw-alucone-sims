# MCR-ALS Summary Report: pos_robust

## Model Configuration

**Components (k*)**: 6
**Initialization**: NMF solution with 3 starts (direct + 2 perturbed)
**Best Start**: perturbed_1
**Convergence**: False

## Reconstruction Quality

- **R²** = 0.997
- **RMSE** = 0.0005
- **Final Residual** = 2.88e-03

## Best Dose-Responsive Component

**Component 3** shows strongest dose correlation:
- ρ₅ (dose means): 0.900 (p = 0.037)
- ρ (all samples): 0.887 (p = 0.000)

### Top Contributing m/z Values

- **39.0** (weight: 0.0136)
- **77.0** (weight: 0.0133)
- **41.0** (weight: 0.0133)
- **91.0** (weight: 0.0127)
- **27.0** (weight: 0.0126)
- **55.0** (weight: 0.0123)
- **115.0** (weight: 0.0120)
- **53.0** (weight: 0.0119)
- **65.0** (weight: 0.0113)
- **51.0** (weight: 0.0112)


## Component Stability

Components show stability across multiple starts:
- **Stable fraction** (≥0.90 similarity): 1.00
- **Individual stabilities**: 1.00, 1.00, 1.00, 1.00, 1.00, 1.00

## All Component Dose Correlations

| Component | ρ₅ | p-value | ρ_all | p-value |
|-----------|-----|---------|-------|---------|
| 1 | 0.000 | 1.000 | 0.000 | 1.000 |
| 2 | 0.000 | 1.000 | -0.297 | 0.282 |
| 3 | 0.900 | 0.037 | 0.887 | 0.000 |
| 4 | -0.100 | 0.873 | 0.098 | 0.728 |
| 5 | 0.700 | 0.188 | 0.268 | 0.335 |
| 6 | -0.700 | 0.188 | -0.393 | 0.148 |


## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.997 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 0.900 (p=0.037) | ✅ PASS |
| Stability (≥67% stable, or ≥90% for k=2) | 100.0% | ✅ PASS |

## MCR-ALS vs NMF Comparison

MCR-ALS provides chemically interpretable pure component spectra through:
- Non-negativity constraints on both spectra (S) and concentrations (C)
- Alternating least squares optimization for factor separation
- Initialization from NMF solution ensures reproducible results

---

*Generated: 2025-08-24 22:10:01*
