# MCR-ALS Summary Report: combined_robust_pareto

## Model Configuration

**Components (k*)**: 6
**Initialization**: NMF solution with 3 starts (direct + 2 perturbed)
**Best Start**: direct_nmf
**Convergence**: False

## Reconstruction Quality

- **R²** = 0.992
- **RMSE** = 0.0103
- **Final Residual** = 2.94e+00

## Best Dose-Responsive Component

**Component 6** shows strongest dose correlation:
- ρ₅ (dose means): 1.000 (p = 0.000)
- ρ (all samples): 0.974 (p = 0.000)

### Top Contributing m/z Values

- **39.0** (weight: 0.0056)
- **53.0** (weight: 0.0052)
- **77.0** (weight: 0.0047)
- **52.0** (weight: 0.0046)
- **65.0** (weight: 0.0040)
- **27.0** (weight: 0.0039)
- **66.0** (weight: 0.0038)
- **115.0** (weight: 0.0037)
- **249.0** (weight: 0.0037)
- **51.0** (weight: 0.0037)


## Component Stability

Components show stability across multiple starts:
- **Stable fraction** (≥0.90 similarity): 1.00
- **Individual stabilities**: 1.00, 1.00, 1.00, 1.00, 1.00, 1.00

## All Component Dose Correlations

| Component | ρ₅ | p-value | ρ_all | p-value |
|-----------|-----|---------|-------|---------|
| 1 | 0.000 | 1.000 | 0.000 | 1.000 |
| 2 | -0.900 | 0.037 | -0.644 | 0.010 |
| 3 | -0.500 | 0.391 | 0.098 | 0.727 |
| 4 | 0.000 | 1.000 | 0.644 | 0.010 |
| 5 | 0.000 | 1.000 | 0.470 | 0.077 |
| 6 | 1.000 | 0.000 | 0.974 | 0.000 |


## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.992 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable, or ≥90% for k=2) | 100.0% | ✅ PASS |

## MCR-ALS vs NMF Comparison

MCR-ALS provides chemically interpretable pure component spectra through:
- Non-negativity constraints on both spectra (S) and concentrations (C)
- Alternating least squares optimization for factor separation
- Initialization from NMF solution ensures reproducible results

---

*Generated: 2025-08-24 22:15:27*
