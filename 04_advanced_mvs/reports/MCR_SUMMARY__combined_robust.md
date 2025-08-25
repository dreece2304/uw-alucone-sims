# MCR-ALS Summary Report: combined_robust

## Model Configuration

**Components (k*)**: 2
**Initialization**: NMF solution with 3 starts (direct + 2 perturbed)
**Best Start**: perturbed_1
**Convergence**: True

## Reconstruction Quality

- **R²** = 0.998
- **RMSE** = 0.0030
- **Final Residual** = 2.58e-01

## Best Dose-Responsive Component

**Component 1** shows strongest dose correlation:
- ρ₅ (dose means): -1.000 (p = 0.000)
- ρ (all samples): -0.982 (p = 0.000)

### Top Contributing m/z Values

- **16.0** (weight: 0.0435)
- **1.0** (weight: 0.0356)
- **25.0** (weight: 0.0296)
- **17.0** (weight: 0.0277)
- **24.0** (weight: 0.0232)
- **60.0** (weight: 0.0201)
- **77.0** (weight: 0.0183)
- **76.0** (weight: 0.0174)
- **19.0** (weight: 0.0171)
- **13.0** (weight: 0.0166)


## Component Stability

Components show stability across multiple starts:
- **Stable fraction** (≥0.90 similarity): 1.00
- **Individual stabilities**: 1.00, 1.00

## All Component Dose Correlations

| Component | ρ₅ | p-value | ρ_all | p-value |
|-----------|-----|---------|-------|---------|
| 1 | -1.000 | 0.000 | -0.982 | 0.000 |
| 2 | 0.000 | 1.000 | 0.022 | 0.938 |


## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.998 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 1.000 (p=0.000) | ✅ PASS |
| Stability (≥67% stable, or ≥90% for k=2) | 100.0% | ✅ PASS |

## MCR-ALS vs NMF Comparison

MCR-ALS provides chemically interpretable pure component spectra through:
- Non-negativity constraints on both spectra (S) and concentrations (C)
- Alternating least squares optimization for factor separation
- Initialization from NMF solution ensures reproducible results

---

*Generated: 2025-08-24 22:08:33*
