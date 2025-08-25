# MCR-ALS Summary Report: combined_baseline

## Model Configuration

**Components (k*)**: 5
**Initialization**: NMF solution with 3 starts (direct + 2 perturbed)
**Best Start**: perturbed_1
**Convergence**: False

## Reconstruction Quality

- **R²** = 0.996
- **RMSE** = 0.0006
- **Final Residual** = 9.32e-03

## Best Dose-Responsive Component

**Component 2** shows strongest dose correlation:
- ρ₅ (dose means): -0.900 (p = 0.037)
- ρ (all samples): -0.698 (p = 0.004)

### Top Contributing m/z Values

- **16.0** (weight: 0.0169)
- **1.0** (weight: 0.0137)
- **25.0** (weight: 0.0115)
- **17.0** (weight: 0.0107)
- **24.0** (weight: 0.0090)
- **27.0** (weight: 0.0090)
- **45.0** (weight: 0.0082)
- **60.0** (weight: 0.0079)
- **39.0** (weight: 0.0076)
- **77.0** (weight: 0.0071)


## Component Stability

Components show stability across multiple starts:
- **Stable fraction** (≥0.90 similarity): 1.00
- **Individual stabilities**: 1.00, 1.00, 1.00, 1.00, 1.00

## All Component Dose Correlations

| Component | ρ₅ | p-value | ρ_all | p-value |
|-----------|-----|---------|-------|---------|
| 1 | 0.000 | 1.000 | 0.044 | 0.877 |
| 2 | -0.900 | 0.037 | -0.698 | 0.004 |
| 3 | 0.900 | 0.037 | 0.909 | 0.000 |
| 4 | -0.200 | 0.747 | -0.066 | 0.816 |
| 5 | 0.200 | 0.747 | 0.270 | 0.331 |


## Phase-Level Acceptance

| Criterion | Result | Status |
|-----------|--------|--------|
| Reconstruction (R² ≥ 0.70) | 0.996 | ✅ PASS |
| Dose Correlation (|ρ₅| ≥ 0.60, p ≤ 0.05) | 0.900 (p=0.037) | ✅ PASS |
| Stability (≥67% stable, or ≥90% for k=2) | 100.0% | ✅ PASS |

## MCR-ALS vs NMF Comparison

MCR-ALS provides chemically interpretable pure component spectra through:
- Non-negativity constraints on both spectra (S) and concentrations (C)
- Alternating least squares optimization for factor separation
- Initialization from NMF solution ensures reproducible results

---

*Generated: 2025-08-24 22:12:33*
