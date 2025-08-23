# Phase 01 — Raw Data Audit & Validation

**Objective.** Validate TSV integrity, emit catalogs, enforce folder hygiene, and index raw stores & spatial sidecars for downstream analysis.

## Inputs
- TSVs: `out/all_positive_data_renamed.tsv`, `out/all_negative_data_renamed.tsv`
- Policy: `_shared/policies/phase1_01_raw_data_policy.json`
- Raw store (external): `/home/dreece23/Data/2025_08_19_DuncanReece/{PositiveIonData,NegativeIonData}`

## Audits & Outcomes
- TSV audit: **PASS** after minimal repair if needed; 15 samples per polarity; integer, sorted `Mass (u)`; non-negative intensities.
- Catalogs: `meta/pos_catalog.csv`, `meta/neg_catalog.csv`
- Folder hygiene: **PASS** (`01_raw_data` contains only audit artifacts, manifests, and optional small spatial index files)
- Spatial sidecars/index: `01_raw_data/imaging_index.json` (if npz present; missing acquisition fields recorded in sidecars)
- Raw manifest: `01_raw_data/manifests/raw_manifest.{tsv,json}`

## Artifacts
- `01_raw_data/AUDIT_REPORT.md`, `01_raw_data/tsv_audit.json`
- `01_raw_data/FOLDER_AUDIT.md`, `01_raw_data/folder_audit.json`
- `01_raw_data/imaging_index.json` (if present)
- `01_raw_data/manifests/raw_manifest.tsv`
- `meta/pos_catalog.csv`, `meta/neg_catalog.csv`

## Next Steps
- **Phase 02 – Preprocessing:** baseline TIC→√ and robust PQN→√→Pareto (centering only inside PCA); produce QC (TIC hist, density, method agreement); emit preprocessed matrices for PCA only (no saved centered copies).

_Environment lock:_ see `env/requirements-lock.txt`.
