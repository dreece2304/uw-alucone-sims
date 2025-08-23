# Claude Commands for SIMS-PCA Workspace

This directory contains three specialized Claude commands for managing the SIMS-PCA analysis workflow. All commands are **plan-only** and **non-destructive** by design.

## Available Commands

### `/phase-validate`
**Purpose:** Validate current phase setup and dependencies  
**Output:** Human-readable findings + machine JSON with validation results

### `/phase-summary` 
**Purpose:** Generate phase summary report draft  
**Output:** PHASE_SUMMARY.md draft + machine JSON with metadata

### `/phase-run-pca`
**Purpose:** Generate PCA analysis plan (no execution)  
**Output:** Shell command sequence + Python stubs + machine JSON plan

## Important: Plan-Only Operation

**These commands never write files or execute code automatically.** They only:
- Read existing files
- Generate plans and drafts  
- Output structured JSON for downstream processing

Any file writes or code execution requires explicit operator approval via separate `/approve write` steps.

## Workflow Integration

Each command outputs a `MACHINE_OUTPUT` JSON block that should be saved and processed by the status updater:

```bash
# Example round-trip workflow:

# 1. Run validation command
claude /phase-validate > validation_output.txt

# 2. Extract JSON from output (manual step)
# Copy the MACHINE_OUTPUT json block to validation_results.json

# 3. Update status files
python scripts/update_status.py --input validation_results.json --type validate

# Similar pattern for other commands:
claude /phase-summary > summary_output.txt
# Extract JSON to summary_results.json  
python scripts/update_status.py --input summary_results.json --type summary

claude /phase-run-pca > pca_plan_output.txt
# Extract JSON to pca_plan.json
python scripts/update_status.py --input pca_plan.json --type pca_plan
```

## JSON Output Format

All commands end with a fenced code block containing machine-readable JSON:

```json
{
  "ok": true/false,
  "findings": [...],
  "problems": [...], 
  "suggested_next_steps": [...],
  "artifacts": [...]
}
```

This JSON should be extracted and passed to `scripts/update_status.py` to maintain consistent project state tracking.