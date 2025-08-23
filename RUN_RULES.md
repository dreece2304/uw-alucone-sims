# Analysis Execution Rules

## Working Directory Requirements

**Always set the IDE/terminal working directory to the phase folder before running.** Analysis scripts must be executed from within their designated phase directory (e.g., `02_preprocessing/`, `03_pca_analysis/`) to ensure proper relative path resolution and write permission enforcement.

## Output Location Restrictions

**Never produce outputs in repository root.** All analysis outputs, intermediate files, temporary data, and results must be written to designated phase directories or approved output locations as specified in the phase agent configurations. Writing to the repository root violates governance policies and will cause execution failures.

## Status Update Requirements

**Every analysis run must update both the project-level and phase-level status JSON files.** This includes:

- Update `_shared/state/STATUS.json` with global project progress
- Update local `{phase}/phase_agent/STATUS.json` with phase-specific status  
- Append entry to `_shared/state/CHANGELOG.md` documenting the analysis run
- Write `output_contract.json` documenting all produced artifacts

Failure to maintain status updates breaks the inter-agent communication protocol and prevents downstream phases from executing properly.