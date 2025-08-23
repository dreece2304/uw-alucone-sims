# Phase Summary Reports

## Purpose

This directory contains completion reports for each analysis phase, providing comprehensive documentation of inputs, outputs, results, and issues encountered during the ToF-SIMS alucone dose-response analysis workflow.

## Report Generation Protocol

**Upon phase completion**, the responsible analysis agent must:

1. **Instantiate phase summary**: Copy `PHASE_SUMMARY.template.md` to `PHASE_{ID}_SUMMARY.md` (e.g., `PHASE_02_SUMMARY.md` for preprocessing phase)

2. **Complete all template fields**:
   - Replace all `{PLACEHOLDER}` fields with actual values
   - Include SHA256 checksums for all input and output files
   - Document key findings with 1-3 bullet points
   - Record any issues encountered and their resolutions

3. **Update global status**: Update `_shared/state/STATUS.json` to reflect phase completion and reference the generated summary report

4. **Link outputs**: Ensure all output file paths in the summary use relative links to enable navigation within the documentation

## Report Structure

Each phase summary includes:
- **Metadata**: Date, git hash, phase identification
- **Inputs**: All consumed files with integrity checksums
- **Outputs**: All produced artifacts with links and checksums  
- **Key Results**: 1-3 major findings or accomplishments
- **Issues & Fixes**: Problems encountered and their solutions
- **Next Steps**: Recommendations for downstream phases
- **Quality Control**: Validation and reproducibility status

## File Naming Convention

- Template: `PHASE_SUMMARY.template.md`
- Phase reports: `PHASE_{ID}_SUMMARY.md` where ID is two-digit phase number (02, 03, 04, etc.)
- Final report: `PHASE_08_SUMMARY.md` serves as the project completion report

## Integration with Project Status

All phase summary reports are referenced in the global project status tracking system and serve as the authoritative record of analysis progress and results for each phase of the workflow.