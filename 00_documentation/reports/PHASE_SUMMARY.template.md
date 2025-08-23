# Phase {PHASE_ID} Summary Report: {PHASE_NAME}

**Completion Date**: {DATE}
**Git Hash**: {GIT_HASH}
**Phase Directory**: `{PHASE_DIRECTORY}`

## Inputs

{INPUT_LIST}
<!-- Template format:
- **{input_name}**: `{file_path}` (SHA256: `{sha256_hash}`)
-->

## Outputs

{OUTPUT_LIST}
<!-- Template format:
- **{output_name}**: [`{file_path}`]({relative_link}) (SHA256: `{sha256_hash}`)
-->

## Key Results

{KEY_RESULTS}
<!-- Template format:
- {Key finding or result 1}
- {Key finding or result 2}  
- {Key finding or result 3}
-->

## Issues & Fixes

{ISSUES_AND_FIXES}
<!-- Template format:
**Issue**: {Description of problem encountered}
**Fix**: {Description of solution applied}
**Status**: {Resolved/Ongoing/Deferred}
-->

## Next Steps

{NEXT_STEPS}
<!-- Template format:
- {Action item 1 for downstream phases}
- {Action item 2 for downstream phases}
- {Any follow-up analysis recommendations}
-->

## Quality Control

**Data validation**: {QC_STATUS}
**Statistical tests**: {STATS_STATUS}  
**Reproducibility check**: {REPRO_STATUS}

---
*Generated on {TIMESTAMP} | Environment: `env/requirements-lock.txt` | Git: `{GIT_HASH}`*