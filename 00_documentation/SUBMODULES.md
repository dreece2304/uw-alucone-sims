# Git Submodules

## ATOFSIMSCLASS Submodule

ATOFSIMSCLASS is a pinned submodule. To update it:

```bash
# Update to latest remote version
git submodule update --remote ATOFSIMSCLASS

# Checkout a specific commit or tag
git -C ATOFSIMSCLASS checkout <desired-commit-or-tag>

# Update parent repository pointer
git add ATOFSIMSCLASS && git commit -m "bump submodule pointer"
```

## CI Integration

CI checks out submodules automatically using `actions/checkout@v4` with `submodules: true`.

## Important Notes

**Do not commit work inside the submodule without also updating the parent pointer.** Any changes to submodule contents must be accompanied by a corresponding commit in the parent repository that updates the submodule pointer.