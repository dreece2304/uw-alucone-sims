#!/usr/bin/env python3
import sys, json, fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_policy(p: Path) -> dict:
    return json.loads(p.read_text())

def bytes_to_mb(n: int) -> float:
    return n / (1024 * 1024)

def matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=str(ROOT / "_shared/policies/phase1_01_raw_data_policy.json"))
    args = ap.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"ERROR: policy file not found: {policy_path}", file=sys.stderr)
        return 2

    policy = load_policy(policy_path)
    base = ROOT / policy["folder"]

    ok = True
    findings, problems = [], []

    if not base.exists():
        problems.append(f"Missing folder: {base}")
        ok = False
    else:
        allowed_dirs   = set(policy.get("allowed_dirs", []))
        allowed_files  = set(policy.get("allowed_files", []))
        allowed_globs  = list(policy.get("allowed_globs", []))
        deny_exts      = set(policy.get("deny_extensions", []))
        max_mb         = float(policy.get("max_file_mb", 50))
        allowlist_file = policy.get("allowlist_file", ".allowlist")

        allowlist = set()
        allowlist_path = base / allowlist_file
        if allowlist_path.exists():
            for line in allowlist_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    allowlist.add(line)

        # Check top-level children in 01_raw_data
        for p in sorted(base.glob("*")):
            rel = p.relative_to(base).as_posix()
            if rel in allowlist:
                findings.append(f"ALLOWLIST: {rel}")
                continue

            if p.is_dir():
                if rel not in allowed_dirs:
                    problems.append(f"Disallowed directory: {rel}")
                    ok = False
                # Scan contents of allowed dirs for forbidden extensions / large files
                for q in p.rglob("*"):
                    if not q.is_file():
                        continue
                    relq = q.relative_to(base).as_posix()
                    if relq in allowlist:
                        findings.append(f"ALLOWLIST: {relq}")
                        continue
                    if q.suffix in deny_exts and not matches_any(q.name, allowed_globs):
                        problems.append(f"Forbidden extension in allowed dir: {relq}")
                        ok = False
                    sz_mb = bytes_to_mb(q.stat().st_size)
                    if sz_mb > max_mb and not matches_any(q.name, allowed_globs):
                        problems.append(f"Large file (> {max_mb} MB): {relq} ({sz_mb:.1f} MB)")
                        ok = False
            else:
                name = p.name
                # Top-level file policy
                if (name not in allowed_files) and (not matches_any(name, allowed_globs)):
                    problems.append(f"Top-level file not allowed: {rel}")
                    ok = False
                if p.suffix in deny_exts:
                    problems.append(f"Forbidden extension at top level: {rel}")
                    ok = False
                sz_mb = bytes_to_mb(p.stat().st_size)
                if sz_mb > max_mb and not matches_any(name, allowed_globs):
                    problems.append(f"Large file (> {max_mb} MB) at top level: {rel} ({sz_mb:.1f} MB)")
                    ok = False

    report = {
        "folder": policy["folder"],
        "ok": ok,
        "patterns": {
            "allowed_dirs": list(allowed_dirs) if base.exists() else [],
            "allowed_files": list(allowed_files) if base.exists() else [],
            "allowed_globs": list(allowed_globs) if base.exists() else [],
            "deny_extensions": list(deny_exts) if base.exists() else [],
            "max_file_mb": policy.get("max_file_mb", 50)
        },
        "findings": findings,
        "problems": problems
    }

    out_json = ROOT / "01_raw_data" / "folder_audit.json"
    out_md   = ROOT / "01_raw_data" / "FOLDER_AUDIT.md"
    out_json.write_text(json.dumps(report, indent=2))

    lines = [
        "# 01_raw_data Folder Audit",
        f"- ok: {ok}",
        "## Problems"
    ] + ([f"- {m}" for m in problems] or ["- (none)"]) + [
        "",
        "## Findings"
    ] + ([f"- {m}" for m in findings] or ["- (none)"])

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_json} and {out_md}")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
