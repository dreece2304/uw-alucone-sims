#!/usr/bin/env python3
import sys, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
reg_path = ROOT / "_shared" / "registry.json"

def main():
    ok = True
    if not reg_path.exists():
        print(f"ERROR: missing {reg_path}", file=sys.stderr); return 2
    reg = json.loads(reg_path.read_text())
    out = {"schema": reg.get("schema",""), "checked_at": datetime.utcnow().isoformat()+"Z",
           "files": [], "dirs": [], "urls": []}
    for k,v in reg.get("files",{}).items():
        p = (ROOT/v).resolve(); exists = p.exists()
        out["files"].append({"key":k,"path":v,"exists":bool(exists)})
        ok &= bool(exists)
    for k,v in reg.get("dirs",{}).items():
        d = (ROOT/v).resolve(); exists = d.exists()
        out["dirs"].append({"key":k,"path":v,"exists":bool(exists)})
        ok &= bool(exists)
    for k,v in reg.get("urls",{}).items():
        out["urls"].append({"key":k,"url":v})
    (ROOT/"_shared/state").mkdir(parents=True, exist_ok=True)
    (ROOT/"_shared/state/REGISTRY_VALIDATION.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Registry validation ({out['checked_at']})", f"- schema: {out['schema']}", "## Files"]
    lines += [f"- {r['key']}: {r['path']}  -->  {'OK' if r['exists'] else 'MISSING'}" for r in out["files"]]
    lines += ["## Directories"]
    lines += [f"- {r['key']}: {r['path']}  -->  {'OK' if r['exists'] else 'MISSING'}" for r in out["dirs"]]
    lines += ["## URLs (not checked on disk)"]
    lines += [f"- {r['key']}: {r['url']}" for r in out["urls"]]
    (ROOT/"00_documentation/REGISTRY_AUDIT.md").write_text("\n".join(lines)+"\n")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())