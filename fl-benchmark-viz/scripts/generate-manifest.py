#!/usr/bin/env python3
"""Scan public/results/*.csv and emit public/results/manifest.json.

The web app uses this manifest at runtime to know which experiments exist
(no server-side directory listing in static deployments).
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "public" / "results"

# Mirror src/lib/constants.ts: longer alternatives first to avoid partial-match.
FILE_PATTERN = re.compile(
    r"^(multi_krum|mean|median|krum|bulyan|fltrust)"
    r"_(model_replacement|none|lie|minmax)"
    r"_(iid|noniid)_m(\d+)\.csv$"
)


def main() -> int:
    if not RESULTS_DIR.is_dir():
        print(f"Error: {RESULTS_DIR} does not exist.", file=sys.stderr)
        print("Run scripts/copy-results.sh (or .ps1) first.", file=sys.stderr)
        return 1

    files: list[str] = []
    skipped: list[str] = []
    for csv in sorted(RESULTS_DIR.glob("*.csv")):
        name = csv.name
        if FILE_PATTERN.match(name):
            files.append(name)
        else:
            skipped.append(name)

    manifest = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "files": files,
    }
    out_path = RESULTS_DIR / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(files)} entries → {out_path.relative_to(PROJECT_DIR)}")
    if skipped:
        print(f"Skipped {len(skipped)} non-conforming file(s):")
        for s in skipped:
            print(f"  - {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
