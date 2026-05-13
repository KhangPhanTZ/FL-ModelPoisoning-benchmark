#!/usr/bin/env bash
# Copy CSV results from the parent benchmark project into public/results/.
# Excludes the summary file (we only want per-config CSVs).
#
# Default layout assumes this viz lives inside the benchmark repo:
#   <repo-root>/
#     ├── results/                  ← canonical benchmark output (source)
#     └── fl-benchmark-viz/         ← this subproject
#         └── public/results/       ← copy used by the web app (destination)
#
# Override with an absolute path:  ./scripts/copy-results.sh /path/to/results
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="${1:-$PROJECT_DIR/../results}"
DST="$PROJECT_DIR/public/results"

if [ ! -d "$SRC" ]; then
  echo "Error: source directory not found: $SRC" >&2
  echo "Pass an explicit path: ./scripts/copy-results.sh /path/to/results" >&2
  exit 1
fi

mkdir -p "$DST"

count=0
for f in "$SRC"/*.csv; do
  [ -e "$f" ] || continue
  base="$(basename "$f")"
  if [ "$base" = "experiment_summary.csv" ]; then
    continue
  fi
  cp "$f" "$DST/$base"
  count=$((count + 1))
done

echo "Copied $count CSV file(s) → $DST"
echo "Next: python scripts/generate-manifest.py"
