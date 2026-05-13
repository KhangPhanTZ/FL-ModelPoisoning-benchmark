# Copy CSV results from the parent benchmark project into public/results/.
# Excludes the summary file (we only want per-config CSVs).
#
# Default layout assumes this viz lives inside the benchmark repo:
#   <repo-root>\
#     +-- results\                  (canonical benchmark output)
#     +-- fl-benchmark-viz\         (this subproject)
#         +-- public\results\       (copy used by the web app)
param(
    [string]$Src = (Join-Path $PSScriptRoot "..\..\results")
)

$ErrorActionPreference = "Stop"

$ProjectDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$Dst = Join-Path $ProjectDir "public\results"

if (-not (Test-Path $Src)) {
    Write-Error "Source directory not found: $Src"
    exit 1
}

New-Item -ItemType Directory -Force -Path $Dst | Out-Null

$count = 0
Get-ChildItem -Path $Src -Filter *.csv | ForEach-Object {
    if ($_.Name -eq "experiment_summary.csv") {
        return
    }
    Copy-Item -Path $_.FullName -Destination (Join-Path $Dst $_.Name) -Force
    $count++
}

Write-Host "Copied $count CSV file(s) -> $Dst"
Write-Host "Next: python scripts\generate-manifest.py"
