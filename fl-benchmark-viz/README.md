# FL-ModelPoisoning Benchmark — Visualizer

Interactive React app that loads the CSV results from the parent
[`FL-ModelPoisoning-benchmark`](../) project (this app is its `fl-benchmark-viz/`
subdirectory) and lets you explore how each aggregation rule performs under each
model-poisoning attack.

- 5 routes: **Overview**, **Training Curves**, **Defense × Attack Matrix**,
  **Attack Deep-Dive**, **Config Explorer**
- Pure client-side (no backend, no DB). CSVs live in `public/results/` and are
  fetched + parsed in the browser
- Supports the current 3 aggregations × 3 attacks setup *and* the future
  6 aggregations × 4 attacks setup (`multi_krum`, `bulyan`, `fltrust`, `none`)
  — UI is data-driven, no hard-coded lists
- Export every chart as PNG / SVG / CSV
- Filter state encoded in URL query params (`?f=…`) so links are shareable
- Light / dark theme, persisted in `localStorage`

## Requirements

- Node ≥ 18 (tested with 20 LTS via nvm)
- Python ≥ 3.8 (for the manifest generator)

## Install

```bash
git clone <repo>
cd fl-benchmark-viz
npm install
```

## Load benchmark results

The app reads CSVs at runtime from `public/results/`, with `manifest.json`
listing the available files. Two steps:

```bash
# 1. Copy CSVs from the Python benchmark (excludes experiment_summary.csv)
#    Linux/Mac:
bash scripts/copy-results.sh
#    Windows PowerShell:
powershell -File scripts\copy-results.ps1

# 2. Regenerate the manifest from whatever's now in public/results/
python scripts/generate-manifest.py
```

Both scripts assume this viz lives inside the benchmark repo:

```
FL-ModelPoisoning-benchmark/        ← repo root
├── results/*.csv                   ← canonical benchmark output (source)
└── fl-benchmark-viz/               ← this subproject
    └── public/results/             ← copy used by the web app (gitignored)
```

If your Python results live elsewhere, pass the path explicitly:

```bash
bash scripts/copy-results.sh /absolute/path/to/results
```

> **Note:** `public/results/` is gitignored — clone, run the two scripts above,
> then `npm run dev`. The CSVs at the repo root are the source of truth.

## Dev / build

```bash
npm run dev      # http://localhost:5173
npm run build    # static bundle in dist/
npm run preview  # serve the built bundle locally
```

## Filename convention

CSV files **must** match this regex (mirrored in TS and Python):

```
{aggregation}_{attack}_{partition}_m{malicious}.csv
```

with the allowed values:

| dimension   | values                                                       |
| ----------- | ------------------------------------------------------------ |
| aggregation | `mean`, `median`, `krum`, `multi_krum`, `bulyan`, `fltrust`  |
| attack      | `none`, `lie`, `minmax`, `model_replacement`                 |
| partition   | `iid`, `noniid`                                              |
| malicious   | any non-negative int                                         |

The regex uses **longest-alternative-first** ordering so `multi_krum` and
`model_replacement` parse correctly. See [`src/lib/constants.ts`](src/lib/constants.ts).

Each CSV is expected to have columns: `round,loss,accuracy,asr,timestamp`.
50 rows per experiment is the assumed cadence.

## ASR semantics

The `asr` column is only a real attack-success-rate for
`attack = model_replacement`. For LIE / Min-Max / none it is the baseline
backdoor-trigger noise (~1–3%) and is rendered muted (50% opacity, dashed) in
the UI; the **Attacks** page only shows the ASR chart on the
`model_replacement` tab.

## Architecture

```
src/
├── App.tsx              ← Router + DataProvider boot
├── context/DataContext  ← loads manifest + all CSVs on mount
├── hooks/               ← useExperiments, useFilteredExperiments, useDarkMode
├── lib/
│   ├── constants.ts     ← types, colors, regex, attack docs
│   ├── config-parser.ts ← filename → ConfigKey
│   ├── data-loader.ts   ← fetch + papaparse, batched (8 parallel)
│   ├── metrics.ts       ← finalAcc, maxAcc, convergenceRound, stabilityStd
│   ├── series.ts        ← Recharts series builders
│   ├── url-state.ts     ← JSON-encoded query-param state hook
│   └── export.ts        ← PNG / SVG (html-to-image) + CSV (blob)
├── components/
│   ├── Layout.tsx       ← header + nav + dark toggle
│   ├── FilterSidebar.tsx
│   ├── MultiLineChart.tsx
│   ├── MatrixHeatmap.tsx
│   ├── MiniHeatmap.tsx
│   ├── StatCard.tsx
│   ├── ExportButton.tsx
│   └── Modal.tsx
└── pages/               ← one file per route
```

## Deploy

The output is a static bundle — host it anywhere.

### Vercel / Netlify

Default settings work. Build command `npm run build`, output directory `dist`.

After deploy, **refresh `public/results/`** if the benchmark re-runs: re-run
`copy-results` + `generate-manifest`, commit, redeploy. (Both scripts can run
at build time on most platforms.)

### GitHub Pages

Set `base` in `vite.config.ts` to `'/<repo-name>/'` if the site is served from
a project path rather than the apex domain. Then:

```bash
npm run build
# upload dist/ to gh-pages branch via your preferred action
```

A typical GitHub Actions workflow looks like:

```yaml
- run: bash scripts/copy-results.sh
- run: python scripts/generate-manifest.py
- run: npm ci && npm run build
- uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: dist
```

## Common tweaks

- **Add a new aggregation** (e.g. `fltrust`): drop `fltrust_*.csv` into the
  benchmark output, re-copy + re-manifest. The UI will surface it automatically.
- **Add a baseline `attack=none` set**: enables the **Degradation vs no-attack**
  metric on the Matrix page (otherwise that radio is disabled).
- **Different number of rounds**: nothing is hard-coded; charts auto-scale on
  X-axis. The Convergence-round metric assumes 1-indexed rounds.

## Troubleshooting

- *"Could not load benchmark data"* — `public/results/manifest.json` is
  missing. Run `python scripts/generate-manifest.py`.
- *Heatmap cells say "—"* — that (defense × attack) combo has no CSV. Either
  the benchmark didn't produce it, or the filename doesn't match the regex.
  Check the browser console for skip warnings.
- *Charts look squished* — the layout is desktop-first; on narrow screens the
  filter sidebar wraps below.

## License

Same license as the parent `FL-ModelPoisoning-benchmark` project.
