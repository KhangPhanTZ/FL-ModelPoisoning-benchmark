import { useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  AGGREGATION_COLORS,
  AGGREGATION_LABELS,
  ATTACKS,
  ATTACK_DESCRIPTIONS,
  ATTACK_LABELS,
  PARTITION_LABELS,
  type Aggregation,
  type Attack,
  type Experiment,
} from '../lib/constants';
import { useAvailableDimensions, useExperiments } from '../hooks/useExperiments';
import { MultiLineChart } from '../components/MultiLineChart';
import { ExportButton } from '../components/ExportButton';
import { buildSeries } from '../lib/series';
import { seriesToCsv } from '../lib/export';

export function AttacksPage() {
  const { experiments, loading } = useExperiments();
  const dims = useAvailableDimensions();
  const tabAttacks = ATTACKS.filter((a) => dims.attacks.includes(a));
  const [activeTab, setActiveTab] = useState<Attack>(tabAttacks[0] ?? 'lie');

  if (loading) {
    return <div className="text-slate-400">Loading…</div>;
  }

  const expsForAttack = experiments.filter((e) => e.attack === activeTab);
  const desc = ATTACK_DESCRIPTIONS[activeTab];

  return (
    <div className="space-y-5">
      <h1 className="text-2xl font-semibold">Attack deep-dive</h1>

      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-800">
        {tabAttacks.map((a) => (
          <button
            key={a}
            type="button"
            onClick={() => setActiveTab(a)}
            className={`px-4 py-2 text-sm border-b-2 -mb-px ${
              a === activeTab
                ? 'border-sky-500 text-sky-600 dark:text-sky-400 font-medium'
                : 'border-transparent text-slate-500 hover:text-slate-900 dark:hover:text-slate-100'
            }`}
          >
            {ATTACK_LABELS[a]}
          </button>
        ))}
      </div>

      <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-5">
        <div className="flex items-baseline gap-3">
          <h2 className="text-lg font-medium">{desc.name}</h2>
          <span className="text-xs text-slate-500">{desc.paper}</span>
        </div>
        <p className="mt-2 text-sm text-slate-600 dark:text-slate-400 max-w-3xl">
          {desc.description}
        </p>
        <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <CodeRow label="Formula" code={desc.formula} />
          <CodeRow label="Parameter" code={desc.parameter} />
        </div>
      </section>

      <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        <FinalAccBarChart experiments={expsForAttack} />
        <ConvergenceBarChart experiments={expsForAttack} />
      </section>

      <section
        data-export="attacks-curves"
        className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
      >
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-medium">Accuracy curves — all defenses, m=4 (IID)</h3>
          <ExportButton
            targetSelector='[data-export="attacks-curves"]'
            filename={`attack_${activeTab}_curves_acc_iid_m4`}
            getCsv={() =>
              seriesToCsv(
                buildSeries(
                  experiments.filter(
                    (e) =>
                      e.attack === activeTab && e.partition === 'iid' && e.malicious === 4,
                  ),
                  'accuracy',
                ),
              )
            }
          />
        </div>
        <CurvesForAttack experiments={experiments} attack={activeTab} />
      </section>

      {activeTab === 'model_replacement' && (
        <section
          data-export="attacks-asr"
          className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
        >
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">
              Backdoor success rate (ASR) over rounds — IID, m=4
            </h3>
            <ExportButton
              targetSelector='[data-export="attacks-asr"]'
              filename="attack_model_replacement_asr_iid_m4"
              getCsv={() =>
                seriesToCsv(
                  buildSeries(
                    experiments.filter(
                      (e) =>
                        e.attack === 'model_replacement' &&
                        e.partition === 'iid' &&
                        e.malicious === 4,
                    ),
                    'asr',
                  ),
                )
              }
            />
          </div>
          <AsrCurves experiments={experiments} />
        </section>
      )}
    </div>
  );
}

function CodeRow({ label, code }: { label: string; code: string }) {
  return (
    <div className="text-sm">
      <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <code className="block mt-1 px-3 py-1.5 rounded bg-slate-100 dark:bg-slate-800 font-mono text-sm">
        {code}
      </code>
    </div>
  );
}

interface BarRow {
  defense: string;
  agg: Aggregation;
  m2: number | null;
  m4: number | null;
  m6: number | null;
}

function buildBarRows(experiments: Experiment[], pickValue: (e: Experiment) => number | null): BarRow[] {
  const byAgg = new Map<Aggregation, { m2: number[]; m4: number[]; m6: number[] }>();
  for (const e of experiments) {
    if (e.partition !== 'iid') continue;
    if (!byAgg.has(e.aggregation))
      byAgg.set(e.aggregation, { m2: [], m4: [], m6: [] });
    const bucket = byAgg.get(e.aggregation)!;
    const v = pickValue(e);
    if (v == null) continue;
    if (e.malicious === 2) bucket.m2.push(v);
    else if (e.malicious === 4) bucket.m4.push(v);
    else if (e.malicious === 6) bucket.m6.push(v);
  }
  const avg = (arr: number[]) => (arr.length === 0 ? null : arr.reduce((a, b) => a + b, 0) / arr.length);
  return [...byAgg.entries()].map(([agg, b]) => ({
    defense: AGGREGATION_LABELS[agg],
    agg,
    m2: avg(b.m2),
    m4: avg(b.m4),
    m6: avg(b.m6),
  }));
}

function FinalAccBarChart({ experiments }: { experiments: Experiment[] }) {
  const rows = useMemo(
    () => buildBarRows(experiments, (e) => e.metrics.finalAccuracy),
    [experiments],
  );
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
      <h3 className="font-medium mb-2">Final accuracy by defense × #malicious (IID)</h3>
      <BarTriple rows={rows} yLabel="Accuracy %" yDomain={[0, 100]} />
    </div>
  );
}

function ConvergenceBarChart({ experiments }: { experiments: Experiment[] }) {
  const rows = useMemo(
    () => buildBarRows(experiments, (e) => e.metrics.convergenceRound),
    [experiments],
  );
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
      <h3 className="font-medium mb-2">Convergence round (IID) — lower is faster</h3>
      <BarTriple rows={rows} yLabel="Round" yDomain={[0, 50]} />
    </div>
  );
}

function BarTriple({
  rows,
  yLabel,
  yDomain,
}: {
  rows: BarRow[];
  yLabel: string;
  yDomain?: [number, number];
}) {
  if (rows.length === 0) {
    return <div className="h-[260px] flex items-center justify-center text-slate-400">No data</div>;
  }
  return (
    <div style={{ width: '100%', height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={rows} margin={{ top: 4, right: 8, bottom: 4, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
          <XAxis dataKey="defense" stroke="currentColor" className="text-slate-500 text-xs" />
          <YAxis
            domain={yDomain}
            stroke="currentColor"
            className="text-slate-500 text-xs"
            label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 8 }}
          />
          <Tooltip
            formatter={(v: unknown) =>
              typeof v === 'number' ? v.toFixed(2) : (v as string)
            }
          />
          <Legend />
          <Bar dataKey="m2" name="m=2" fill="#94a3b8" isAnimationActive animationDuration={300}>
            {rows.map((r) => (
              <Cell key={`m2-${r.agg}`} fill={tint(AGGREGATION_COLORS[r.agg], 0.7)} />
            ))}
          </Bar>
          <Bar dataKey="m4" name="m=4" fill="#64748b" isAnimationActive animationDuration={300}>
            {rows.map((r) => (
              <Cell key={`m4-${r.agg}`} fill={tint(AGGREGATION_COLORS[r.agg], 0.45)} />
            ))}
          </Bar>
          <Bar dataKey="m6" name="m=6" fill="#334155" isAnimationActive animationDuration={300}>
            {rows.map((r) => (
              <Cell key={`m6-${r.agg}`} fill={AGGREGATION_COLORS[r.agg]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function tint(hex: string, lightenBy: number): string {
  // Lighten by mixing with white. lightenBy=0 → original, 1 → white.
  const m = hex.match(/^#([0-9a-f]{6})$/i);
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  let r = (n >> 16) & 0xff;
  let g = (n >> 8) & 0xff;
  let b = n & 0xff;
  r = Math.round(r + (255 - r) * lightenBy);
  g = Math.round(g + (255 - g) * lightenBy);
  b = Math.round(b + (255 - b) * lightenBy);
  return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
}

function CurvesForAttack({
  experiments,
  attack,
}: {
  experiments: Experiment[];
  attack: Attack;
}) {
  const filtered = experiments.filter(
    (e) => e.attack === attack && e.partition === 'iid' && e.malicious === 4,
  );
  const series = buildSeries(filtered, 'accuracy');
  return <MultiLineChart series={series} yLabel="Accuracy (%)" yDomain={[0, 100]} height={300} />;
}

function AsrCurves({ experiments }: { experiments: Experiment[] }) {
  const filtered = experiments.filter(
    (e) =>
      e.attack === 'model_replacement' &&
      e.partition === 'iid' &&
      e.malicious === 4,
  );
  const series = buildSeries(filtered, 'asr');
  return (
    <div>
      <MultiLineChart series={series} yLabel="ASR (%)" yDomain={[0, 100]} height={300} />
      <div className="text-xs text-slate-500 dark:text-slate-400 mt-2">
        ASR = fraction of trigger-stamped test inputs classified as target class 7.
        Higher = backdoor more successful (worse for the defender). Showing {PARTITION_LABELS.iid}, m=4.
      </div>
    </div>
  );
}
