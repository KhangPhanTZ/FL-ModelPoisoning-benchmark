import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  AGGREGATION_LABELS,
  type Aggregation,
  type Experiment,
} from '../lib/constants';
import { useExperiments } from '../hooks/useExperiments';

interface Point {
  tau: number;
  asr: number;
  evasion: number;
  n: number;
}

/** Average final ASR / Evasion per tau for one (defense) GeoTox sweep. */
function buildPoints(experiments: Experiment[], agg: Aggregation): Point[] {
  const byTau = new Map<number, { asr: number; evasion: number; n: number }>();
  for (const e of experiments) {
    if (e.aggregation !== agg || e.attack !== 'geotox' || e.tau == null) continue;
    if (e.attackUntil) continue; // exclude durability runs
    const cur = byTau.get(e.tau) ?? { asr: 0, evasion: 0, n: 0 };
    cur.asr += e.metrics.finalAsr;
    cur.evasion += e.metrics.finalEvasion;
    cur.n += 1;
    byTau.set(e.tau, cur);
  }
  return [...byTau.entries()]
    .map(([tau, v]) => ({ tau, asr: v.asr / v.n, evasion: v.evasion / v.n, n: v.n }))
    .sort((a, b) => a.tau - b.tau);
}

export function TradeoffPage() {
  const { experiments, loading } = useExperiments();

  const defenses = useMemo(() => {
    const set = new Set<Aggregation>();
    for (const e of experiments) {
      if (e.attack === 'geotox' && e.tau != null && !e.attackUntil) set.add(e.aggregation);
    }
    return [...set];
  }, [experiments]);

  const [defense, setDefense] = useState<Aggregation | null>(null);
  const activeDefense = defense && defenses.includes(defense) ? defense : defenses[0] ?? null;

  const points = useMemo(
    () => (activeDefense ? buildPoints(experiments, activeDefense) : []),
    [experiments, activeDefense],
  );

  if (loading) {
    return <div className="text-slate-500 dark:text-slate-400">Loading…</div>;
  }

  if (defenses.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h1 className="text-2xl font-semibold mb-2">GeoTox trade-off</h1>
        <p className="text-sm text-slate-500 dark:text-slate-400">
          No GeoTox sweep found. Run <code>run_quickstart.py</code> (which sweeps the stealth knob
          τ), copy the results into <code>public/results/</code>, regenerate the manifest, and
          reload.
        </p>
      </div>
    );
  }

  const maxN = Math.max(...points.map((p) => p.n), 0);

  return (
    <div className="space-y-5">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">GeoTox Evasion ↔ ASR trade-off</h1>
        <span className="text-sm text-slate-500">averaged over {maxN} seed(s)</span>
      </div>

      <p className="text-sm text-slate-600 dark:text-slate-300 max-w-3xl">
        The stealth knob <strong>τ</strong> trades off detectability against backdoor strength.
        Low τ keeps the backdoor direction (high ASR, easily filtered); high τ aligns the update
        with benign clients (high evasion, weaker backdoor). The shape of these two curves per
        defense is the core RQ1 result.
      </p>

      {/* Defense selector */}
      <div className="flex flex-wrap gap-2">
        {defenses.map((d) => (
          <button
            key={d}
            type="button"
            onClick={() => setDefense(d)}
            className={`px-3 py-1.5 text-sm rounded-md border ${
              d === activeDefense
                ? 'bg-sky-600 text-white border-sky-600'
                : 'border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            {AGGREGATION_LABELS[d]}
          </button>
        ))}
      </div>

      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={points} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#94a3b833" />
            <XAxis
              dataKey="tau"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 12 }}
              label={{ value: 'τ (stealth)', position: 'insideBottom', offset: -10, fontSize: 12 }}
            />
            <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} width={44} />
            <Tooltip
              formatter={(value) => `${Number(value).toFixed(1)}%`}
              labelFormatter={(l) => `τ = ${l}`}
            />
            <Legend />
            <Line type="monotone" dataKey="asr" name="ASR (%)" stroke="#a855f7" strokeWidth={2} dot />
            <Line
              type="monotone"
              dataKey="evasion"
              name="Evasion (%)"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Numeric table */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-100 dark:bg-slate-800/60 text-slate-600 dark:text-slate-300">
            <tr>
              <th className="text-left px-3 py-2">τ</th>
              <th className="text-right px-3 py-2">ASR (%)</th>
              <th className="text-right px-3 py-2">Evasion (%)</th>
              <th className="text-right px-3 py-2">seeds</th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.tau} className="border-t border-slate-100 dark:border-slate-800">
                <td className="px-3 py-2 font-mono">{p.tau}</td>
                <td className="px-3 py-2 text-right">{p.asr.toFixed(1)}</td>
                <td className="px-3 py-2 text-right">{p.evasion.toFixed(1)}</td>
                <td className="px-3 py-2 text-right text-slate-400">{p.n}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
