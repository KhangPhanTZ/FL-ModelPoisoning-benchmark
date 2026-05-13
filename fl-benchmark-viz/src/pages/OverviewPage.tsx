import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  PARTITION_LABELS,
  type Aggregation,
  type Attack,
  type Experiment,
} from '../lib/constants';
import { meanFinalAccuracy } from '../lib/metrics';
import { useAvailableDimensions, useExperiments } from '../hooks/useExperiments';
import { StatCard } from '../components/StatCard';
import { MiniHeatmap } from '../components/MiniHeatmap';

export function OverviewPage() {
  const { experiments, loading, progress, warnings, error } = useExperiments();
  const dims = useAvailableDimensions();

  const stats = useMemo(() => computeOverviewStats(experiments), [experiments]);

  if (loading) {
    return (
      <div className="text-slate-500 dark:text-slate-400">
        Loading {progress.loaded} / {progress.total}
        {progress.current ? ` — ${progress.current}` : ''}…
      </div>
    );
  }

  if (error && experiments.length === 0) {
    return (
      <div className="rounded-lg border border-rose-300 dark:border-rose-700 bg-rose-50 dark:bg-rose-950/40 p-4 text-rose-700 dark:text-rose-300">
        <div className="font-medium">Could not load benchmark data</div>
        <div className="text-sm mt-1">{error}</div>
        <ol className="text-sm mt-2 list-decimal list-inside">
          <li>Run <code>scripts/copy-results.sh</code> (or <code>.ps1</code> on Windows)</li>
          <li>Run <code>python scripts/generate-manifest.py</code></li>
          <li>Reload the page</li>
        </ol>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Overview</h1>
        <span className="text-sm text-slate-500">
          {experiments.length} experiments loaded
        </span>
      </div>

      {warnings.length > 0 && (
        <div className="rounded border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-950/40 p-3 text-sm text-amber-800 dark:text-amber-300">
          {warnings.length} file(s) skipped — see browser console for details.
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Configs loaded"
          value={experiments.length}
          hint={`${dims.aggregations.length} defenses × ${dims.attacks.length} attacks × ${dims.partitions.length} parts × ${dims.malicious.length} mal counts`}
        />
        <StatCard
          label="Best defense (avg final acc)"
          value={
            stats.bestDefense
              ? `${stats.bestDefense.acc.toFixed(2)}%`
              : '—'
          }
          hint={stats.bestDefense ? AGGREGATION_LABELS[stats.bestDefense.agg] : undefined}
          accent="green"
        />
        <StatCard
          label="Worst (attack, defense)"
          value={
            stats.worstCell
              ? `${stats.worstCell.acc.toFixed(2)}%`
              : '—'
          }
          hint={
            stats.worstCell
              ? `${AGGREGATION_LABELS[stats.worstCell.agg]} × ${ATTACK_LABELS[stats.worstCell.atk]}`
              : undefined
          }
          accent="rose"
        />
        <StatCard
          label="Most robust defense (min variance)"
          value={
            stats.mostRobust
              ? AGGREGATION_LABELS[stats.mostRobust.agg]
              : '—'
          }
          hint={
            stats.mostRobust
              ? `acc spread ${stats.mostRobust.spread.toFixed(2)}%`
              : undefined
          }
          accent="blue"
        />
      </div>

      <section className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="font-medium">Defense × Attack — average final accuracy</h2>
          <Link to="/matrix" className="text-sm text-sky-600 hover:underline">
            Open full matrix →
          </Link>
        </div>
        <MiniHeatmap
          experiments={experiments}
          aggregations={dims.aggregations}
          attacks={dims.attacks}
        />
      </section>

      <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <RankingTable title="Top 5 configs (final accuracy)" rows={stats.top5} accent="green" />
        <RankingTable title="Bottom 5 configs (final accuracy)" rows={stats.bottom5} accent="rose" />
      </section>
    </div>
  );
}

function RankingTable({
  title,
  rows,
  accent,
}: {
  title: string;
  rows: Experiment[];
  accent: 'green' | 'rose';
}) {
  const accentText = accent === 'green' ? 'text-emerald-600' : 'text-rose-600';
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
      <h2 className="font-medium mb-3">{title}</h2>
      <table className="w-full text-sm">
        <thead className="text-slate-500 dark:text-slate-400 text-xs uppercase">
          <tr>
            <th className="text-left font-medium pb-1">Defense</th>
            <th className="text-left font-medium pb-1">Attack</th>
            <th className="text-left font-medium pb-1">Part.</th>
            <th className="text-left font-medium pb-1">m</th>
            <th className="text-right font-medium pb-1">Final acc</th>
          </tr>
        </thead>
        <tbody>
          {rows.length === 0 ? (
            <tr>
              <td colSpan={5} className="text-slate-400 py-2">
                No data
              </td>
            </tr>
          ) : (
            rows.map((e) => (
              <tr key={e.filename} className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1.5">{AGGREGATION_LABELS[e.aggregation]}</td>
                <td className="py-1.5">{ATTACK_LABELS[e.attack]}</td>
                <td className="py-1.5">{PARTITION_LABELS[e.partition]}</td>
                <td className="py-1.5">{e.malicious}</td>
                <td className={`py-1.5 text-right font-medium ${accentText}`}>
                  {e.metrics.finalAccuracy.toFixed(2)}%
                </td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

interface OverviewStats {
  bestDefense: { agg: Aggregation; acc: number } | null;
  worstCell: { agg: Aggregation; atk: Attack; acc: number } | null;
  mostRobust: { agg: Aggregation; spread: number } | null;
  top5: Experiment[];
  bottom5: Experiment[];
}

function computeOverviewStats(experiments: Experiment[]): OverviewStats {
  if (experiments.length === 0) {
    return { bestDefense: null, worstCell: null, mostRobust: null, top5: [], bottom5: [] };
  }

  // Best defense: highest avg final accuracy across all its experiments
  const byAgg = groupBy(experiments, (e) => e.aggregation);
  let bestDefense: OverviewStats['bestDefense'] = null;
  let mostRobust: OverviewStats['mostRobust'] = null;
  for (const [agg, list] of byAgg.entries()) {
    const accs = list.map((e) => e.metrics.finalAccuracy);
    const avg = meanFinalAccuracy(list);
    const spread = Math.max(...accs) - Math.min(...accs);
    if (!bestDefense || avg > bestDefense.acc) bestDefense = { agg, acc: avg };
    if (!mostRobust || spread < mostRobust.spread) mostRobust = { agg, spread };
  }

  // Worst (attack × defense) cell: lowest avg final accuracy
  const cells = new Map<string, { agg: Aggregation; atk: Attack; list: Experiment[] }>();
  for (const e of experiments) {
    const k = `${e.aggregation}__${e.attack}`;
    if (!cells.has(k))
      cells.set(k, { agg: e.aggregation, atk: e.attack, list: [] });
    cells.get(k)!.list.push(e);
  }
  let worstCell: OverviewStats['worstCell'] = null;
  for (const { agg, atk, list } of cells.values()) {
    const avg = meanFinalAccuracy(list);
    if (!worstCell || avg < worstCell.acc) worstCell = { agg, atk, acc: avg };
  }

  const sorted = [...experiments].sort((a, b) => b.metrics.finalAccuracy - a.metrics.finalAccuracy);
  return {
    bestDefense,
    worstCell,
    mostRobust,
    top5: sorted.slice(0, 5),
    bottom5: sorted.slice(-5).reverse(),
  };
}

function groupBy<T, K>(arr: T[], key: (t: T) => K): Map<K, T[]> {
  const m = new Map<K, T[]>();
  for (const item of arr) {
    const k = key(item);
    if (!m.has(k)) m.set(k, []);
    m.get(k)!.push(item);
  }
  return m;
}
