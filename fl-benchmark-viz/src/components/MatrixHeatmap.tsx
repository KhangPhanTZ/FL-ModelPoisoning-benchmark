import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  HEATMAP_STOPS,
  colorForAccuracy,
  type Aggregation,
  type Attack,
  type Experiment,
} from '../lib/constants';
import { MATRIX_METRIC_LABELS, type CellInfo, type MatrixMetric } from '../lib/matrix-metrics';
import { stddev } from '../lib/metrics';

interface Props {
  experiments: Experiment[];
  baselineByAgg?: Map<Aggregation, number>;
  aggregations: Aggregation[];
  attacks: Attack[];
  metric: MatrixMetric;
  onCellClick?: (cell: CellInfo) => void;
}

function aggregateCell(
  cellExps: Experiment[],
  metric: MatrixMetric,
  baseline?: number,
): number {
  if (cellExps.length === 0) return NaN;
  switch (metric) {
    case 'finalAcc': {
      const vals = cellExps.map((e) => e.metrics.finalAccuracy);
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    case 'maxAcc': {
      const vals = cellExps.map((e) => e.metrics.maxAccuracy);
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    case 'degradation': {
      if (baseline === undefined) return NaN;
      const avgFinal =
        cellExps.reduce((a, e) => a + e.metrics.finalAccuracy, 0) / cellExps.length;
      return baseline - avgFinal;
    }
    case 'convergence': {
      const vals = cellExps
        .map((e) => e.metrics.convergenceRound)
        .filter((v): v is number => v !== null);
      if (vals.length === 0) return NaN;
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    }
    case 'stability': {
      // average of round-50 std across cell, but here we use already-computed stabilityStd
      const vals = cellExps.map((e) => e.metrics.stabilityStd);
      return vals.reduce((a, b) => a + b, 0) / vals.length;
    }
  }
}

function colorForCell(metric: MatrixMetric, value: number): string {
  if (Number.isNaN(value)) return 'transparent';
  switch (metric) {
    case 'finalAcc':
    case 'maxAcc':
      return colorForAccuracy(value);
    case 'degradation': {
      // 0 (no drop) = green, >40 = red
      const acc = 100 - value;
      return colorForAccuracy(Math.max(0, Math.min(100, acc)));
    }
    case 'convergence': {
      // earlier round = green, later = red. 50 rounds total.
      const fakeAcc = 100 - (value / 50) * 100;
      return colorForAccuracy(Math.max(0, Math.min(100, fakeAcc)));
    }
    case 'stability': {
      // lower std = green. clamp std at 20 → red
      const fakeAcc = 100 - Math.min(100, (value / 20) * 100);
      return colorForAccuracy(fakeAcc);
    }
  }
}

function formatCell(metric: MatrixMetric, value: number): string {
  if (Number.isNaN(value)) return '—';
  switch (metric) {
    case 'finalAcc':
    case 'maxAcc':
      return value.toFixed(1);
    case 'degradation':
      return `${value >= 0 ? '−' : '+'}${Math.abs(value).toFixed(1)}`;
    case 'convergence':
      return value.toFixed(0);
    case 'stability':
      return value.toFixed(2);
  }
}

export function MatrixHeatmap({
  experiments,
  aggregations,
  attacks,
  metric,
  baselineByAgg,
  onCellClick,
}: Props) {
  const grid = aggregations.map((agg) =>
    attacks.map<CellInfo>((atk) => {
      const cellExps = experiments.filter((e) => e.aggregation === agg && e.attack === atk);
      const value = aggregateCell(cellExps, metric, baselineByAgg?.get(agg));
      return { aggregation: agg, attack: atk, value, count: cellExps.length, experiments: cellExps };
    }),
  );

  // For inferring how spread-out the values are (just for tooltip context)
  const flatVals = grid.flat().map((c) => c.value).filter((v) => !Number.isNaN(v));
  const valStd = stddev(flatVals);

  return (
    <div className="overflow-auto">
      <table className="text-sm border-separate border-spacing-1.5">
        <thead>
          <tr>
            <th className="px-2" />
            {attacks.map((atk) => (
              <th
                key={atk}
                className="px-2 py-1 text-slate-600 dark:text-slate-400 font-medium text-xs uppercase tracking-wider whitespace-nowrap"
              >
                {ATTACK_LABELS[atk]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.map((row, i) => (
            <tr key={aggregations[i]}>
              <td className="px-2 py-1 text-right font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap">
                {AGGREGATION_LABELS[aggregations[i]]}
              </td>
              {row.map((cell) => {
                const empty = Number.isNaN(cell.value);
                const bg = colorForCell(metric, cell.value);
                return (
                  <td
                    key={cell.attack}
                    onClick={empty || !onCellClick ? undefined : () => onCellClick(cell)}
                    className={`min-w-[100px] h-16 text-center rounded font-semibold text-slate-900 border border-slate-300 dark:border-slate-700 ${
                      empty ? 'opacity-40' : 'cursor-pointer hover:ring-2 hover:ring-offset-1 hover:ring-sky-500'
                    }`}
                    style={{ background: bg }}
                    title={
                      empty
                        ? 'no data'
                        : `${AGGREGATION_LABELS[cell.aggregation]} × ${ATTACK_LABELS[cell.attack]}\n${MATRIX_METRIC_LABELS[metric]}: ${formatCell(metric, cell.value)}\n${cell.count} run(s)\nspread σ across grid: ${valStd.toFixed(2)}`
                    }
                  >
                    {formatCell(metric, cell.value)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function HeatmapLegend() {
  return (
    <div className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
      <span>scale:</span>
      {HEATMAP_STOPS.map((s) => (
        <span key={s.label} className="flex items-center gap-1">
          <span
            className="inline-block w-4 h-4 rounded border border-slate-300 dark:border-slate-700"
            style={{ background: s.color }}
          />
          {s.label}
        </span>
      ))}
    </div>
  );
}
