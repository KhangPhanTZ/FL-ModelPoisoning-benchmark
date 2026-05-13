import { useMemo, useState } from 'react';
import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  PARTITION_LABELS,
  type Aggregation,
  type Attack,
  type Experiment,
  type Partition,
} from '../lib/constants';
import { useAvailableDimensions, useExperiments } from '../hooks/useExperiments';
import { HeatmapLegend, MatrixHeatmap } from '../components/MatrixHeatmap';
import {
  MATRIX_METRIC_LABELS,
  type CellInfo,
  type MatrixMetric,
} from '../lib/matrix-metrics';
import { Modal } from '../components/Modal';
import { MultiLineChart } from '../components/MultiLineChart';
import { ExportButton } from '../components/ExportButton';
import { buildSeries, METRIC_LABELS, type Metric } from '../lib/series';
import { useUrlState } from '../lib/url-state';

function buildMatrixCsv(
  experiments: Experiment[],
  aggregations: Aggregation[],
  attacks: Attack[],
) {
  const rows = aggregations.map((agg) => {
    const row: Record<string, number | string> = { defense: agg };
    for (const atk of attacks) {
      const cell = experiments.filter((e) => e.aggregation === agg && e.attack === atk);
      const avg =
        cell.length === 0
          ? ''
          : cell.reduce((a, e) => a + e.metrics.finalAccuracy, 0) / cell.length;
      row[atk] = avg;
    }
    return row;
  });
  return { rows, columns: ['defense', ...attacks] };
}

interface MatrixFilter {
  partition: Partition;
  malicious: number;
  metric: MatrixMetric;
}

const DEFAULT_FILTER: MatrixFilter = {
  partition: 'iid',
  malicious: 4,
  metric: 'finalAcc',
};

export function MatrixPage() {
  const { experiments, loading } = useExperiments();
  const dims = useAvailableDimensions();
  const [filter, setFilter] = useUrlState<MatrixFilter>('m', DEFAULT_FILTER);
  const [selectedCell, setSelectedCell] = useState<CellInfo | null>(null);

  const filtered = useMemo(
    () =>
      experiments.filter(
        (e) => e.partition === filter.partition && e.malicious === filter.malicious,
      ),
    [experiments, filter.partition, filter.malicious],
  );

  // Baseline accuracy per aggregation when attack='none' (used by 'degradation' metric)
  const baselineByAgg = useMemo(() => {
    const map = new Map<Aggregation, number>();
    for (const e of filtered) {
      if (e.attack !== 'none') continue;
      map.set(e.aggregation, e.metrics.finalAccuracy);
    }
    return map;
  }, [filtered]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Defense × Attack matrix</h1>
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-500">
            {filtered.length} runs ·{' '}
            {dims.aggregations.length} defenses × {dims.attacks.length} attacks
          </span>
          <ExportButton
            targetSelector='[data-export="matrix-heatmap"]'
            filename={`matrix_${filter.metric}_${filter.partition}_m${filter.malicious}`}
            getCsv={() => buildMatrixCsv(filtered, dims.aggregations, dims.attacks)}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3 text-sm">
        <Toggle
          label="Partition"
          options={dims.partitions.map((p) => ({ value: p, label: PARTITION_LABELS[p] }))}
          current={filter.partition}
          onChange={(v) => setFilter((prev) => ({ ...prev, partition: v as Partition }))}
        />
        <Toggle
          label="Malicious"
          options={dims.malicious.map((m) => ({ value: m, label: `m=${m}` }))}
          current={filter.malicious}
          onChange={(v) => setFilter((prev) => ({ ...prev, malicious: v as number }))}
        />
        <Toggle
          label="Metric"
          options={(Object.keys(MATRIX_METRIC_LABELS) as MatrixMetric[]).map((m) => ({
            value: m,
            label: MATRIX_METRIC_LABELS[m],
          }))}
          current={filter.metric}
          onChange={(v) => setFilter((prev) => ({ ...prev, metric: v as MatrixMetric }))}
          disabled={(v) =>
            v === 'degradation' && !dims.attacks.includes('none')
          }
        />
        <div className="ml-auto">
          <HeatmapLegend />
        </div>
      </div>

      <div
        data-export="matrix-heatmap"
        className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
      >
        {loading ? (
          <div className="h-40 flex items-center justify-center text-slate-400">
            Loading…
          </div>
        ) : (
          <MatrixHeatmap
            experiments={filtered}
            aggregations={dims.aggregations}
            attacks={dims.attacks}
            metric={filter.metric}
            baselineByAgg={baselineByAgg}
            onCellClick={setSelectedCell}
          />
        )}
        {filter.metric === 'degradation' && !dims.attacks.includes('none') && (
          <div className="mt-3 text-xs text-amber-600 dark:text-amber-400">
            Degradation needs <code>attack=none</code> baseline runs (none loaded).
          </div>
        )}
      </div>

      <Modal
        open={selectedCell !== null}
        onClose={() => setSelectedCell(null)}
        title={
          selectedCell
            ? `${AGGREGATION_LABELS[selectedCell.aggregation]} × ${ATTACK_LABELS[selectedCell.attack]} — ${PARTITION_LABELS[filter.partition]} m${filter.malicious}`
            : ''
        }
        width="max-w-4xl"
      >
        {selectedCell && <CellDetail cell={selectedCell} />}
      </Modal>
    </div>
  );
}

function CellDetail({ cell }: { cell: CellInfo }) {
  const [metric, setMetric] = useState<Metric>('accuracy');
  const series = buildSeries(cell.experiments, metric);
  const yDomain: [number | 'auto', number | 'auto'] | undefined =
    metric === 'accuracy' || metric === 'asr' ? [0, 100] : undefined;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2 text-sm">
        {(['accuracy', 'loss', 'asr'] as Metric[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMetric(m)}
            className={`px-3 py-1 rounded-md border text-xs ${
              metric === m
                ? 'bg-slate-900 text-white border-slate-900 dark:bg-slate-100 dark:text-slate-900'
                : 'border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800'
            }`}
          >
            {METRIC_LABELS[m]}
          </button>
        ))}
      </div>
      <MultiLineChart series={series} yLabel={METRIC_LABELS[metric]} yDomain={yDomain} height={320} />
      <ExperimentList experiments={cell.experiments} />
    </div>
  );
}

function ExperimentList({ experiments }: { experiments: Experiment[] }) {
  return (
    <table className="w-full text-xs">
      <thead className="text-slate-500 uppercase tracking-wider">
        <tr>
          <th className="text-left font-medium pb-1">Run</th>
          <th className="text-right font-medium pb-1">Final acc</th>
          <th className="text-right font-medium pb-1">Max acc</th>
          <th className="text-right font-medium pb-1">Conv. round</th>
          <th className="text-right font-medium pb-1">Stab. σ</th>
        </tr>
      </thead>
      <tbody>
        {experiments.map((e) => (
          <tr key={e.filename} className="border-t border-slate-100 dark:border-slate-800">
            <td className="py-1.5 font-mono">{e.filename}</td>
            <td className="py-1.5 text-right">{e.metrics.finalAccuracy.toFixed(2)}</td>
            <td className="py-1.5 text-right">{e.metrics.maxAccuracy.toFixed(2)}</td>
            <td className="py-1.5 text-right">{e.metrics.convergenceRound ?? '—'}</td>
            <td className="py-1.5 text-right">{e.metrics.stabilityStd.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

interface ToggleProps<T extends string | number> {
  label: string;
  options: { value: T; label: string }[];
  current: T;
  disabled?: (value: T) => boolean;
  onChange: (v: T) => void;
}

function Toggle<T extends string | number>({
  label,
  options,
  current,
  disabled,
  onChange,
}: ToggleProps<T>) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs uppercase tracking-wider text-slate-500">{label}</span>
      <div className="flex rounded-md overflow-hidden border border-slate-200 dark:border-slate-700">
        {options.map((opt) => {
          const isDisabled = disabled?.(opt.value) ?? false;
          const active = opt.value === current;
          return (
            <button
              key={String(opt.value)}
              type="button"
              disabled={isDisabled}
              onClick={() => onChange(opt.value)}
              className={`px-3 py-1 text-xs whitespace-nowrap ${
                active
                  ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                  : 'bg-white dark:bg-slate-900 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
              } ${isDisabled ? 'opacity-40 cursor-not-allowed' : ''}`}
            >
              {opt.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
