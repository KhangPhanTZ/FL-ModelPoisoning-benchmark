import { useMemo } from 'react';
import {
  AGGREGATIONS,
  AGGREGATION_COLORS,
  AGGREGATION_LABELS,
  ATTACKS,
  ATTACK_LABELS,
  PARTITIONS,
  PARTITION_LABELS,
  type Aggregation,
  type Attack,
  type Partition,
} from '../lib/constants';
import {
  filterExperiments,
  useAvailableDimensions,
  useExperiments,
} from '../hooks/useExperiments';
import {
  CheckOption,
  FilterSection,
  FilterSidebar,
  RadioOption,
} from '../components/FilterSidebar';
import { MultiLineChart } from '../components/MultiLineChart';
import { ExportButton } from '../components/ExportButton';
import { buildSeries, METRIC_LABELS, type Metric } from '../lib/series';
import { seriesToCsv } from '../lib/export';
import { useUrlState } from '../lib/url-state';

interface CurvesFilter {
  aggregations: Aggregation[];
  attacks: Attack[];
  partition: Partition;
  malicious: number;
  metric: Metric;
}

const DEFAULT_FILTER: CurvesFilter = {
  aggregations: ['mean', 'krum'],
  attacks: ['lie'],
  partition: 'iid',
  malicious: 4,
  metric: 'accuracy',
};

interface Preset {
  label: string;
  filter: CurvesFilter;
}

const PRESETS: Preset[] = [
  {
    label: 'Mean vs Krum under LIE m=4 (IID)',
    filter: {
      aggregations: ['mean', 'krum'],
      attacks: ['lie'],
      partition: 'iid',
      malicious: 4,
      metric: 'accuracy',
    },
  },
  {
    label: 'All defenses · Min-Max · m=6 (IID)',
    filter: {
      aggregations: [...AGGREGATIONS],
      attacks: ['minmax'],
      partition: 'iid',
      malicious: 6,
      metric: 'accuracy',
    },
  },
  {
    label: 'All defenses · Backdoor ASR · m=4 (IID)',
    filter: {
      aggregations: [...AGGREGATIONS],
      attacks: ['model_replacement'],
      partition: 'iid',
      malicious: 4,
      metric: 'asr',
    },
  },
  {
    label: 'IID vs Non-IID — Median, all attacks, m=4',
    filter: {
      aggregations: ['median'],
      attacks: [...ATTACKS],
      partition: 'iid', // user can flip Non-IID after
      malicious: 4,
      metric: 'accuracy',
    },
  },
];

export function CurvesPage() {
  const { experiments, loading } = useExperiments();
  const dims = useAvailableDimensions();
  const [filter, setFilter] = useUrlState<CurvesFilter>('f', DEFAULT_FILTER);

  const visible = useMemo(() => {
    const matched = filterExperiments(experiments, {
      aggregations: filter.aggregations,
      attacks: filter.attacks,
      partitions: [filter.partition],
      malicious: [filter.malicious],
    });
    return matched;
  }, [experiments, filter]);

  const series = useMemo(() => buildSeries(visible, filter.metric), [visible, filter.metric]);

  const yDomain: [number | 'auto', number | 'auto'] | undefined =
    filter.metric === 'accuracy' || filter.metric === 'asr' ? [0, 100] : undefined;

  const toggleAggregation = (a: Aggregation) =>
    setFilter((prev) => ({
      ...prev,
      aggregations: prev.aggregations.includes(a)
        ? prev.aggregations.filter((x) => x !== a)
        : [...prev.aggregations, a],
    }));

  const toggleAttack = (a: Attack) =>
    setFilter((prev) => ({
      ...prev,
      attacks: prev.attacks.includes(a)
        ? prev.attacks.filter((x) => x !== a)
        : [...prev.attacks, a],
    }));

  return (
    <div className="flex gap-6 items-start">
      <FilterSidebar>
        <FilterSection title="Aggregation">
          {AGGREGATIONS.map((a) => (
            <CheckOption
              key={a}
              value={a}
              label={AGGREGATION_LABELS[a]}
              swatch={AGGREGATION_COLORS[a]}
              checked={filter.aggregations.includes(a)}
              disabled={!dims.aggregations.includes(a)}
              onToggle={toggleAggregation}
            />
          ))}
        </FilterSection>

        <FilterSection title="Attack">
          {ATTACKS.map((a) => (
            <CheckOption
              key={a}
              value={a}
              label={ATTACK_LABELS[a]}
              checked={filter.attacks.includes(a)}
              disabled={!dims.attacks.includes(a)}
              onToggle={toggleAttack}
            />
          ))}
        </FilterSection>

        <FilterSection title="Partition">
          {PARTITIONS.map((p) => (
            <RadioOption
              key={p}
              value={p}
              label={PARTITION_LABELS[p]}
              group="curves-partition"
              current={filter.partition}
              disabled={!dims.partitions.includes(p)}
              onChange={(v) => setFilter((prev) => ({ ...prev, partition: v }))}
            />
          ))}
        </FilterSection>

        <FilterSection title="Malicious clients">
          {dims.malicious.map((m) => (
            <RadioOption
              key={m}
              value={m}
              label={`m = ${m}`}
              group="curves-malicious"
              current={filter.malicious}
              onChange={(v) => setFilter((prev) => ({ ...prev, malicious: v }))}
            />
          ))}
        </FilterSection>

        <FilterSection title="Metric">
          {(['accuracy', 'loss', 'asr', 'evasion'] as Metric[]).map((m) => (
            <RadioOption
              key={m}
              value={m}
              label={METRIC_LABELS[m]}
              group="curves-metric"
              current={filter.metric}
              onChange={(v) => setFilter((prev) => ({ ...prev, metric: v }))}
            />
          ))}
        </FilterSection>
      </FilterSidebar>

      <div className="flex-1 min-w-0 space-y-4">
        <div className="flex items-baseline justify-between">
          <h1 className="text-2xl font-semibold">Training curves</h1>
          <div className="flex items-center gap-3">
            <span className="text-sm text-slate-500">
              {visible.length} curve{visible.length === 1 ? '' : 's'} ·{' '}
              {METRIC_LABELS[filter.metric]}
            </span>
            <ExportButton
              targetSelector='[data-export="curves-chart"]'
              filename={`curves_${filter.metric}_${filter.partition}_m${filter.malicious}`}
              getCsv={() => seriesToCsv(series)}
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              onClick={() => setFilter(p.filter)}
              className="px-3 py-1.5 text-xs rounded-md border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
            >
              {p.label}
            </button>
          ))}
        </div>

        <div
          className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
          data-export="curves-chart"
        >
          {loading ? (
            <div className="h-[380px] flex items-center justify-center text-slate-400">
              Loading…
            </div>
          ) : (
            <MultiLineChart
              series={series}
              yLabel={METRIC_LABELS[filter.metric]}
              yDomain={yDomain}
            />
          )}
        </div>

        {filter.metric === 'asr' &&
          filter.attacks.some(
            (a) => !['model_replacement', 'geotox', 'geotox_adaptive'].includes(a),
          ) && (
            <div className="text-xs text-slate-500 dark:text-slate-400 px-1">
              Note: ASR is only meaningful for backdoor attacks (<strong>Model Replacement</strong>,{' '}
              <strong>GeoTox</strong>, <strong>GeoTox-Adaptive</strong>). For other attacks the curve
              reflects baseline trigger noise (≈ 1–3%) rather than a real backdoor success rate.
            </div>
          )}
        {filter.metric === 'evasion' && (
          <div className="text-xs text-slate-500 dark:text-slate-400 px-1">
            Evasion = % of malicious updates accepted (not filtered) by the defense each round.
            Coordinate-wise defenses (Median, Trimmed-Mean) report 100% by convention.
          </div>
        )}
      </div>
    </div>
  );
}
