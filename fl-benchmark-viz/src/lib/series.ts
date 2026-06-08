import {
  AGGREGATION_COLORS,
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  PARTITION_LABELS,
  type Experiment,
} from './constants';

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
  data: { round: number; value: number }[];
}

export type Metric = 'accuracy' | 'loss' | 'asr' | 'evasion';

export const METRIC_LABELS: Record<Metric, string> = {
  accuracy: 'Accuracy (%)',
  loss: 'Test loss',
  asr: 'ASR (%)',
  evasion: 'Evasion (%)',
};

export function describeConfig(e: Experiment): string {
  return `${AGGREGATION_LABELS[e.aggregation]} · ${ATTACK_LABELS[e.attack]} · ${PARTITION_LABELS[e.partition]} · m${e.malicious}`;
}

/**
 * Build series for Recharts. Each experiment becomes one line.
 * Color is derived from aggregation; suffix differs by partition / malicious to keep distinguishability.
 */
export function buildSeries(experiments: Experiment[], metric: Metric): ChartSeries[] {
  return experiments.map((e) => ({
    key: e.filename,
    label: describeConfig(e),
    color: AGGREGATION_COLORS[e.aggregation],
    data: e.rounds.map((r) => ({ round: r.round, value: r[metric] })),
  }));
}

/**
 * Pivot multiple series into a single array suitable for Recharts <LineChart data={…}>:
 * [{ round: 1, "<key1>": 0.84, "<key2>": 0.79 }, …]
 */
export function pivotForChart(series: ChartSeries[]): Record<string, number>[] {
  const byRound = new Map<number, Record<string, number>>();
  for (const s of series) {
    for (const point of s.data) {
      if (!byRound.has(point.round)) byRound.set(point.round, { round: point.round });
      byRound.get(point.round)![s.key] = point.value;
    }
  }
  return [...byRound.values()].sort((a, b) => a.round - b.round);
}
