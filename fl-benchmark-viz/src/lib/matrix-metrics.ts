import type { Aggregation, Attack, Experiment } from './constants';

export type MatrixMetric =
  | 'finalAcc'
  | 'maxAcc'
  | 'degradation'
  | 'convergence'
  | 'stability';

export const MATRIX_METRIC_LABELS: Record<MatrixMetric, string> = {
  finalAcc: 'Final accuracy',
  maxAcc: 'Max accuracy',
  degradation: 'Degradation vs no-attack',
  convergence: 'Convergence round',
  stability: 'Stability (lower = better)',
};

export interface CellInfo {
  aggregation: Aggregation;
  attack: Attack;
  value: number;
  count: number;
  experiments: Experiment[];
}
