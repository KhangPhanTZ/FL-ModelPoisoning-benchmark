// =============================================================================
// Domain types
// =============================================================================

export const AGGREGATIONS = [
  'mean',
  'median',
  'krum',
  'multi_krum',
  'bulyan',
  'fltrust',
] as const;

export const ATTACKS = ['none', 'lie', 'minmax', 'model_replacement'] as const;

export const PARTITIONS = ['iid', 'noniid'] as const;

export const MALICIOUS_COUNTS = [2, 4, 6] as const;

export type Aggregation = (typeof AGGREGATIONS)[number];
export type Attack = (typeof ATTACKS)[number];
export type Partition = (typeof PARTITIONS)[number];
export type MaliciousCount = (typeof MALICIOUS_COUNTS)[number];

export interface ConfigKey {
  aggregation: Aggregation;
  attack: Attack;
  partition: Partition;
  malicious: number;
}

export interface RoundData {
  round: number;
  loss: number;
  accuracy: number;
  asr: number;
  timestamp: string;
}

export interface DerivedMetrics {
  finalAccuracy: number;
  finalLoss: number;
  finalAsr: number;
  maxAccuracy: number;
  maxAccuracyRound: number;
  minLoss: number;
  minLossRound: number;
  maxAsr: number;
  maxAsrRound: number;
  convergenceRound: number | null;
  stabilityStd: number;
  asrIsMeaningful: boolean;
}

export interface Experiment extends ConfigKey {
  filename: string;
  rounds: RoundData[];
  metrics: DerivedMetrics;
}

// =============================================================================
// Filename parser regex (alternation order: longer first to avoid partial match)
// =============================================================================

export const FILE_REGEX =
  /^(multi_krum|mean|median|krum|bulyan|fltrust)_(model_replacement|none|lie|minmax)_(iid|noniid)_m(\d+)\.csv$/;

// =============================================================================
// Color palette
// =============================================================================

export const AGGREGATION_COLORS: Record<Aggregation, string> = {
  mean: '#4E79A7',
  median: '#F28E2B',
  krum: '#59A14F',
  multi_krum: '#76B7B2',
  bulyan: '#B07AA1',
  fltrust: '#E15759',
};

export const ATTACK_COLORS: Record<Attack, string> = {
  none: '#94a3b8',
  lie: '#ef4444',
  minmax: '#f59e0b',
  model_replacement: '#a855f7',
};

// Heatmap diverging scale (red → yellow → green) for accuracy %
export interface HeatmapStop {
  threshold: number;
  color: string;
  label: string;
}

export const HEATMAP_STOPS: HeatmapStop[] = [
  { threshold: 30, color: '#d73027', label: '<30' },
  { threshold: 60, color: '#fdae61', label: '30-60' },
  { threshold: 80, color: '#fee090', label: '60-80' },
  { threshold: 95, color: '#a6d96a', label: '80-95' },
  { threshold: Infinity, color: '#1a9850', label: '>95' },
];

export function colorForAccuracy(acc: number): string {
  for (const stop of HEATMAP_STOPS) {
    if (acc < stop.threshold) return stop.color;
  }
  return HEATMAP_STOPS[HEATMAP_STOPS.length - 1].color;
}

// =============================================================================
// Display labels
// =============================================================================

export const AGGREGATION_LABELS: Record<Aggregation, string> = {
  mean: 'FedAvg (Mean)',
  median: 'Coordinate Median',
  krum: 'Krum',
  multi_krum: 'Multi-Krum',
  bulyan: 'Bulyan',
  fltrust: 'FLTrust',
};

export const ATTACK_LABELS: Record<Attack, string> = {
  none: 'No Attack',
  lie: 'LIE',
  minmax: 'Min-Max',
  model_replacement: 'Model Replacement',
};

export const PARTITION_LABELS: Record<Partition, string> = {
  iid: 'IID',
  noniid: 'Non-IID',
};

// =============================================================================
// Attack descriptions (for /attacks page)
// =============================================================================

export interface AttackDescription {
  name: string;
  paper: string;
  description: string;
  formula: string;
  parameter: string;
}

export const ATTACK_DESCRIPTIONS: Record<Attack, AttackDescription> = {
  lie: {
    name: 'LIE (Little Is Enough)',
    paper: 'Baruch et al., 2019',
    description:
      'Malicious clients send carefully crafted updates based on benign statistics, staying within natural variance to evade detection.',
    formula: 'w_mal = μ_benign − z · σ_benign',
    parameter: 'z = 3.0 (attack strength multiplier)',
  },
  minmax: {
    name: 'Min-Max',
    paper: 'Shejwalkar & Houmansadr, 2021',
    description:
      'Maximizes perturbation from benign mean while staying within distance bound of pairwise benign updates.',
    formula: 'w_mal = μ_benign − γ · max_deviation · sign(μ)',
    parameter: 'γ = 15.0 (perturbation scale)',
  },
  model_replacement: {
    name: 'Model Replacement (Backdoor)',
    paper: 'Bagdasaryan et al., 2020',
    description:
      "Scales malicious update so that after FedAvg, global model is replaced by attacker's backdoored local model. Trigger: 4×4 white square in bottom-right corner, target class 7.",
    formula: 'w_mal = global + (N / m_data) · (local − global)',
    parameter: 'Trigger pattern, target class 7',
  },
  none: {
    name: 'No Attack (Baseline)',
    paper: '-',
    description:
      'Standard federated learning without any malicious behavior. Used as baseline for measuring accuracy drop.',
    formula: '-',
    parameter: '-',
  },
};
