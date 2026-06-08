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
  'trimmed_mean',
  'norm_clip',
  'flame',
] as const;

export const ATTACKS = [
  'none',
  'lie',
  'minmax',
  'model_replacement',
  'geotox',
  'geotox_adaptive',
] as const;

export const PARTITIONS = ['iid', 'noniid'] as const;

export const DATASETS = ['mnist', 'fashion_mnist'] as const;

export const MALICIOUS_COUNTS = [0, 2, 4, 6, 8] as const;

export type Aggregation = (typeof AGGREGATIONS)[number];
export type Attack = (typeof ATTACKS)[number];
export type Partition = (typeof PARTITIONS)[number];
export type Dataset = (typeof DATASETS)[number];
export type MaliciousCount = (typeof MALICIOUS_COUNTS)[number];

/** Attacks that carry a backdoor → ASR is a meaningful metric. */
export const BACKDOOR_ATTACKS: Attack[] = [
  'model_replacement',
  'geotox',
  'geotox_adaptive',
];

export interface ConfigKey {
  aggregation: Aggregation;
  attack: Attack;
  partition: Partition;
  malicious: number;
  // New optional dimensions (default-filled by the parser for legacy files).
  dataset: Dataset;
  alpha: number | null; // Dirichlet alpha (non-IID only)
  tau: number | null; // GeoTox stealth knob
  seed: number | null;
  attackUntil: number; // durability: 0 = attacker never leaves
}

export interface RoundData {
  round: number;
  loss: number;
  accuracy: number;
  asr: number;
  evasion: number; // evasion_rate (%) — malicious updates accepted by the defense
  timestamp: string;
}

export interface DerivedMetrics {
  finalAccuracy: number;
  finalLoss: number;
  finalAsr: number;
  finalEvasion: number;
  maxAccuracy: number;
  maxAccuracyRound: number;
  minLoss: number;
  minLossRound: number;
  maxAsr: number;
  maxAsrRound: number;
  maxEvasion: number;
  convergenceRound: number | null;
  stabilityStd: number;
  asrIsMeaningful: boolean;
  evasionIsMeaningful: boolean;
}

export interface Experiment extends ConfigKey {
  filename: string;
  rounds: RoundData[];
  metrics: DerivedMetrics;
}

// =============================================================================
// Filename parsers (vocab-anchored; longer alternatives first)
// =============================================================================

const AGG_ALT = 'multi_krum|trimmed_mean|norm_clip|mean|median|krum|bulyan|fltrust|flame';
const ATK_ALT = 'model_replacement|geotox_adaptive|geotox|none|lie|minmax';

// New scheme:
//   {dataset}_{agg}_{attack}_{partition}[_a{alpha}]_m{mal}[_u{until}][_t{tau}][_s{seed}].csv
export const FILE_REGEX = new RegExp(
  `^(mnist|fashion_mnist)_(${AGG_ALT})_(${ATK_ALT})_(iid|noniid)` +
    `(?:_a([0-9.]+))?_m(\\d+)(?:_u(\\d+))?(?:_t([0-9.]+))?(?:_s(\\d+))?\\.csv$`,
);

// Legacy scheme (pre-update results): {agg}_{attack}_{partition}_m{mal}.csv
export const LEGACY_FILE_REGEX = new RegExp(
  `^(${AGG_ALT})_(${ATK_ALT})_(iid|noniid)_m(\\d+)\\.csv$`,
);

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
  trimmed_mean: '#9C755F',
  norm_clip: '#7C7C7C',
  flame: '#EDC948',
};

export const ATTACK_COLORS: Record<Attack, string> = {
  none: '#94a3b8',
  lie: '#ef4444',
  minmax: '#f59e0b',
  model_replacement: '#a855f7',
  geotox: '#0ea5e9',
  geotox_adaptive: '#6366f1',
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
  trimmed_mean: 'Trimmed Mean',
  norm_clip: 'Norm Clipping',
  flame: 'FLAME',
};

export const ATTACK_LABELS: Record<Attack, string> = {
  none: 'No Attack',
  lie: 'LIE',
  minmax: 'Min-Max',
  model_replacement: 'Model Replacement',
  geotox: 'GeoTox',
  geotox_adaptive: 'GeoTox-Adaptive',
};

export const PARTITION_LABELS: Record<Partition, string> = {
  iid: 'IID',
  noniid: 'Non-IID',
};

export const DATASET_LABELS: Record<Dataset, string> = {
  mnist: 'MNIST',
  fashion_mnist: 'Fashion-MNIST',
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
    formula: 'u_mal = μ_benign − z · σ_benign',
    parameter: 'z = 3.0 (attack strength multiplier)',
  },
  minmax: {
    name: 'Min-Max',
    paper: 'Shejwalkar & Houmansadr, 2021',
    description:
      'Maximizes perturbation from benign mean while staying within distance bound of pairwise benign updates.',
    formula: 'u_mal = μ_benign − γ · max_deviation · sign(μ)',
    parameter: 'γ = 15.0 (perturbation scale)',
  },
  model_replacement: {
    name: 'Model Replacement (Backdoor)',
    paper: 'Bagdasaryan et al., 2020',
    description:
      "Scales the malicious update so that after FedAvg the global model is replaced by the attacker's backdoored model. Trigger: 4×4 white square, target class 7.",
    formula: 'u_mal = (N / m_data) · u_local',
    parameter: 'Trigger pattern, target class 7',
  },
  geotox: {
    name: 'GeoTox (this work)',
    paper: 'Proposed',
    description:
      'Multi-constraint stealthy backdoor: blends the malicious update toward the benign mean direction until cos ≥ τ (directional stealth) and rescales to the median benign norm (magnitude stealth). τ is the Evasion↔ASR trade-off knob.',
    formula: 'u_mal = B · normalize(λ·μ̂ + (1−λ)·d̂),  cos(u_mal, μ̂) ≥ τ',
    parameter: 'τ ∈ [0,1] (stealth), mask_ratio (durability, opt-in)',
  },
  geotox_adaptive: {
    name: 'GeoTox-Adaptive (this work)',
    paper: 'Proposed',
    description:
      'White-box variant: after GeoTox shaping, binary-searches the largest magnitude the known defense still accepts, operating at the acceptance boundary for maximum backdoor strength.',
    formula: 'maximize s · u_mal  s.t.  defense accepts (s ≤ s_max)',
    parameter: 's_max (max scale searched), τ',
  },
  none: {
    name: 'No Attack (Baseline)',
    paper: '-',
    description:
      'Standard federated learning without any malicious behavior. Used as a baseline for measuring accuracy drop.',
    formula: '-',
    parameter: '-',
  },
};
