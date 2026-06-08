import {
  AGGREGATIONS,
  ATTACKS,
  FILE_REGEX,
  LEGACY_FILE_REGEX,
  PARTITIONS,
  type Aggregation,
  type Attack,
  type ConfigKey,
  type Dataset,
  type Partition,
} from './constants';

export class ConfigParseError extends Error {
  readonly filename: string;
  constructor(filename: string, reason: string) {
    super(`Cannot parse "${filename}": ${reason}`);
    this.name = 'ConfigParseError';
    this.filename = filename;
  }
}

function num(v: string | undefined): number | null {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function valid(agg: string, atk: string, part: string): boolean {
  return (
    AGGREGATIONS.includes(agg as Aggregation) &&
    ATTACKS.includes(atk as Attack) &&
    PARTITIONS.includes(part as Partition)
  );
}

/**
 * Parse a benchmark CSV filename into its config dimensions.
 * Supports the current scheme
 *   `mnist_flame_geotox_noniid_a0.5_m4_u15_t0.9_s42.csv`
 * and the legacy scheme
 *   `mean_lie_iid_m4.csv`.
 * Returns null on failure — callers may skip.
 */
export function parseConfigName(filename: string): ConfigKey | null {
  const m = filename.match(FILE_REGEX);
  if (m) {
    const [, dataset, agg, atk, part, alpha, mal, until, tau, seed] = m;
    if (!valid(agg, atk, part)) return null;
    const malicious = parseInt(mal, 10);
    if (!Number.isFinite(malicious)) return null;
    return {
      dataset: dataset as Dataset,
      aggregation: agg as Aggregation,
      attack: atk as Attack,
      partition: part as Partition,
      alpha: num(alpha),
      malicious,
      attackUntil: num(until) ?? 0,
      tau: num(tau),
      seed: num(seed),
    };
  }

  const lm = filename.match(LEGACY_FILE_REGEX);
  if (lm) {
    const [, agg, atk, part, mal] = lm;
    if (!valid(agg, atk, part)) return null;
    const malicious = parseInt(mal, 10);
    if (!Number.isFinite(malicious)) return null;
    return {
      dataset: 'mnist',
      aggregation: agg as Aggregation,
      attack: atk as Attack,
      partition: part as Partition,
      alpha: null,
      malicious,
      attackUntil: 0,
      tau: null,
      seed: null,
    };
  }

  return null;
}

export function parseConfigNameOrThrow(filename: string): ConfigKey {
  const result = parseConfigName(filename);
  if (!result) {
    throw new ConfigParseError(filename, 'does not match expected pattern');
  }
  return result;
}

/** Stable canonical key for indexing/dedup (includes all dimensions). */
export function configToKey(c: ConfigKey): string {
  const a = c.alpha == null ? '' : `_a${c.alpha}`;
  const u = c.attackUntil ? `_u${c.attackUntil}` : '';
  const t = c.tau == null ? '' : `_t${c.tau}`;
  const s = c.seed == null ? '' : `_s${c.seed}`;
  return `${c.dataset}_${c.aggregation}_${c.attack}_${c.partition}${a}_m${c.malicious}${u}${t}${s}`;
}

export function configToFilename(c: ConfigKey): string {
  return `${configToKey(c)}.csv`;
}

// -----------------------------------------------------------------------------
// Smoke test
// -----------------------------------------------------------------------------

const SAMPLES: Array<[string, Partial<ConfigKey> | null]> = [
  [
    'mnist_krum_geotox_noniid_a0.5_m4_t0.9_s42.csv',
    {
      dataset: 'mnist',
      aggregation: 'krum',
      attack: 'geotox',
      partition: 'noniid',
      alpha: 0.5,
      malicious: 4,
      tau: 0.9,
      seed: 42,
      attackUntil: 0,
    },
  ],
  [
    'mnist_flame_geotox_adaptive_noniid_a0.5_m4_u15_t0.0_s42.csv',
    {
      dataset: 'mnist',
      aggregation: 'flame',
      attack: 'geotox_adaptive',
      partition: 'noniid',
      alpha: 0.5,
      malicious: 4,
      attackUntil: 15,
      tau: 0.0,
      seed: 42,
    },
  ],
  [
    'fashion_mnist_fltrust_none_iid_m0_s7.csv',
    {
      dataset: 'fashion_mnist',
      aggregation: 'fltrust',
      attack: 'none',
      partition: 'iid',
      malicious: 0,
      seed: 7,
    },
  ],
  // Legacy
  ['mean_lie_iid_m2.csv', { dataset: 'mnist', aggregation: 'mean', attack: 'lie', partition: 'iid', malicious: 2 }],
  // Negative cases
  ['experiment_summary.csv', null],
  ['summary_by_config.csv', null],
  ['mnist_mean_lie_iid_m.csv', null],
];

export function runParserSmokeTest(log: (msg: string) => void = console.log): boolean {
  let allPass = true;
  for (const [name, expected] of SAMPLES) {
    const got = parseConfigName(name);
    const pass =
      expected == null
        ? got == null
        : got != null && Object.entries(expected).every(([k, v]) => (got as unknown as Record<string, unknown>)[k] === v);
    if (!pass) allPass = false;
    log(`${pass ? '✓' : '✗'} ${name} → ${JSON.stringify(got)}`);
  }
  log(allPass ? 'All parser tests passed.' : 'Parser tests FAILED.');
  return allPass;
}
