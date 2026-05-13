import {
  AGGREGATIONS,
  ATTACKS,
  FILE_REGEX,
  PARTITIONS,
  type Aggregation,
  type Attack,
  type ConfigKey,
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

/**
 * Parse a benchmark CSV filename into its config dimensions.
 * Returns null on failure rather than throwing — callers can choose to skip.
 *
 * Filenames look like: `mean_lie_iid_m4.csv`, `multi_krum_model_replacement_noniid_m6.csv`.
 */
export function parseConfigName(filename: string): ConfigKey | null {
  const match = filename.match(FILE_REGEX);
  if (!match) return null;

  const [, aggregation, attack, partition, malStr] = match;
  const malicious = parseInt(malStr, 10);

  if (
    !AGGREGATIONS.includes(aggregation as Aggregation) ||
    !ATTACKS.includes(attack as Attack) ||
    !PARTITIONS.includes(partition as Partition) ||
    !Number.isFinite(malicious)
  ) {
    return null;
  }

  return {
    aggregation: aggregation as Aggregation,
    attack: attack as Attack,
    partition: partition as Partition,
    malicious,
  };
}

export function parseConfigNameOrThrow(filename: string): ConfigKey {
  const result = parseConfigName(filename);
  if (!result) {
    throw new ConfigParseError(filename, 'does not match expected pattern');
  }
  return result;
}

/** Stable canonical key for indexing/dedup. */
export function configToKey(c: ConfigKey): string {
  return `${c.aggregation}__${c.attack}__${c.partition}__m${c.malicious}`;
}

export function configToFilename(c: ConfigKey): string {
  return `${c.aggregation}_${c.attack}_${c.partition}_m${c.malicious}.csv`;
}

// -----------------------------------------------------------------------------
// Smoke test (run via `npx tsx src/lib/config-parser.ts` or the bundled test)
// -----------------------------------------------------------------------------

const SAMPLES: Array<[string, ConfigKey | null]> = [
  ['mean_lie_iid_m2.csv', { aggregation: 'mean', attack: 'lie', partition: 'iid', malicious: 2 }],
  [
    'krum_model_replacement_noniid_m6.csv',
    { aggregation: 'krum', attack: 'model_replacement', partition: 'noniid', malicious: 6 },
  ],
  [
    'multi_krum_minmax_iid_m4.csv',
    { aggregation: 'multi_krum', attack: 'minmax', partition: 'iid', malicious: 4 },
  ],
  [
    'bulyan_none_noniid_m2.csv',
    { aggregation: 'bulyan', attack: 'none', partition: 'noniid', malicious: 2 },
  ],
  [
    'fltrust_model_replacement_iid_m6.csv',
    { aggregation: 'fltrust', attack: 'model_replacement', partition: 'iid', malicious: 6 },
  ],
  ['median_lie_iid_m2.csv', { aggregation: 'median', attack: 'lie', partition: 'iid', malicious: 2 }],
  // Negative cases
  ['experiment_summary.csv', null],
  ['mean_lie_iid_m.csv', null],
  ['mean_lie_xx_m2.csv', null],
  ['unknown_lie_iid_m2.csv', null],
];

export function runParserSmokeTest(log: (msg: string) => void = console.log): boolean {
  let allPass = true;
  for (const [name, expected] of SAMPLES) {
    const got = parseConfigName(name);
    const pass = JSON.stringify(got) === JSON.stringify(expected);
    if (!pass) allPass = false;
    log(`${pass ? '✓' : '✗'} ${name} → ${JSON.stringify(got)}`);
  }
  log(allPass ? 'All parser tests passed.' : 'Parser tests FAILED.');
  return allPass;
}
