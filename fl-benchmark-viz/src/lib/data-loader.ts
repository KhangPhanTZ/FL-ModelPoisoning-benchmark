import Papa from 'papaparse';
import { parseConfigName } from './config-parser';
import type { Experiment, RoundData } from './constants';
import { computeMetrics } from './metrics';

export interface Manifest {
  generated_at: string;
  files: string[];
}

export interface LoadProgress {
  loaded: number;
  total: number;
  current: string | null;
}

export interface LoadResult {
  experiments: Experiment[];
  warnings: string[];
  manifest: Manifest | null;
}

const RESULTS_BASE = `${import.meta.env.BASE_URL}results/`;

async function fetchManifest(): Promise<Manifest> {
  const res = await fetch(`${RESULTS_BASE}manifest.json`, { cache: 'no-cache' });
  if (!res.ok) {
    throw new Error(
      `Failed to fetch manifest.json (HTTP ${res.status}). Did you run scripts/generate-manifest.py?`,
    );
  }
  const json = (await res.json()) as Manifest;
  if (!json || !Array.isArray(json.files)) {
    throw new Error('manifest.json is malformed: expected { files: string[] }');
  }
  return json;
}

async function loadOneExperiment(filename: string): Promise<RoundData[]> {
  const res = await fetch(`${RESULTS_BASE}${filename}`, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const text = await res.text();
  return new Promise<RoundData[]>((resolve, reject) => {
    Papa.parse<RoundData>(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        // Filter out any rows that don't have a numeric round (defensive)
        const cleaned = results.data.filter(
          (r): r is RoundData =>
            r != null &&
            typeof r.round === 'number' &&
            typeof r.accuracy === 'number' &&
            typeof r.loss === 'number',
        );
        if (cleaned.length === 0) {
          reject(new Error('no valid rows'));
          return;
        }
        // Ensure asr always present (default 0 if missing)
        for (const row of cleaned) {
          if (typeof row.asr !== 'number') row.asr = 0;
          if (typeof row.timestamp !== 'string') row.timestamp = '';
        }
        cleaned.sort((a, b) => a.round - b.round);
        resolve(cleaned);
      },
      error: (err: Error) => reject(err),
    });
  });
}

export async function loadAllExperiments(
  onProgress?: (p: LoadProgress) => void,
): Promise<LoadResult> {
  const warnings: string[] = [];
  let manifest: Manifest;

  try {
    manifest = await fetchManifest();
  } catch (err) {
    warnings.push(err instanceof Error ? err.message : String(err));
    return { experiments: [], warnings, manifest: null };
  }

  const total = manifest.files.length;
  const experiments: Experiment[] = [];
  let loaded = 0;

  // Load in small parallel batches to keep network busy without thrashing.
  const BATCH = 8;
  for (let i = 0; i < manifest.files.length; i += BATCH) {
    const batch = manifest.files.slice(i, i + BATCH);
    const results = await Promise.all(
      batch.map(async (filename) => {
        const config = parseConfigName(filename);
        if (!config) {
          warnings.push(`Skip ${filename}: cannot parse config from name`);
          return null;
        }
        try {
          const rounds = await loadOneExperiment(filename);
          const metrics = computeMetrics(rounds, config.attack);
          return { filename, ...config, rounds, metrics } satisfies Experiment;
        } catch (err) {
          warnings.push(
            `Skip ${filename}: ${err instanceof Error ? err.message : String(err)}`,
          );
          return null;
        }
      }),
    );
    for (let k = 0; k < batch.length; k++) {
      loaded++;
      const exp = results[k];
      if (exp) experiments.push(exp);
      onProgress?.({ loaded, total, current: batch[k] });
    }
  }

  return { experiments, warnings, manifest };
}
