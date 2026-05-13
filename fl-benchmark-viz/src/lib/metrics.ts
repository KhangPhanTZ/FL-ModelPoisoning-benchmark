import type { Attack, DerivedMetrics, RoundData } from './constants';

export function computeMetrics(rounds: RoundData[], attack: Attack): DerivedMetrics {
  if (rounds.length === 0) {
    return {
      finalAccuracy: 0,
      finalLoss: 0,
      finalAsr: 0,
      maxAccuracy: 0,
      maxAccuracyRound: 0,
      minLoss: 0,
      minLossRound: 0,
      maxAsr: 0,
      maxAsrRound: 0,
      convergenceRound: null,
      stabilityStd: 0,
      asrIsMeaningful: attack === 'model_replacement',
    };
  }

  const sorted = [...rounds].sort((a, b) => a.round - b.round);
  const last = sorted[sorted.length - 1];

  let maxAccuracy = -Infinity;
  let maxAccuracyRound = sorted[0].round;
  let minLoss = Infinity;
  let minLossRound = sorted[0].round;
  let maxAsr = -Infinity;
  let maxAsrRound = sorted[0].round;

  for (const r of sorted) {
    if (r.accuracy > maxAccuracy) {
      maxAccuracy = r.accuracy;
      maxAccuracyRound = r.round;
    }
    if (r.loss < minLoss) {
      minLoss = r.loss;
      minLossRound = r.round;
    }
    if (r.asr > maxAsr) {
      maxAsr = r.asr;
      maxAsrRound = r.round;
    }
  }

  // Convergence: first round where accuracy ≥ 90% of maxAccuracy
  const target = 0.9 * maxAccuracy;
  let convergenceRound: number | null = null;
  for (const r of sorted) {
    if (r.accuracy >= target) {
      convergenceRound = r.round;
      break;
    }
  }

  // Stability: stddev of accuracy over last 10 rounds
  const tail = sorted.slice(-10);
  const stabilityStd = stddev(tail.map((r) => r.accuracy));

  return {
    finalAccuracy: last.accuracy,
    finalLoss: last.loss,
    finalAsr: last.asr,
    maxAccuracy,
    maxAccuracyRound,
    minLoss,
    minLossRound,
    maxAsr,
    maxAsrRound,
    convergenceRound,
    stabilityStd,
    asrIsMeaningful: attack === 'model_replacement',
  };
}

export function stddev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance =
    values.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / values.length;
  return Math.sqrt(variance);
}

/** Mean accuracy across a list of experiments (returns NaN if empty). */
export function meanFinalAccuracy(experiments: { metrics: DerivedMetrics }[]): number {
  if (experiments.length === 0) return NaN;
  const sum = experiments.reduce((acc, e) => acc + e.metrics.finalAccuracy, 0);
  return sum / experiments.length;
}
