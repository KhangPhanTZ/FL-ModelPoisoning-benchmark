import { BACKDOOR_ATTACKS, type Attack, type DerivedMetrics, type RoundData } from './constants';

export function computeMetrics(rounds: RoundData[], attack: Attack): DerivedMetrics {
  const asrIsMeaningful = BACKDOOR_ATTACKS.includes(attack);
  const evasionIsMeaningful = attack !== 'none';
  if (rounds.length === 0) {
    return {
      finalAccuracy: 0,
      finalLoss: 0,
      finalAsr: 0,
      finalEvasion: 0,
      maxAccuracy: 0,
      maxAccuracyRound: 0,
      minLoss: 0,
      minLossRound: 0,
      maxAsr: 0,
      maxAsrRound: 0,
      maxEvasion: 0,
      convergenceRound: null,
      stabilityStd: 0,
      asrIsMeaningful,
      evasionIsMeaningful,
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
  let maxEvasion = -Infinity;

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
    if (r.evasion > maxEvasion) maxEvasion = r.evasion;
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
    finalEvasion: last.evasion,
    maxAccuracy,
    maxAccuracyRound,
    minLoss,
    minLossRound,
    maxAsr,
    maxAsrRound,
    maxEvasion: maxEvasion === -Infinity ? 0 : maxEvasion,
    convergenceRound,
    stabilityStd,
    asrIsMeaningful,
    evasionIsMeaningful,
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
