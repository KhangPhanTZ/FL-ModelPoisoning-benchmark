import { useEffect, useMemo } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  Legend,
} from 'recharts';
import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  PARTITION_LABELS,
  type Aggregation,
  type Attack,
  type Experiment,
  type Partition,
} from '../lib/constants';
import { useAvailableDimensions, useExperiments } from '../hooks/useExperiments';
import { configToFilename } from '../lib/config-parser';
import { useUrlState } from '../lib/url-state';
import { StatCard } from '../components/StatCard';
import { ExportButton } from '../components/ExportButton';

interface ExplorerPick {
  aggregation: Aggregation | '';
  attack: Attack | '';
  partition: Partition | '';
  malicious: number | '';
  showAcc: boolean;
  showLoss: boolean;
  showAsr: boolean;
}

const DEFAULT_PICK: ExplorerPick = {
  aggregation: 'mean',
  attack: 'lie',
  partition: 'iid',
  malicious: 4,
  showAcc: true,
  showLoss: true,
  showAsr: false,
};

function coreKeyOf(a: string, att: string, p: string, m: number): string {
  return `${a}__${att}__${p}__m${m}`;
}

export function ExplorerPage() {
  const { experiments, loading } = useExperiments();
  const dims = useAvailableDimensions();
  const [pick, setPick] = useUrlState<ExplorerPick>('e', DEFAULT_PICK);

  // Map a (defense, attack, partition, malicious) tuple to the first matching
  // experiment (the new data may have several seeds/taus per tuple).
  const coreMap = useMemo(() => {
    const map = new Map<string, Experiment>();
    for (const e of experiments) {
      const k = coreKeyOf(e.aggregation, e.attack, e.partition, e.malicious);
      if (!map.has(k)) map.set(k, e);
    }
    return map;
  }, [experiments]);

  // Auto-pick first available experiment if current pick has no data
  useEffect(() => {
    if (loading || experiments.length === 0) return;
    const candidate =
      pick.aggregation && pick.attack && pick.partition && pick.malicious !== ''
        ? coreMap.get(
            coreKeyOf(pick.aggregation, pick.attack, pick.partition, Number(pick.malicious)),
          )
        : undefined;
    if (!candidate) {
      const first = experiments[0];
      setPick({
        aggregation: first.aggregation,
        attack: first.attack,
        partition: first.partition,
        malicious: first.malicious,
        showAcc: pick.showAcc,
        showLoss: pick.showLoss,
        showAsr: pick.showAsr || first.metrics.asrIsMeaningful,
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [loading, experiments.length]);

  const exp = useMemo<Experiment | null>(() => {
    if (!pick.aggregation || !pick.attack || !pick.partition || pick.malicious === '') {
      return null;
    }
    return (
      coreMap.get(
        coreKeyOf(pick.aggregation, pick.attack, pick.partition, Number(pick.malicious)),
      ) ?? null
    );
  }, [coreMap, pick.aggregation, pick.attack, pick.partition, pick.malicious]);

  // Determine which (att, part, mal) options are available given current aggregation, etc.
  const validKeys = useMemo(() => new Set([...coreMap.keys()]), [coreMap]);
  const isOptionEnabled = (override: Partial<ExplorerPick>) => {
    const merged = { ...pick, ...override };
    if (!merged.aggregation || !merged.attack || !merged.partition || merged.malicious === '') {
      return false;
    }
    return validKeys.has(
      coreKeyOf(merged.aggregation, merged.attack, merged.partition, Number(merged.malicious)),
    );
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Config explorer</h1>

      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Picker
          label="Aggregation"
          options={dims.aggregations.map((a) => ({
            value: a,
            label: AGGREGATION_LABELS[a],
            enabled: isOptionEnabled({ aggregation: a }),
          }))}
          value={pick.aggregation}
          onChange={(v) => setPick((p) => ({ ...p, aggregation: v as Aggregation }))}
        />
        <Picker
          label="Attack"
          options={dims.attacks.map((a) => ({
            value: a,
            label: ATTACK_LABELS[a],
            enabled: isOptionEnabled({ attack: a }),
          }))}
          value={pick.attack}
          onChange={(v) => setPick((p) => ({ ...p, attack: v as Attack }))}
        />
        <Picker
          label="Partition"
          options={dims.partitions.map((p) => ({
            value: p,
            label: PARTITION_LABELS[p],
            enabled: isOptionEnabled({ partition: p }),
          }))}
          value={pick.partition}
          onChange={(v) => setPick((p) => ({ ...p, partition: v as Partition }))}
        />
        <Picker
          label="Malicious"
          options={dims.malicious.map((m) => ({
            value: String(m),
            label: `m = ${m}`,
            enabled: isOptionEnabled({ malicious: m }),
          }))}
          value={String(pick.malicious)}
          onChange={(v) => setPick((p) => ({ ...p, malicious: Number(v) }))}
        />
      </div>

      {!exp ? (
        <div className="text-slate-400">No experiment matches the current selection.</div>
      ) : (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            <StatCard label="Final accuracy" value={`${exp.metrics.finalAccuracy.toFixed(2)}%`} accent="green" />
            <StatCard label="Max accuracy" value={`${exp.metrics.maxAccuracy.toFixed(2)}%`} hint={`round ${exp.metrics.maxAccuracyRound}`} />
            <StatCard label="Min loss" value={exp.metrics.minLoss.toFixed(4)} hint={`round ${exp.metrics.minLossRound}`} />
            <StatCard
              label="Convergence round"
              value={exp.metrics.convergenceRound ?? '—'}
              hint="first round ≥ 90% peak acc"
            />
            <StatCard label="Stability σ" value={exp.metrics.stabilityStd.toFixed(2)} hint="last 10 rounds" accent="amber" />
          </div>

          {exp.metrics.asrIsMeaningful && (
            <StatCard
              label="Max ASR (backdoor)"
              value={`${exp.metrics.maxAsr.toFixed(2)}%`}
              hint={`round ${exp.metrics.maxAsrRound}`}
              accent="rose"
            />
          )}

          <div
            data-export="explorer-chart"
            className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4"
          >
            <div className="flex items-center justify-between mb-3">
              <h2 className="font-medium">{exp.filename}</h2>
              <div className="flex flex-wrap items-center gap-3 text-xs">
                <Toggle
                  label="Acc %"
                  on={pick.showAcc}
                  onChange={(v) => setPick((p) => ({ ...p, showAcc: v }))}
                />
                <Toggle
                  label="Loss"
                  on={pick.showLoss}
                  onChange={(v) => setPick((p) => ({ ...p, showLoss: v }))}
                />
                <Toggle
                  label={`ASR % ${exp.metrics.asrIsMeaningful ? '' : '(noise ref.)'}`}
                  on={pick.showAsr}
                  onChange={(v) => setPick((p) => ({ ...p, showAsr: v }))}
                  muted={!exp.metrics.asrIsMeaningful}
                />
                <ExportButton
                  targetSelector='[data-export="explorer-chart"]'
                  filename={exp.filename.replace(/\.csv$/, '')}
                  getCsv={() => ({
                    columns: ['round', 'loss', 'accuracy', 'asr', 'timestamp'],
                    rows: exp.rounds.map((r) => ({
                      round: r.round,
                      loss: r.loss,
                      accuracy: r.accuracy,
                      asr: r.asr,
                      timestamp: r.timestamp,
                    })),
                  })}
                />
              </div>
            </div>
            <MultiAxisChart
              experiment={exp}
              showAcc={pick.showAcc}
              showLoss={pick.showLoss}
              showAsr={pick.showAsr}
            />
          </div>

          <RawTable experiment={exp} />
        </>
      )}
    </div>
  );
}

function Picker({
  label,
  options,
  value,
  onChange,
}: {
  label: string;
  options: { value: string; label: string; enabled: boolean }[];
  value: string | number | '';
  onChange: (v: string) => void;
}) {
  return (
    <label className="text-sm">
      <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">
        {label}
      </div>
      <select
        className="w-full rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-800 px-2 py-1.5 text-sm"
        value={String(value)}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} disabled={!o.enabled}>
            {o.label}
            {o.enabled ? '' : ' (no data)'}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle({
  label,
  on,
  onChange,
  muted,
}: {
  label: string;
  on: boolean;
  onChange: (v: boolean) => void;
  muted?: boolean;
}) {
  return (
    <label className={`flex items-center gap-1.5 cursor-pointer ${muted ? 'opacity-60' : ''}`}>
      <input
        type="checkbox"
        className="w-3.5 h-3.5 accent-sky-600"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
      />
      {label}
    </label>
  );
}

function MultiAxisChart({
  experiment,
  showAcc,
  showLoss,
  showAsr,
}: {
  experiment: Experiment;
  showAcc: boolean;
  showLoss: boolean;
  showAsr: boolean;
}) {
  const data = experiment.rounds;
  const asrOpacity = experiment.metrics.asrIsMeaningful ? 1 : 0.5;
  return (
    <div style={{ width: '100%', height: 360 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 32, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
          <XAxis dataKey="round" type="number" domain={['dataMin', 'dataMax']} stroke="currentColor" className="text-slate-500" />
          <YAxis
            yAxisId="pct"
            domain={[0, 100]}
            stroke="currentColor"
            className="text-slate-500"
            label={{ value: '%', angle: -90, position: 'insideLeft', offset: 8 }}
          />
          <YAxis
            yAxisId="loss"
            orientation="right"
            stroke="currentColor"
            className="text-slate-500"
            label={{ value: 'loss', angle: 90, position: 'insideRight', offset: 8 }}
          />
          <Tooltip
            formatter={(v: unknown) =>
              typeof v === 'number' ? v.toFixed(4) : (v as string)
            }
            labelFormatter={(round) => `Round ${round}`}
          />
          <Legend />
          {showAcc && (
            <Line
              yAxisId="pct"
              type="monotone"
              dataKey="accuracy"
              name="Accuracy %"
              stroke="#0ea5e9"
              strokeWidth={2}
              dot={false}
              isAnimationActive
              animationDuration={300}
            />
          )}
          {showLoss && (
            <Line
              yAxisId="loss"
              type="monotone"
              dataKey="loss"
              name="Loss"
              stroke="#f97316"
              strokeWidth={2}
              dot={false}
              isAnimationActive
              animationDuration={300}
            />
          )}
          {showAsr && (
            <Line
              yAxisId="pct"
              type="monotone"
              dataKey="asr"
              name={`ASR %${experiment.metrics.asrIsMeaningful ? '' : ' (noise)'}`}
              stroke="#a855f7"
              strokeWidth={2}
              strokeOpacity={asrOpacity}
              strokeDasharray={experiment.metrics.asrIsMeaningful ? undefined : '4 4'}
              dot={false}
              isAnimationActive
              animationDuration={300}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function RawTable({ experiment }: { experiment: Experiment }) {
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
      <div className="flex items-center justify-between px-4 py-2 border-b border-slate-200 dark:border-slate-700">
        <h3 className="font-medium text-sm">Raw data</h3>
        <span className="text-xs text-slate-500">{experiment.rounds.length} rows</span>
      </div>
      <div className="max-h-[360px] overflow-auto">
        <table className="w-full text-xs font-mono">
          <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800 text-slate-500">
            <tr>
              <th className="text-right px-3 py-1.5 font-medium">round</th>
              <th className="text-right px-3 py-1.5 font-medium">loss</th>
              <th className="text-right px-3 py-1.5 font-medium">accuracy</th>
              <th className="text-right px-3 py-1.5 font-medium">asr</th>
              <th className="text-left px-3 py-1.5 font-medium">timestamp</th>
            </tr>
          </thead>
          <tbody>
            {experiment.rounds.map((r) => (
              <tr key={r.round} className="border-t border-slate-100 dark:border-slate-800">
                <td className="text-right px-3 py-1">{r.round}</td>
                <td className="text-right px-3 py-1">{r.loss.toFixed(6)}</td>
                <td className="text-right px-3 py-1">{r.accuracy.toFixed(4)}</td>
                <td className="text-right px-3 py-1">{r.asr.toFixed(4)}</td>
                <td className="text-left px-3 py-1 text-slate-500">{r.timestamp}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-2 border-t border-slate-200 dark:border-slate-700 text-xs text-slate-500">
        File: <code className="font-mono">{configToFilename(experiment)}</code>
      </div>
    </div>
  );
}
