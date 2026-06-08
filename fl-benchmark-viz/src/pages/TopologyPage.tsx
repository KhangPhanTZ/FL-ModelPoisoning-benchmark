import { useMemo, useState } from 'react';
import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  BACKDOOR_ATTACKS,
  type Aggregation,
  type Attack,
  type Experiment,
  type Partition,
} from '../lib/constants';
import { useAvailableDimensions, useExperiments } from '../hooks/useExperiments';
import { StatCard } from '../components/StatCard';

// Benchmark defaults (see run_quickstart.py / main.py).
const NUM_CLIENTS = 20;
const CLIENTS_PER_ROUND = 10;

const SIZE = 560;
const CENTER = SIZE / 2;
const RING_R = 230;
const SERVER_R = 46;
const CLIENT_R = 16;

function coreKeyOf(a: string, att: string, p: string, m: number): string {
  return `${a}__${att}__${p}__m${m}`;
}

interface ClientNode {
  i: number;
  x: number;
  y: number;
  malicious: boolean;
  accepted: boolean;
}

export function TopologyPage() {
  const { experiments, loading } = useExperiments();
  const dims = useAvailableDimensions();

  const coreMap = useMemo(() => {
    const map = new Map<string, Experiment>();
    for (const e of experiments) {
      const k = coreKeyOf(e.aggregation, e.attack, e.partition, e.malicious);
      if (!map.has(k)) map.set(k, e);
    }
    return map;
  }, [experiments]);

  // Default: an attacked config with malicious clients, if available.
  const defaultExp = useMemo(
    () =>
      experiments.find((e) => e.attack !== 'none' && e.malicious > 0) ??
      experiments[0] ??
      null,
    [experiments],
  );

  const [pick, setPick] = useState<{
    aggregation: Aggregation;
    attack: Attack;
    partition: Partition;
    malicious: number;
  } | null>(null);

  const active =
    pick ??
    (defaultExp
      ? {
          aggregation: defaultExp.aggregation,
          attack: defaultExp.attack,
          partition: defaultExp.partition,
          malicious: defaultExp.malicious,
        }
      : null);

  const exp = active
    ? coreMap.get(coreKeyOf(active.aggregation, active.attack, active.partition, active.malicious)) ??
      null
    : null;

  if (loading) return <div className="text-slate-500 dark:text-slate-400">Loading…</div>;
  if (!active || experiments.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h1 className="text-2xl font-semibold mb-2">Global Topology &amp; Architecture</h1>
        <p className="text-sm text-slate-500">No experiments loaded.</p>
      </div>
    );
  }

  const malicious = active.malicious;
  const evasion = exp?.metrics.finalEvasion ?? 0;
  const asr = exp?.metrics.finalAsr ?? 0;
  const acc = exp?.metrics.finalAccuracy ?? 0;
  // Coarse visual: malicious updates "get through" the defense when the measured
  // evasion rate is high; otherwise the defense filters them.
  const maliciousAccepted = evasion >= 50;
  const isBackdoor = BACKDOOR_ATTACKS.includes(active.attack);

  const nodes: ClientNode[] = Array.from({ length: NUM_CLIENTS }, (_, i) => {
    const angle = (i / NUM_CLIENTS) * 2 * Math.PI - Math.PI / 2;
    return {
      i,
      x: CENTER + RING_R * Math.cos(angle),
      y: CENTER + RING_R * Math.sin(angle),
      malicious: i < malicious,
      accepted: i < malicious ? maliciousAccepted : true,
    };
  });

  const update = (patch: Partial<typeof active>) => setPick({ ...active, ...patch });

  return (
    <div className="space-y-5">
      <div className="flex items-baseline justify-between">
        <h1 className="text-2xl font-semibold">Global Topology &amp; Architecture</h1>
        <span className="text-sm text-slate-500">
          {NUM_CLIENTS} clients · {CLIENTS_PER_ROUND}/round · {malicious} malicious (
          {((malicious / NUM_CLIENTS) * 100).toFixed(0)}%)
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap gap-3">
        <Select
          label="Defense"
          value={active.aggregation}
          options={dims.aggregations}
          labels={AGGREGATION_LABELS}
          onChange={(v) => update({ aggregation: v as Aggregation })}
        />
        <Select
          label="Attack"
          value={active.attack}
          options={dims.attacks}
          labels={ATTACK_LABELS}
          onChange={(v) => update({ attack: v as Attack })}
        />
        <Select
          label="Partition"
          value={active.partition}
          options={dims.partitions}
          labels={{ iid: 'IID', noniid: 'Non-IID' }}
          onChange={(v) => update({ partition: v as Partition })}
        />
        <Select
          label="Malicious"
          value={String(active.malicious)}
          options={dims.malicious.map(String)}
          labels={Object.fromEntries(dims.malicious.map((m) => [String(m), `${m} clients`]))}
          onChange={(v) => update({ malicious: Number(v) })}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Topology diagram */}
        <div className="lg:col-span-2 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
          <svg viewBox={`0 0 ${SIZE} ${SIZE}`} className="w-full h-auto">
            {/* edges */}
            {nodes.map((n) => {
              const color = n.malicious
                ? n.accepted
                  ? '#ef4444'
                  : '#ef444455'
                : '#10b981';
              return (
                <g key={`e${n.i}`}>
                  <line
                    x1={n.x}
                    y1={n.y}
                    x2={CENTER}
                    y2={CENTER}
                    stroke={color}
                    strokeWidth={n.malicious ? 2.5 : 1.5}
                    strokeDasharray={n.malicious && !n.accepted ? '4 4' : '6 8'}
                  >
                    <animate
                      attributeName="stroke-dashoffset"
                      from="28"
                      to="0"
                      dur={n.malicious ? '0.9s' : '1.6s'}
                      repeatCount="indefinite"
                    />
                  </line>
                  {/* blocked marker for filtered malicious */}
                  {n.malicious && !n.accepted && (
                    <text
                      x={(n.x + CENTER) / 2}
                      y={(n.y + CENTER) / 2}
                      fontSize="16"
                      fill="#ef4444"
                      textAnchor="middle"
                      dominantBaseline="central"
                    >
                      ✕
                    </text>
                  )}
                </g>
              );
            })}

            {/* defense ring */}
            {active.aggregation !== 'mean' && (
              <circle
                cx={CENTER}
                cy={CENTER}
                r={SERVER_R + 14}
                fill="none"
                stroke="#0ea5e9"
                strokeWidth={2}
                strokeDasharray="3 5"
              />
            )}

            {/* server */}
            <circle cx={CENTER} cy={CENTER} r={SERVER_R} fill="#1e293b" stroke="#0ea5e9" strokeWidth={2} />
            <text x={CENTER} y={CENTER - 6} fontSize="13" fill="#e2e8f0" textAnchor="middle" fontWeight="600">
              Server
            </text>
            <text x={CENTER} y={CENTER + 12} fontSize="11" fill="#7dd3fc" textAnchor="middle">
              {AGGREGATION_LABELS[active.aggregation]}
            </text>

            {/* clients */}
            {nodes.map((n) => (
              <g key={`c${n.i}`}>
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={CLIENT_R}
                  fill={n.malicious ? '#ef4444' : '#10b981'}
                  stroke="#fff"
                  strokeWidth={1.5}
                />
                <text x={n.x} y={n.y} fontSize="11" fill="#fff" textAnchor="middle" dominantBaseline="central">
                  {n.malicious ? '☠' : n.i}
                </text>
              </g>
            ))}
          </svg>

          <div className="flex flex-wrap items-center gap-4 px-2 pb-1 text-xs text-slate-500 dark:text-slate-400">
            <Legend color="#10b981" label="Benign client" />
            <Legend color="#ef4444" label="Malicious client (poisoned update)" />
            <span>
              {active.attack === 'none'
                ? 'No attack'
                : maliciousAccepted
                  ? `Poisoned updates ACCEPTED by ${AGGREGATION_LABELS[active.aggregation]} (evasion ${evasion.toFixed(0)}%)`
                  : `Poisoned updates FILTERED by ${AGGREGATION_LABELS[active.aggregation]} (evasion ${evasion.toFixed(0)}%)`}
            </span>
          </div>
        </div>

        {/* Stats */}
        <div className="space-y-3">
          <StatCard
            label="Attack Success Rate"
            value={isBackdoor ? `${asr.toFixed(1)}%` : 'n/a'}
            hint={isBackdoor ? 'Backdoor trigger → target class' : 'Untargeted attack (see accuracy)'}
            accent="rose"
          />
          <StatCard label="Main accuracy" value={`${acc.toFixed(1)}%`} accent="green" />
          <StatCard
            label="Evasion rate"
            value={active.attack === 'none' ? '—' : `${evasion.toFixed(0)}%`}
            hint="malicious updates accepted by the defense"
            accent="amber"
          />
          <StatCard
            label="Malicious fraction"
            value={`${malicious}/${NUM_CLIENTS}`}
            hint={`${((malicious / NUM_CLIENTS) * 100).toFixed(0)}% of clients`}
            accent="blue"
          />
        </div>
      </div>

      {/* Architecture pipeline */}
      <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
        <h2 className="text-sm font-semibold mb-3 text-slate-600 dark:text-slate-300">
          One federated round — data flow
        </h2>
        <div className="flex flex-wrap items-stretch gap-2 text-xs">
          <Stage title="1. Local training" sub="clients train on private data" />
          <Arrow />
          <Stage title="2. Update Δ" sub="Δ = w_local − w_global" />
          <Arrow />
          <Stage
            title="3. Poisoning"
            sub={active.attack === 'none' ? 'none' : ATTACK_LABELS[active.attack]}
            danger={active.attack !== 'none'}
          />
          <Arrow />
          <Stage
            title="4. Aggregation / Defense"
            sub={AGGREGATION_LABELS[active.aggregation]}
            highlight
          />
          <Arrow />
          <Stage title="5. Global model" sub="w ← w + Δ_agg" />
        </div>
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="inline-block w-3 h-3 rounded-full" style={{ background: color }} />
      {label}
    </span>
  );
}

function Stage({
  title,
  sub,
  highlight,
  danger,
}: {
  title: string;
  sub: string;
  highlight?: boolean;
  danger?: boolean;
}) {
  const border = danger
    ? 'border-rose-400 dark:border-rose-600'
    : highlight
      ? 'border-sky-400 dark:border-sky-600'
      : 'border-slate-200 dark:border-slate-700';
  return (
    <div className={`flex-1 min-w-[120px] rounded-md border ${border} bg-slate-50 dark:bg-slate-800/40 p-2`}>
      <div className="font-semibold text-slate-700 dark:text-slate-200">{title}</div>
      <div className={`mt-0.5 ${danger ? 'text-rose-600 dark:text-rose-400' : 'text-slate-500 dark:text-slate-400'}`}>
        {sub}
      </div>
    </div>
  );
}

function Arrow() {
  return <div className="self-center text-slate-400">→</div>;
}

function Select({
  label,
  value,
  options,
  labels,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  labels: Record<string, string>;
  onChange: (v: string) => void;
}) {
  return (
    <label className="text-xs">
      <span className="block mb-1 text-slate-500 dark:text-slate-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-sm"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {labels[o] ?? o}
          </option>
        ))}
      </select>
    </label>
  );
}
