import { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import type { ChartSeries } from '../lib/series';
import { pivotForChart } from '../lib/series';

interface Props {
  series: ChartSeries[];
  yLabel?: string;
  yDomain?: [number | 'auto', number | 'auto'];
  height?: number;
}

export function MultiLineChart({ series, yLabel, yDomain, height = 380 }: Props) {
  const data = useMemo(() => pivotForChart(series), [series]);
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const handleLegendClick = (payload: unknown) => {
    const obj = payload as { dataKey?: unknown; value?: unknown };
    const raw = obj?.dataKey ?? obj?.value;
    const key = typeof raw === 'string' ? raw : null;
    if (!key) return;
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  if (series.length === 0) {
    return (
      <div className="flex items-center justify-center h-[380px] text-slate-400">
        No series selected — adjust filters.
      </div>
    );
  }

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 24, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-slate-200 dark:stroke-slate-700" />
          <XAxis
            dataKey="round"
            type="number"
            domain={['dataMin', 'dataMax']}
            label={{ value: 'Round', position: 'insideBottom', offset: -4 }}
            stroke="currentColor"
            className="text-slate-500"
          />
          <YAxis
            domain={yDomain ?? ['auto', 'auto']}
            label={
              yLabel
                ? { value: yLabel, angle: -90, position: 'insideLeft', offset: 8 }
                : undefined
            }
            stroke="currentColor"
            className="text-slate-500"
            width={64}
          />
          <Tooltip
            formatter={(value) =>
              typeof value === 'number' ? value.toFixed(3) : (value as string)
            }
            labelFormatter={(round) => `Round ${round}`}
          />
          <Legend onClick={handleLegendClick as unknown as never} />
          {series.map((s, i) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.label}
              stroke={s.color}
              strokeWidth={2}
              strokeDasharray={(i % 4) * 4 ? `${(i % 4) * 4} 4` : undefined}
              dot={false}
              activeDot={{ r: 4 }}
              hide={hidden.has(s.key)}
              isAnimationActive
              animationDuration={300}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
