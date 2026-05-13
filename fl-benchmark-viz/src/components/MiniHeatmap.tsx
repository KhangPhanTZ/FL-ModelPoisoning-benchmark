import {
  AGGREGATION_LABELS,
  ATTACK_LABELS,
  colorForAccuracy,
  type Aggregation,
  type Attack,
  type Experiment,
} from '../lib/constants';
import { meanFinalAccuracy } from '../lib/metrics';

interface Props {
  experiments: Experiment[];
  aggregations: Aggregation[];
  attacks: Attack[];
  onCellClick?: (cell: { aggregation: Aggregation; attack: Attack }) => void;
}

export function MiniHeatmap({ experiments, aggregations, attacks, onCellClick }: Props) {
  const grid = aggregations.map((agg) =>
    attacks.map((atk) => {
      const cellExps = experiments.filter(
        (e) => e.aggregation === agg && e.attack === atk,
      );
      const value = meanFinalAccuracy(cellExps);
      return { agg, atk, value, count: cellExps.length };
    }),
  );

  return (
    <div className="overflow-auto">
      <table className="text-xs border-separate border-spacing-1">
        <thead>
          <tr>
            <th />
            {attacks.map((atk) => (
              <th
                key={atk}
                className="px-2 py-1 text-slate-600 dark:text-slate-400 font-medium"
              >
                {ATTACK_LABELS[atk]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {grid.map((row, i) => (
            <tr key={aggregations[i]}>
              <td className="px-2 py-1 text-right font-medium text-slate-600 dark:text-slate-400 whitespace-nowrap">
                {AGGREGATION_LABELS[aggregations[i]]}
              </td>
              {row.map((cell) => {
                const empty = cell.count === 0 || Number.isNaN(cell.value);
                const bg = empty ? 'transparent' : colorForAccuracy(cell.value);
                return (
                  <td
                    key={cell.atk}
                    onClick={
                      empty || !onCellClick
                        ? undefined
                        : () => onCellClick({ aggregation: cell.agg, attack: cell.atk })
                    }
                    className={`min-w-[70px] h-12 text-center rounded text-slate-900 font-medium border border-slate-300 dark:border-slate-700 ${
                      empty ? 'opacity-40' : 'cursor-pointer hover:ring-2 hover:ring-slate-900 dark:hover:ring-slate-100'
                    }`}
                    style={{ background: bg }}
                    title={
                      empty
                        ? 'no data'
                        : `${AGGREGATION_LABELS[cell.agg]} × ${ATTACK_LABELS[cell.atk]}\nAvg final acc: ${cell.value.toFixed(2)}%\n(${cell.count} runs)`
                    }
                  >
                    {empty ? '—' : `${cell.value.toFixed(1)}`}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
