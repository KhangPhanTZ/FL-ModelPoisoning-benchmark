import type { ReactNode } from 'react';

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  accent?: 'green' | 'rose' | 'blue' | 'amber';
}

const ACCENTS: Record<NonNullable<Props['accent']>, string> = {
  green: 'text-emerald-600 dark:text-emerald-400',
  rose: 'text-rose-600 dark:text-rose-400',
  blue: 'text-sky-600 dark:text-sky-400',
  amber: 'text-amber-600 dark:text-amber-400',
};

export function StatCard({ label, value, hint, accent }: Props) {
  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4">
      <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div
        className={`mt-1 text-2xl font-semibold ${
          accent ? ACCENTS[accent] : 'text-slate-900 dark:text-slate-100'
        }`}
      >
        {value}
      </div>
      {hint && (
        <div className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</div>
      )}
    </div>
  );
}
