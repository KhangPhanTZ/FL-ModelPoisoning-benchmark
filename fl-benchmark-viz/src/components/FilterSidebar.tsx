import type { ReactNode } from 'react';

export function FilterSidebar({ children }: { children: ReactNode }) {
  return (
    <aside className="w-64 shrink-0 space-y-5 rounded-lg border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-4 self-start sticky top-20 max-h-[calc(100vh-6rem)] overflow-auto">
      {children}
    </aside>
  );
}

export function FilterSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400 font-medium mb-2">
        {title}
      </div>
      <div className="flex flex-col gap-1.5">{children}</div>
    </div>
  );
}

interface CheckOptionProps<T extends string | number> {
  value: T;
  label: ReactNode;
  swatch?: string;
  checked: boolean;
  disabled?: boolean;
  onToggle: (v: T) => void;
}

export function CheckOption<T extends string | number>({
  value,
  label,
  swatch,
  checked,
  disabled,
  onToggle,
}: CheckOptionProps<T>) {
  return (
    <label
      className={`flex items-center gap-2 text-sm cursor-pointer select-none ${
        disabled ? 'opacity-40 cursor-not-allowed' : ''
      }`}
    >
      <input
        type="checkbox"
        className="w-4 h-4 accent-sky-600"
        checked={checked}
        disabled={disabled}
        onChange={() => onToggle(value)}
      />
      {swatch && (
        <span
          className="inline-block w-3 h-3 rounded-sm"
          style={{ background: swatch }}
        />
      )}
      <span className="flex-1">{label}</span>
    </label>
  );
}

interface RadioOptionProps<T extends string | number> {
  value: T;
  label: ReactNode;
  group: string;
  current: T;
  disabled?: boolean;
  onChange: (v: T) => void;
}

export function RadioOption<T extends string | number>({
  value,
  label,
  group,
  current,
  disabled,
  onChange,
}: RadioOptionProps<T>) {
  return (
    <label
      className={`flex items-center gap-2 text-sm cursor-pointer select-none ${
        disabled ? 'opacity-40 cursor-not-allowed' : ''
      }`}
    >
      <input
        type="radio"
        name={group}
        className="w-4 h-4 accent-sky-600"
        checked={current === value}
        disabled={disabled}
        onChange={() => onChange(value)}
      />
      {label}
    </label>
  );
}
