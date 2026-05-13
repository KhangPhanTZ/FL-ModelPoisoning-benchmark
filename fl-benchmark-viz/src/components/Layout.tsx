import { NavLink, Outlet } from 'react-router-dom';
import {
  BarChart3,
  Grid3x3,
  LineChart,
  Moon,
  Search,
  ShieldAlert,
  Sun,
} from 'lucide-react';
import { useDarkMode } from '../hooks/useDarkMode';
import { useExperiments } from '../hooks/useExperiments';

const NAV = [
  { to: '/', label: 'Overview', icon: BarChart3, end: true },
  { to: '/curves', label: 'Training Curves', icon: LineChart, end: false },
  { to: '/matrix', label: 'Defense × Attack', icon: Grid3x3, end: false },
  { to: '/attacks', label: 'Attacks', icon: ShieldAlert, end: false },
  { to: '/explorer', label: 'Explorer', icon: Search, end: false },
];

export function Layout() {
  const [isDark, toggleDark] = useDarkMode();
  const { loading, progress, experiments, error, warnings } = useExperiments();

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950">
      <header className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 sticky top-0 z-20">
        <div className="max-w-[1400px] mx-auto px-6 h-14 flex items-center gap-6">
          <div className="font-semibold text-lg">
            FL Benchmark <span className="text-slate-400 font-normal">— Model Poisoning</span>
          </div>
          <nav className="flex items-center gap-1 ml-2">
            {NAV.map(({ to, label, icon: Icon, end }) => (
              <NavLink
                key={to}
                to={to}
                end={end}
                className={({ isActive }) =>
                  [
                    'inline-flex items-center gap-2 px-3 py-1.5 rounded-md text-sm transition-colors',
                    isActive
                      ? 'bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900'
                      : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800',
                  ].join(' ')
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400">
            {loading ? (
              <span>
                Loading {progress.loaded}/{progress.total}…
              </span>
            ) : error ? (
              <span className="text-rose-500">{error}</span>
            ) : (
              <span>
                {experiments.length} configs
                {warnings.length > 0 && (
                  <span className="ml-2 text-amber-500">({warnings.length} warnings)</span>
                )}
              </span>
            )}
            <button
              type="button"
              onClick={toggleDark}
              className="p-1.5 rounded-md hover:bg-slate-100 dark:hover:bg-slate-800"
              aria-label="Toggle dark mode"
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>
      <main className="flex-1 max-w-[1400px] mx-auto w-full px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
