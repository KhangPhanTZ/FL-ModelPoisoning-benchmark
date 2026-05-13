import { useRef, useState } from 'react';
import { Download } from 'lucide-react';
import {
  exportCsv,
  exportElementAsPng,
  exportElementAsSvg,
  type CsvSeries,
} from '../lib/export';

interface Props {
  /** Selector for the chart-bearing element to capture (e.g. '[data-export="curves-chart"]'). */
  targetSelector: string;
  /** Filename base (no extension). */
  filename: string;
  /** Optional CSV provider — if omitted, the CSV option is hidden. */
  getCsv?: () => CsvSeries;
}

export function ExportButton({ targetSelector, filename, getCsv }: Props) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const wrap = useRef<HTMLDivElement | null>(null);

  const findTarget = (): HTMLElement | null => {
    return document.querySelector<HTMLElement>(targetSelector);
  };

  const handle = async (kind: 'png' | 'svg' | 'csv') => {
    setBusy(kind);
    setOpen(false);
    try {
      if (kind === 'csv') {
        if (!getCsv) return;
        exportCsv(`${filename}.csv`, getCsv());
        return;
      }
      const target = findTarget();
      if (!target) {
        console.warn(`Export target not found: ${targetSelector}`);
        return;
      }
      if (kind === 'png') await exportElementAsPng(target, `${filename}.png`);
      else await exportElementAsSvg(target, `${filename}.svg`);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div ref={wrap} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-md border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-800"
      >
        <Download size={14} />
        {busy ? `Exporting ${busy}…` : 'Export'}
      </button>
      {open && (
        <div
          className="absolute right-0 mt-1 z-10 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-lg overflow-hidden text-sm min-w-[120px]"
          onMouseLeave={() => setOpen(false)}
        >
          <button
            type="button"
            className="block w-full text-left px-3 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"
            onClick={() => handle('png')}
          >
            PNG
          </button>
          <button
            type="button"
            className="block w-full text-left px-3 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"
            onClick={() => handle('svg')}
          >
            SVG
          </button>
          {getCsv && (
            <button
              type="button"
              className="block w-full text-left px-3 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-800"
              onClick={() => handle('csv')}
            >
              CSV
            </button>
          )}
        </div>
      )}
    </div>
  );
}
