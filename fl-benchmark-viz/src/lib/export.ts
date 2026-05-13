import { toPng, toSvg } from 'html-to-image';

export type ExportFormat = 'png' | 'svg' | 'csv';

function triggerDownload(filename: string, dataUrl: string) {
  const link = document.createElement('a');
  link.download = filename;
  link.href = dataUrl;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function triggerBlobDownload(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  triggerDownload(filename, url);
  // Defer revoke so download has time to start.
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function exportElementAsPng(el: HTMLElement, filename: string): Promise<void> {
  const dataUrl = await toPng(el, {
    cacheBust: true,
    pixelRatio: 2,
    backgroundColor: getComputedBackground(el),
  });
  triggerDownload(filename, dataUrl);
}

export async function exportElementAsSvg(el: HTMLElement, filename: string): Promise<void> {
  const dataUrl = await toSvg(el, {
    cacheBust: true,
    backgroundColor: getComputedBackground(el),
  });
  triggerDownload(filename, dataUrl);
}

function getComputedBackground(el: HTMLElement): string {
  // Walk up until we find a non-transparent background, fall back to body color.
  let node: HTMLElement | null = el;
  while (node) {
    const bg = getComputedStyle(node).backgroundColor;
    if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') return bg;
    node = node.parentElement;
  }
  return getComputedStyle(document.body).backgroundColor || '#ffffff';
}

export interface CsvSeries {
  /** Distinct round numbers across all series will be unioned as the X-axis. */
  rows: Record<string, number | string>[];
  columns: string[];
}

export function exportCsv(filename: string, csv: CsvSeries): void {
  const lines = [csv.columns.join(',')];
  for (const row of csv.rows) {
    lines.push(
      csv.columns
        .map((col) => {
          const v = row[col];
          if (v === undefined || v === null) return '';
          if (typeof v === 'string' && /[",\n]/.test(v)) {
            return `"${v.replace(/"/g, '""')}"`;
          }
          return String(v);
        })
        .join(','),
    );
  }
  triggerBlobDownload(filename, new Blob([lines.join('\n')], { type: 'text/csv' }));
}

/** Convenience for exporting line-chart data: list of {round, <label>: value}. */
export function seriesToCsv(
  series: { label: string; data: { round: number; value: number }[] }[],
): CsvSeries {
  const rowsByRound = new Map<number, Record<string, number | string>>();
  for (const s of series) {
    for (const p of s.data) {
      if (!rowsByRound.has(p.round)) rowsByRound.set(p.round, { round: p.round });
      rowsByRound.get(p.round)![s.label] = p.value;
    }
  }
  const rows = [...rowsByRound.values()].sort(
    (a, b) => (a.round as number) - (b.round as number),
  );
  const columns = ['round', ...series.map((s) => s.label)];
  return { rows, columns };
}
