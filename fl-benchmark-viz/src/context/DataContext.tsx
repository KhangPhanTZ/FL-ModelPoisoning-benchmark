import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { loadAllExperiments, type LoadProgress } from '../lib/data-loader';
import type { Experiment } from '../lib/constants';
import { configToKey } from '../lib/config-parser';
import { DataContext } from './data-context-types';

interface LoadState {
  experiments: Experiment[];
  loading: boolean;
  progress: LoadProgress;
  warnings: string[];
  error: string | null;
}

const INITIAL: LoadState = {
  experiments: [],
  loading: true,
  progress: { loaded: 0, total: 0, current: null },
  warnings: [],
  error: null,
};

export function DataProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LoadState>(INITIAL);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    let cancelled = false;

    loadAllExperiments((p) => {
      if (cancelled) return;
      setState((prev) => ({ ...prev, progress: p }));
    })
      .then((res) => {
        if (cancelled) return;
        setState({
          experiments: res.experiments,
          loading: false,
          progress: {
            loaded: res.experiments.length,
            total: res.experiments.length,
            current: null,
          },
          warnings: res.warnings,
          error: res.manifest
            ? null
            : res.warnings[0] ??
              'No manifest.json found in /results/. Run scripts/generate-manifest.py.',
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setState((prev) => ({
          ...prev,
          loading: false,
          error: err instanceof Error ? err.message : String(err),
        }));
      });

    return () => {
      cancelled = true;
    };
  }, [reloadToken]);

  const reload = useCallback(() => {
    setState(INITIAL);
    setReloadToken((n) => n + 1);
  }, []);

  const byKey = useMemo(() => {
    const map = new Map<string, Experiment>();
    for (const e of state.experiments) map.set(configToKey(e), e);
    return map;
  }, [state.experiments]);

  const value = { ...state, byKey, reload };

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}
