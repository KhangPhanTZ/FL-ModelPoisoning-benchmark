import { createContext } from 'react';
import type { Experiment } from '../lib/constants';
import type { LoadProgress } from '../lib/data-loader';

export interface DataContextValue {
  experiments: Experiment[];
  byKey: Map<string, Experiment>;
  loading: boolean;
  progress: LoadProgress;
  warnings: string[];
  error: string | null;
  reload: () => void;
}

export const DataContext = createContext<DataContextValue | null>(null);
