import { useContext, useMemo } from 'react';
import { DataContext, type DataContextValue } from '../context/data-context-types';
import type { Aggregation, Attack, Experiment, Partition } from '../lib/constants';

export function useExperiments(): DataContextValue {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error('useExperiments must be used inside <DataProvider>');
  return ctx;
}

export interface ExperimentFilter {
  aggregations?: Aggregation[];
  attacks?: Attack[];
  partitions?: Partition[];
  malicious?: number[];
}

export function filterExperiments(
  experiments: Experiment[],
  filter: ExperimentFilter,
): Experiment[] {
  return experiments.filter((e) => {
    if (filter.aggregations?.length && !filter.aggregations.includes(e.aggregation)) return false;
    if (filter.attacks?.length && !filter.attacks.includes(e.attack)) return false;
    if (filter.partitions?.length && !filter.partitions.includes(e.partition)) return false;
    if (filter.malicious?.length && !filter.malicious.includes(e.malicious)) return false;
    return true;
  });
}

export function useFilteredExperiments(filter: ExperimentFilter): Experiment[] {
  const { experiments } = useExperiments();
  return useMemo(() => filterExperiments(experiments, filter), [experiments, filter]);
}

/** Distinct values present in the loaded data, for populating dropdowns. */
export function useAvailableDimensions() {
  const { experiments } = useExperiments();
  return useMemo(() => {
    const aggs = new Set<Aggregation>();
    const attacks = new Set<Attack>();
    const parts = new Set<Partition>();
    const mals = new Set<number>();
    for (const e of experiments) {
      aggs.add(e.aggregation);
      attacks.add(e.attack);
      parts.add(e.partition);
      mals.add(e.malicious);
    }
    return {
      aggregations: [...aggs],
      attacks: [...attacks],
      partitions: [...parts],
      malicious: [...mals].sort((a, b) => a - b),
    };
  }, [experiments]);
}
