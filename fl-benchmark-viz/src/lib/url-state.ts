import { useCallback, useEffect, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';

/**
 * Hook for syncing an arbitrary state object to a single URL query param.
 * State is JSON-encoded so we don't pollute the URL key namespace.
 *
 * Consumers should pass a stable `defaultValue` (define it as a module
 * constant rather than re-creating per render) so the parsed value stays
 * referentially stable when no override is present.
 */
export function useUrlState<T>(
  paramName: string,
  defaultValue: T,
): [T, (next: T | ((prev: T) => T)) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get(paramName);

  const value = useMemo<T>(() => {
    if (!raw) return defaultValue;
    try {
      return JSON.parse(decodeURIComponent(raw)) as T;
    } catch {
      return defaultValue;
    }
  }, [raw, defaultValue]);

  // Track latest value so the setter's functional-update form can access it
  // without re-creating the callback every render.
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  }, [value]);

  const setValue = useCallback(
    (next: T | ((prev: T) => T)) => {
      setParams(
        (prev) => {
          const out = new URLSearchParams(prev);
          const resolved =
            typeof next === 'function' ? (next as (p: T) => T)(ref.current) : next;
          out.set(paramName, encodeURIComponent(JSON.stringify(resolved)));
          return out;
        },
        { replace: true },
      );
    },
    [paramName, setParams],
  );

  return [value, setValue];
}
