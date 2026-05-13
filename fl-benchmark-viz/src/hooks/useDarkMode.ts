import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEY = 'fl-viz-theme';

function readInitial(): boolean {
  if (typeof window === 'undefined') return false;
  const saved = localStorage.getItem(STORAGE_KEY);
  if (saved === 'dark') return true;
  if (saved === 'light') return false;
  return false; // light by default per spec
}

export function useDarkMode(): [boolean, () => void] {
  const [isDark, setIsDark] = useState(readInitial);

  useEffect(() => {
    const root = document.documentElement;
    if (isDark) root.classList.add('dark');
    else root.classList.remove('dark');
    localStorage.setItem(STORAGE_KEY, isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggle = useCallback(() => setIsDark((v) => !v), []);
  return [isDark, toggle];
}
