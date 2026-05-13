/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx,js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        agg: {
          mean: '#4E79A7',
          median: '#F28E2B',
          krum: '#59A14F',
          multi_krum: '#76B7B2',
          bulyan: '#B07AA1',
          fltrust: '#E15759',
        },
      },
      fontFamily: {
        sans: ['system-ui', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['ui-monospace', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
