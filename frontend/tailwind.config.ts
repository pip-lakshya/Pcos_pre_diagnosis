import type { Config } from 'tailwindcss';
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: { extend: { colors: {
    bg: 'var(--color-bg)',
    accent: 'var(--color-accent)',
    'accent-soft': 'var(--color-accent-soft)',
    ink: 'var(--color-ink)',
  } } },
  plugins: [],
} satisfies Config;
