import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
      colors: {
        // Pure noir surface scale (no blue tint)
        's0': '#000000',
        's1': '#050505',
        's2': '#0a0a0a',
        's3': '#0f0f12',
        's4': '#121215',
        // Text
        'text-primary': '#ffffff',
        'text-secondary': '#a1a1aa',
        'text-muted': '#52525b',
        // Accent — Cinema Amber
        'accent': '#D97706',
        'accent-dim': '#92400E',
        'accent-bright': '#F59E0B',
      },
      animation: {
        shimmer: 'shimmer 1.6s infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      keyframes: {
        shimmer: {
          '0%':   { backgroundPosition: '-400px 0' },
          '100%': { backgroundPosition:  '400px 0' },
        },
      },
      letterSpacing: {
        'tighter-apple': '-0.03em',
        'tight-apple':   '-0.02em',
        'body-apple':    '-0.015em',
        'label':         '0.09em',
        'label-wide':    '0.14em',
      },
      lineHeight: {
        'display':  '1.00',
        'headline': '1.08',
        'card':     '1.19',
      },
      borderRadius: {
        'pill':        '9999px',
        'card':        '4px',
        'panel':       '10px',
        'apple-btn':   '6px',
        'apple-panel': '10px',
      },
      boxShadow: {
        'card':       '0 8px 24px rgba(0,0,0,0.7)',
        'hover':      '0 20px 60px rgba(0,0,0,0.85), 0 0 1px rgba(255,255,255,0.06)',
        'modal':      '0 32px 80px rgba(0,0,0,0.9)',
        'apple-card': 'rgba(0,0,0,0.4) 3px 5px 30px 0px',
      },
    },
  },
  plugins: [],
} satisfies Config
