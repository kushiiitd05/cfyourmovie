import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

const CINEMATIC_GRADIENTS = [
  ['#1e1b4b', '#312e81'],
  ['#1a1a2e', '#16213e'],
  ['#0f2027', '#203a43'],
  ['#2d1b69', '#11998e'],
  ['#200122', '#6f0000'],
  ['#0a2342', '#2ca2b4'],
  ['#1a0533', '#3d1a78'],
  ['#0d1b2a', '#1b4332'],
]

export function genreGradient(genres: string | string[] | undefined, idx: 0 | 1): string {
  const str = Array.isArray(genres) ? genres.join(' ') : (genres ?? '')
  let hash = 0
  for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash)
  return CINEMATIC_GRADIENTS[Math.abs(hash) % CINEMATIC_GRADIENTS.length][idx]
}

export function genreLabel(genres: string | string[] | undefined): string {
  if (!genres) return ''
  if (Array.isArray(genres)) return genres.slice(0, 2).join(' · ')
  return String(genres).split('|').slice(0, 2).join(' · ')
}

export function scorePercent(score: number): number {
  return Math.round(((Math.min(Math.max(score, -2), 2) + 2) / 4) * 100)
}

export function formatScore(score: number): string {
  return (score >= 0 ? '+' : '') + score.toFixed(3)
}

export function pipelineKey(pipelineStr: string): string {
  const m = pipelineStr.match(/^(P\d)/)
  return m ? m[1] : 'auto'
}
