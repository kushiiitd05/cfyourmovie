export type MovieSignal = 'Top Pick' | 'Trending' | 'Hidden Gem' | null

export interface Movie {
  title: string
  year?: string | number
  genres?: string | string[]
  vote_average?: number
  rating?: number
  fused_score?: number
  mix_score?: number
  faiss_score_norm?: number
  score?: number
  _score?: number
  poster_url?: string
  signal?: MovieSignal
  [key: string]: unknown
}

export interface RecommendResult {
  pipeline: string
  query: string
  user_id?: number
  movies: Movie[]
  explanation: string
  hypothetical?: string
  sub_queries?: string[]
  parsed?: Record<string, unknown>
}

export interface PipelineMeta {
  name: string
  badge: string
  description: string
  color: PipelineColor
  requires_user: boolean
}

export type PipelineColor = 'violet' | 'blue' | 'emerald' | 'amber' | 'rose' | 'slate'
export type PipelineKey = 'P5' | 'P4' | 'P2' | 'P1' | 'P3' | 'auto'

export type AppState = 'idle' | 'loading' | 'results' | 'error'
