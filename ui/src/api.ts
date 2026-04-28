import type { PipelineMeta, PipelineKey, RecommendResult, Movie, MovieSignal } from './types'

export const PIPELINE_META: Record<PipelineKey, PipelineMeta> = {
  P5: {
    name: 'Hybrid Deep',
    badge: 'BEST',
    description: 'BM25 + FAISS + HyDE + sub-query decomposition + post-fusion reranking',
    color: 'violet',
    requires_user: true,
  },
  P4: {
    name: 'HyDE Hybrid',
    badge: 'COMPLEX',
    description: 'Hypothetical Document Embeddings — ideal for stylistic / comparative queries',
    color: 'blue',
    requires_user: true,
  },
  P2: {
    name: 'Dual Engine',
    badge: 'BALANCED',
    description: 'Mix_GPU CF + FAISS fusion — personalized and fast',
    color: 'emerald',
    requires_user: true,
  },
  P1: {
    name: 'Sequential CF',
    badge: 'FASTEST',
    description: 'Pure collaborative filtering — lowest latency',
    color: 'amber',
    requires_user: true,
  },
  P3: {
    name: 'Cold Start',
    badge: 'NO LOGIN',
    description: 'Content-only FAISS retrieval — works without a user profile',
    color: 'rose',
    requires_user: false,
  },
  auto: {
    name: 'Auto Select',
    badge: 'AUTO',
    description: 'Picks the best pipeline based on query complexity and user context',
    color: 'slate',
    requires_user: false,
  },
}

// ── TMDB poster lookup (uses the same key from spidey-search) ──────────
const TMDB_KEY = 'ad00441f8862f6a0dd2ea3a9f927b8cd'
const TMDB_BASE = 'https://api.themoviedb.org/3'
export const TMDB_IMG = 'https://image.tmdb.org/t/p/w500'
export const TMDB_IMG_SM = 'https://image.tmdb.org/t/p/w185'
export const TMDB_IMG_LG = 'https://image.tmdb.org/t/p/w1280'

const posterCache = new Map<string, string | null>()

export async function fetchPoster(title: string, year?: string | number): Promise<string | null> {
  const cacheKey = `${title}|${year ?? ''}`
  if (posterCache.has(cacheKey)) return posterCache.get(cacheKey)!

  try {
    const yearStr = year ? String(year).slice(0, 4) : ''
    const q = encodeURIComponent(title)
    const url = yearStr
      ? `${TMDB_BASE}/search/movie?api_key=${TMDB_KEY}&query=${q}&year=${yearStr}&language=en-US`
      : `${TMDB_BASE}/search/movie?api_key=${TMDB_KEY}&query=${q}&language=en-US`

    const res = await fetch(url)
    if (!res.ok) { posterCache.set(cacheKey, null); return null }
    const data = await res.json()
    const hit = data.results?.[0]
    const url2 = hit?.poster_path ? `${TMDB_IMG}${hit.poster_path}` : null
    posterCache.set(cacheKey, url2)
    return url2
  } catch {
    posterCache.set(cacheKey, null)
    return null
  }
}

export async function fetchMovieDetails(title: string, year?: string | number) {
  try {
    const q = encodeURIComponent(title)
    const yearStr = year ? String(year).slice(0, 4) : ''
    const url = yearStr
      ? `${TMDB_BASE}/search/movie?api_key=${TMDB_KEY}&query=${q}&year=${yearStr}&language=en-US`
      : `${TMDB_BASE}/search/movie?api_key=${TMDB_KEY}&query=${q}&language=en-US`

    const res = await fetch(url)
    if (!res.ok) return null
    const data = await res.json()
    const hit = data.results?.[0]
    if (!hit) return null

    // Fetch full details + credits
    const [detail, credits] = await Promise.all([
      fetch(`${TMDB_BASE}/movie/${hit.id}?api_key=${TMDB_KEY}&language=en-US`).then(r => r.json()),
      fetch(`${TMDB_BASE}/movie/${hit.id}/credits?api_key=${TMDB_KEY}&language=en-US`).then(r => r.json()),
    ])

    return { detail, credits, tmdbId: hit.id }
  } catch {
    return null
  }
}

// Assign a meaningful signal badge based on score tier
export function assignSignal(score: number, rank: number): MovieSignal {
  if (rank === 1 || score > 1.0) return 'Top Pick'
  if (score > 0.5) return 'Trending'
  return 'Hidden Gem'
}

export async function fetchRecommend(params: {
  query: string
  user_id: number | null
  pipeline: PipelineKey
  n: number
}): Promise<RecommendResult> {
  const res = await fetch('/api/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail ?? `HTTP ${res.status}`)
  }
  const data: RecommendResult = await res.json()
  // Normalize score + assign signals
  data.movies = data.movies.slice(0, params.n).map((m, i) => ({
    ...m,
    _score: (m.fused_score ?? m.mix_score ?? m.faiss_score_norm ?? m.score ?? 0) as number,
    signal: assignSignal((m.fused_score ?? m.mix_score ?? m.faiss_score_norm ?? m.score ?? 0) as number, i),
  }))
  return data
}
