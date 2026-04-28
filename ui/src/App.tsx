import { useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import SearchHero from '@/components/SearchHero'
import PipelineSelector from '@/components/PipelineSelector'
import ResultsRail from '@/components/ResultsRail'
import SkeletonGrid from '@/components/SkeletonGrid'
import ExplanationPanel from '@/components/ExplanationPanel'
import MovieModal from '@/components/MovieModal'
import { fetchRecommend } from '@/api'
import type { RecommendResult, AppState, PipelineKey, Movie } from '@/types'

const GENRE_CHIPS = [
  { label: 'Dark Thriller', query: 'dark psychological thriller' },
  { label: 'Sci-Fi Epic', query: 'epic science fiction space adventure' },
  { label: '90s Classics', query: 'feel-good 90s comedies' },
  { label: 'Crime Drama', query: 'crime drama with complex characters' },
  { label: 'Atmospheric Horror', query: 'atmospheric horror isolated places' },
  { label: 'Romance', query: 'beautiful romance with emotional depth' },
]

export default function App() {
  const [query, setQuery] = useState('')
  const [userId, setUserId] = useState<number | null>(null)
  const [pipeline, setPipeline] = useState<PipelineKey>('P5')
  const [n, setN] = useState(8)
  const [state, setState] = useState<AppState>('idle')
  const [result, setResult] = useState<RecommendResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null)

  const handleSearch = async (overrideQuery?: string) => {
    const q = (overrideQuery ?? query).trim()
    if (!q) return
    if (overrideQuery) setQuery(overrideQuery)
    setState('loading')
    setError(null)
    setResult(null)
    try {
      const data = await fetchRecommend({ query: q, user_id: userId, pipeline, n })
      setResult(data)
      setState('results')
    } catch (e) {
      setError((e as Error).message)
      setState('error')
    }
  }

  const handleClear = () => {
    setState('idle')
    setResult(null)
    setError(null)
  }

  return (
    <div className="relative z-10 min-h-screen flex flex-col">

      {/* ── Navigation ───────────────────────────────────── */}
      <header className="glass-nav sticky top-0 z-40 flex items-center justify-between px-6 py-3.5">
        {/* Wordmark — Minimal Noir Style */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-sm bg-accent flex items-center justify-center text-black font-black text-xs tracking-tighter">
            CF
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-base font-black text-white uppercase tracking-tighter">
              CfYourMovie
            </span>
          </div>
        </div>

        {/* Status pill — Minimal Amber label */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1 bg-s2 border border-white/6 rounded-sm text-[10px] font-bold uppercase tracking-widest text-text-secondary">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse" />
          Cinema Mode · 1,682 films
        </div>
      </header>

      {/* ── Hero ─────────────────────────────────────────── */}
      <section className="relative flex flex-col items-center px-4 pt-24 pb-12" style={{ minHeight: state === 'idle' ? '65vh' : 'auto' }}>

        <AnimatePresence>
          {state === 'idle' && (
            <motion.div
              key="hero-text"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              className="text-center mb-16 w-full max-w-4xl"
            >
              <h1 className="text-hero text-white mb-6 uppercase">
                Selection<br />is an art.
              </h1>
              <p className="text-[10px] uppercase font-bold tracking-[0.2em]" style={{ color: 'var(--accent-1)' }}>
                Intelligent Recommendations for the Cinematic Mind
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Search — open layout, no card wrapper */}
        <div className="w-full max-w-2xl">
          <SearchHero
            value={query}
            onChange={setQuery}
            onSubmit={handleSearch}
            loading={state === 'loading'}
          />

          {/* Genre chips */}
          <AnimatePresence>
            {state === 'idle' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="flex flex-wrap gap-2 mt-4 mb-5"
              >
                {GENRE_CHIPS.map(chip => (
                  <button
                    key={chip.label}
                    onClick={() => handleSearch(chip.query)}
                    className="genre-chip"
                    id={`genre-chip-${chip.label.toLowerCase().replace(' ', '-')}`}
                  >
                    {chip.label}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Controls row */}
          <div className="flex flex-col sm:flex-row gap-3 mb-5">
            <div className="flex-1">
              <label className="text-[10px] uppercase tracking-widest mb-1.5 block" style={{ color: 'var(--t-muted)' }}>
                User ID <span style={{ opacity: 0.5 }}>(1–943, optional)</span>
              </label>
              <input
                type="number"
                min={1}
                max={943}
                placeholder="e.g. 42  (for personalized results)"
                value={userId ?? ''}
                onChange={e => setUserId(e.target.value ? Number(e.target.value) : null)}
                className="w-full px-4 py-2.5 rounded text-white placeholder:text-zinc-700 text-sm bg-white/[0.03] border border-white/[0.07] focus:outline-none focus:border-amber-500/50 focus:ring-1 focus:ring-amber-500/15 transition"
                id="user-id-input"
              />
            </div>
            <div className="w-28">
              <label className="text-[10px] uppercase tracking-widest mb-1.5 block" style={{ color: 'var(--t-muted)' }}>Results</label>
              <select
                value={n}
                onChange={e => setN(Number(e.target.value))}
                className="w-full px-4 py-2.5 rounded text-white text-sm bg-white/[0.03] border border-white/[0.07] focus:outline-none focus:border-amber-500/50 transition appearance-none cursor-pointer"
                id="results-count-select"
              >
                {[5, 8, 10, 15].map(v => <option key={v} value={v}>{v}</option>)}
              </select>
            </div>
          </div>
 
          <div>
            <label className="text-[10px] uppercase tracking-widest mb-2 block" style={{ color: 'var(--t-muted)' }}>Pipeline</label>
            <PipelineSelector value={pipeline} onChange={setPipeline} />
          </div>
        </div>
      </section>

      {/* ── Results ──────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 pb-24">
        <AnimatePresence mode="wait">

          {/* Loading */}
          {state === 'loading' && (
            <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <p className="text-xs uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>
                Searching through 1,682 films…
              </p>
              <SkeletonGrid count={n} />
            </motion.div>
          )}

          {/* Error */}
          {state === 'error' && (
            <motion.div
              key="error"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass rounded-2xl p-5"
              style={{ border: '1px solid rgba(239,68,68,0.3)' }}
            >
              <p className="text-sm text-red-300">{error}</p>
              <button
                onClick={handleClear}
                className="mt-2 text-xs underline"
                style={{ color: 'rgba(239,68,68,0.6)' }}
                id="error-dismiss-btn"
              >
                Dismiss
              </button>
            </motion.div>
          )}

          {/* Results */}
          {state === 'results' && result && (
            <motion.div key="results" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <ResultsRail
                result={result}
                onMovieClick={setSelectedMovie}
                onClear={handleClear}
              />

              {result.explanation && (
                <div className="mt-8">
                  <ExplanationPanel result={result} />
                </div>
              )}
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      {/* ── Movie detail modal ────────────────────────────── */}
      <AnimatePresence>
        {selectedMovie && (
          <MovieModal
            key="modal"
            movie={selectedMovie}
            onClose={() => setSelectedMovie(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
