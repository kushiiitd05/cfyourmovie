import { useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import MovieCard from '@/components/MovieCard'
import type { Movie, RecommendResult } from '@/types'

interface Props {
  result: RecommendResult
  onMovieClick: (movie: Movie) => void
  onClear: () => void
}

const PIPELINE_LABELS: Record<string, string> = {
  P5: 'Hybrid Deep AI',
  P4: 'HyDE Hybrid',
  P2: 'Dual Engine',
  P1: 'Sequential CF',
  P3: 'Cold Start',
  auto: 'Auto',
}

export default function ResultsRail({ result, onMovieClick, onClear }: Props) {
  const railRef = useRef<HTMLDivElement>(null)

  const scroll = (dir: 'left' | 'right') => {
    if (!railRef.current) return
    const amount = dir === 'right' ? 432 : -432
    railRef.current.scrollBy({ left: amount, behavior: 'smooth' })
  }

  const pipelineLabel = PIPELINE_LABELS[result.pipeline?.match(/^(P\d)/)?.[1] ?? ''] ?? result.pipeline

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.6, ease: [0.2, 1, 0.2, 1] }}
      className="w-full"
    >
      {/* Section header */}
      <div className="flex items-center justify-between mb-5 px-1 pt-4">
        <div>
          <h2
            className="text-sm font-bold text-white uppercase tracking-widest"
          >
            {result.movies.length} recommendations
          </h2>
          <p className="text-xs mt-1" style={{ color: 'var(--t-secondary)' }}>
            Found in database via {pipelineLabel}
          </p>
        </div>

        <button
          onClick={onClear}
          className="text-[10px] px-3.5 py-1.5 rounded uppercase font-bold tracking-widest transition-all"
          style={{
            color: 'var(--t-secondary)',
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid rgba(255,255,255,0.04)',
          }}
          id="results-clear-btn"
        >
          Reset
        </button>
      </div>

      {/* Rail wrapper */}
      <div className="relative group">
        {/* Left arrow */}
        <button
          onClick={() => scroll('left')}
          className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 z-20 w-10 h-10 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all shadow-2xl"
          style={{ background: '#000', border: '1px solid rgba(255,255,255,0.1)' }}
          aria-label="Scroll left"
          id="rail-scroll-left"
        >
          <ChevronLeft size={18} className="text-white" />
        </button>

        {/* The rail */}
        <div ref={railRef} className="poster-rail px-1 pb-6 gap-5">
          <AnimatePresence>
            {result.movies.map((movie, i) => (
              <div key={`${movie.title}-${i}`} className="poster-rail-item">
                <MovieCard
                  movie={movie}
                  rank={i + 1}
                  delay={i * 0.05}
                  onClick={onMovieClick}
                />
              </div>
            ))}
          </AnimatePresence>
        </div>

        {/* Right arrow */}
        <button
          onClick={() => scroll('right')}
          className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 z-20 w-10 h-10 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-all shadow-2xl"
          style={{ background: '#000', border: '1px solid rgba(255,255,255,0.1)' }}
          aria-label="Scroll right"
          id="rail-scroll-right"
        >
          <ChevronRight size={18} className="text-white" />
        </button>

        {/* Right fade */}
        <div
          className="absolute right-0 top-0 bottom-4 w-16 pointer-events-none"
          style={{ background: 'linear-gradient(to left, #000 0%, transparent 100%)' }}
        />
      </div>
    </motion.div>
  )
}
