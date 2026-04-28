import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Star, Film } from 'lucide-react'
import { fetchPoster } from '@/api'
import { genreLabel } from '@/lib/utils'
import type { Movie } from '@/types'

interface Props {
  movie: Movie
  rank: number
  delay?: number
  onClick: (movie: Movie) => void
}

const SIGNAL_STYLES: Record<string, string> = {
  'Top Pick':   'signal-top-pick',
  'Trending':   'signal-trending',
  'Hidden Gem': 'signal-hidden-gem',
}
const SIGNAL_ICONS: Record<string, string> = {
  'Top Pick':   '⭐',
  'Trending':   '🔥',
  'Hidden Gem': '💎',
}

// Cinematic gradient fallback per genre
const GENRE_GRADIENTS: Record<string, string> = {
  action:    'linear-gradient(145deg, #1a0a00, #3d1500)',
  thriller:  'linear-gradient(145deg, #0a0a1a, #1a0a2e)',
  horror:    'linear-gradient(145deg, #0f0000, #1a0a0a)',
  comedy:    'linear-gradient(145deg, #0a1a00, #1a2e0a)',
  romance:   'linear-gradient(145deg, #1a000a, #2e0a1a)',
  drama:     'linear-gradient(145deg, #0a0a1a, #1a1a2e)',
  scifi:     'linear-gradient(145deg, #001a1a, #0a2e2e)',
  sci:       'linear-gradient(145deg, #001a1a, #0a2e2e)',
  fantasy:   'linear-gradient(145deg, #1a0a1a, #2e1a3d)',
  default:   'linear-gradient(145deg, #0d0d1a, #151528)',
}

function getGradient(genres?: string | string[]): string {
  const str = Array.isArray(genres) ? genres.join(' ') : (genres ?? '')
  const key = Object.keys(GENRE_GRADIENTS).find(k => str.toLowerCase().includes(k))
  return GENRE_GRADIENTS[key ?? 'default']
}

export default function MovieCard({ movie, rank, delay = 0, onClick }: Props) {
  const [poster, setPoster] = useState<string | null>(movie.poster_url ?? null)
  const [imgError, setImgError] = useState(false)
  const [hovered, setHovered] = useState(false)
  const cardRef = useRef<HTMLDivElement>(null)
  const [tilt, setTilt] = useState({ x: 0, y: 0 })

  const rating = movie.vote_average ?? movie.rating
  const genres = genreLabel(movie.genres)

  // Fetch TMDB poster
  useEffect(() => {
    if (!movie.poster_url) {
      fetchPoster(movie.title, movie.year).then(url => {
        if (url) setPoster(url)
      })
    }
  }, [movie.title, movie.year, movie.poster_url])

  const handleMouseLeave = () => {
    setHovered(false)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay, ease: [0.2, 1, 0.2, 1] }}
    >
      <motion.div
        ref={cardRef}
        className="poster-card"
        style={{
          width: 190,
          height: 280,
        }}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={handleMouseLeave}
        onClick={() => onClick(movie)}
        whileHover={{ y: -4, scale: 1.02 }}
        id={`movie-card-${rank}`}
      >
        {/* ── Poster image (75% of card) ── */}
        <div className="relative" style={{ height: '75%', overflow: 'hidden' }}>
          {poster && !imgError ? (
            <img
              src={poster}
              alt={movie.title}
              className="w-full h-full object-cover"
              loading="lazy"
              onError={() => setImgError(true)}
            />
          ) : (
            <div
              className="w-full h-full flex items-center justify-center"
              style={{ background: getGradient(movie.genres) }}
            >
              <Film size={36} className="text-white/15" />
            </div>
          )}

          {/* Gradient overlay bottom — Darker & Sharper */}
          <div
            className="absolute bottom-0 left-0 right-0 h-1/2 pointer-events-none"
            style={{
              background: 'linear-gradient(to top, #000 0%, rgba(0,0,0,0.8) 40%, transparent 100%)',
            }}
          />

          {/* Rank badge */}
          <div
            className="absolute top-2.5 left-2.5 w-6 h-6 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
            style={{ background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)' }}
          >
            {rank}
          </div>

          {/* Rating badge */}
          {rating !== undefined && (
            <div
              className="absolute top-2.5 right-2.5 flex items-center gap-1 px-1.5 py-0.5 rounded-md"
              style={{ background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)' }}
            >
              <Star size={9} className="fill-amber-400 text-amber-400" />
              <span className="text-[10px] font-bold text-amber-300">{Number(rating).toFixed(1)}</span>
            </div>
          )}

          {/* Hover preview overlay */}
          <motion.div
            className="absolute inset-0 flex items-end pointer-events-none"
            initial={{ opacity: 0 }}
            animate={{ opacity: hovered ? 1 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <div
              className="w-full px-3 pb-3 pt-6"
              style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%)' }}
            >
              <p className="text-[9px] text-white/60 uppercase tracking-wider">Click for details</p>
            </div>
          </motion.div>
        </div>

        {/* ── Info strip (25% of card) ── */}
        <div className="px-3 pt-2.5 pb-4 flex flex-col gap-1" style={{ height: '25%', background: '#050505' }}>
          <h3
            className="text-[11px] font-bold text-white leading-tight line-clamp-2"
            style={{ letterSpacing: '-0.02em' }}
          >
            {movie.title}
          </h3>

          <div className="flex items-center justify-between gap-1 mt-auto">
            <span className="text-[10px] font-medium" style={{ color: 'var(--t-secondary)', letterSpacing: '-0.01em' }}>
              {movie.year} · {genres}
            </span>
          </div>
        </div>

        {/* Amber hover edge */}
        {hovered && (
          <div
            className="absolute inset-0 rounded pointer-events-none"
            style={{ boxShadow: 'inset 0 0 0 1px rgba(217,119,6,0.15)' }}
          />
        )}
      </motion.div>
    </motion.div>
  )
}
