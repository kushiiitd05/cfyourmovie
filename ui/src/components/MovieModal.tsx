import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Star, X, Film } from 'lucide-react'
import { fetchMovieDetails, TMDB_IMG, TMDB_IMG_LG, TMDB_IMG_SM } from '@/api'
import { genreLabel } from '@/lib/utils'
import type { Movie } from '@/types'

interface Props {
  movie: Movie
  onClose: () => void
}

export default function MovieModal({ movie, onClose }: Props) {
  const [details, setDetails] = useState<Record<string, unknown> | null>(null)
  const [loading, setLoading] = useState(true)
  const overlayRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchMovieDetails(movie.title, movie.year).then(r => {
      setDetails(r)
      setLoading(false)
    })
  }, [movie.title, movie.year])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const detail = (details as Record<string, unknown> | null)?.detail as Record<string, unknown> | undefined
  const credits = (details as Record<string, unknown> | null)?.credits as Record<string, unknown> | undefined
  const cast = ((credits?.cast as unknown[]) ?? []).slice(0, 5) as Array<Record<string, unknown>>

  const poster = detail?.poster_path ? `${TMDB_IMG}${detail.poster_path}` : movie.poster_url ?? null
  const backdrop = detail?.backdrop_path ? `${TMDB_IMG_LG}${detail.backdrop_path}` : null
  const rating = detail?.vote_average ? Number(detail.vote_average).toFixed(1) : null
  const year = String(detail?.release_date ?? movie.year ?? '').slice(0, 4) || null
  const overview = detail?.overview as string | undefined
  const tagline = detail?.tagline as string | undefined
  const runtime = detail?.runtime as number | undefined
  const tmdbGenres = (detail?.genres as Array<{ name: string }> | undefined)?.map(g => g.name) ?? []
  const displayGenres = tmdbGenres.length ? tmdbGenres : (
    Array.isArray(movie.genres) ? movie.genres : String(movie.genres ?? '').split('|').filter(Boolean)
  )
  const imdbId = (detail?.imdb_id || movie.imdb_id) as string | undefined
  const streamUrl = imdbId ? `https://www.playimdb.com/title/${imdbId}/` : null

  return (
    <motion.div
      ref={overlayRef}
      className="modal-backdrop"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={(e) => { if (e.target === overlayRef.current) onClose() }}
    >
      <motion.div
        className="relative w-full max-w-3xl mx-4 rounded-3xl overflow-hidden"
        style={{
          background: 'var(--s1)',
          border: '1px solid rgba(255,255,255,0.08)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.9), 0 0 0 1px rgba(217,119,6,0.15)',
          maxHeight: '90vh',
          overflowY: 'auto',
        }}
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 20, scale: 0.97 }}
        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
      >
        {/* Blurred backdrop image */}
        {backdrop && (
          <div
            className="absolute inset-0 opacity-10 pointer-events-none"
            style={{
              backgroundImage: `url(${backdrop})`,
              backgroundSize: 'cover',
              backgroundPosition: 'center top',
              filter: 'blur(3px)',
            }}
          />
        )}

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-20 w-9 h-9 rounded-full flex items-center justify-center transition-all hover:scale-110"
          style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid rgba(255,255,255,0.12)' }}
          aria-label="Close"
          id="modal-close-btn"
        >
          <X size={16} className="text-white" />
        </button>

        <div className="relative z-10 p-6 md:p-8">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-16 gap-4">
              <div className="w-10 h-10 rounded-full border-2 border-amber-500/30 border-t-amber-400 animate-spin" />
              <p className="text-sm text-text-muted">Fetching details…</p>
            </div>
          ) : (
            <>
              {/* Top: poster + meta */}
              <div className="flex gap-6 mb-6">
                {/* Poster */}
                <div
                  className="flex-shrink-0 rounded-2xl overflow-hidden"
                  style={{ width: 140, minWidth: 140, height: 210, boxShadow: '0 12px 40px rgba(0,0,0,0.6)' }}
                >
                  {poster ? (
                    <img src={poster} alt={movie.title} className="w-full h-full object-cover" loading="lazy" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center" style={{ background: 'rgba(255,255,255,0.04)' }}>
                      <Film size={40} className="text-white/20" />
                    </div>
                  )}
                </div>

                {/* Meta */}
                <div className="flex flex-col gap-2 min-w-0 flex-1 pt-1">
                  {/* Signal badge */}
                  {movie.signal && (
                    <span
                      className={`inline-flex self-start text-[10px] font-bold px-2.5 py-1 rounded-full uppercase tracking-wider signal-${movie.signal.toLowerCase().replace(' ', '-')}`}
                    >
                      {movie.signal === 'Top Pick' ? '⭐ Top Pick' : movie.signal === 'Trending' ? '🔥 Trending' : '💎 Hidden Gem'}
                    </span>
                  )}

                  <h2
                    className="text-xl md:text-2xl font-bold text-white leading-tight"
                    style={{ letterSpacing: '-0.02em' }}
                  >
                    {movie.title}
                  </h2>

                  {tagline && (
                    <p className="text-xs italic" style={{ color: 'var(--accent-1)' }}>&quot;{tagline}&quot;</p>
                  )}

                  {/* Rating + year + runtime */}
                  <div className="flex items-center gap-3 flex-wrap mt-1">
                    {rating && (
                      <div className="flex items-center gap-1.5">
                        <Star size={13} className="fill-amber-400 text-amber-400" />
                        <span className="text-sm font-bold text-amber-300">{rating}</span>
                        <span className="text-xs" style={{ color: 'var(--text-muted)' }}>/10</span>
                      </div>
                    )}
                    {year && <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>{year}</span>}
                    {runtime && <span className="text-xs px-2 py-0.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)', color: 'var(--text-secondary)' }}>{Math.floor(runtime / 60)}h {runtime % 60}m</span>}
                    {imdbId && (
                      <a
                        href={`https://www.imdb.com/title/${imdbId}/`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] font-bold px-2 py-0.5 rounded transition-opacity hover:opacity-80"
                        style={{ background: '#F5C518', color: '#000' }}
                      >
                        IMDb ↗
                      </a>
                    )}
                    {streamUrl && (
                      <a
                        href={streamUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] font-bold px-3 py-1 rounded transition-opacity hover:opacity-80 flex items-center gap-1"
                        style={{ background: 'rgba(217,119,6,0.2)', border: '1px solid rgba(217,119,6,0.3)', color: 'var(--accent-1)' }}
                      >
                        <Film size={12} /> Stream
                      </a>
                    )}
                  </div>

                  {/* Genres */}
                  {displayGenres.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {displayGenres.slice(0, 5).map(g => (
                        <span
                          key={g}
                          className="text-[10px] font-medium px-2.5 py-0.5 rounded-full"
                          style={{ background: 'rgba(217,119,6,0.1)', border: '1px solid rgba(217,119,6,0.2)', color: 'var(--accent-1)' }}
                        >
                          {g}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Overview */}
              {overview && (
                <div className="mb-6">
                  <h3 className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>Overview</h3>
                  <p className="text-sm leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{overview}</p>
                </div>
              )}

              {/* Cast */}
              {cast.length > 0 && (
                <div>
                  <h3 className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>Cast</h3>
                  <div className="flex gap-3 flex-wrap">
                    {cast.map(actor => (
                      <div key={String(actor.id)} className="flex flex-col items-center gap-1.5 text-center" style={{ width: 72 }}>
                        {actor.profile_path ? (
                          <img
                            src={`${TMDB_IMG_SM}${actor.profile_path}`}
                            alt={String(actor.name ?? '')}
                            className="w-14 h-14 rounded-full object-cover"
                            style={{ border: '2px solid rgba(217,119,6,0.2)' }}
                            loading="lazy"
                          />
                        ) : (
                          <div
                            className="w-14 h-14 rounded-full flex items-center justify-center"
                            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)' }}
                          >
                            <span className="text-xl">👤</span>
                          </div>
                        )}
                        <span className="text-[10px] font-semibold text-white leading-tight line-clamp-2">{String(actor.name ?? '')}</span>
                        <span className="text-[9px] leading-tight" style={{ color: 'var(--text-muted)' }}>{String(actor.character ?? '')}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}
