import { motion } from 'framer-motion'

function SkeletonCard({ delay }: { delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay }}
      className="flex-shrink-0 rounded-xl overflow-hidden"
      style={{ width: 200, height: 300, background: 'var(--surface-2)', border: '1px solid rgba(255,255,255,0.04)' }}
    >
      {/* Poster area (75%) */}
      <div className="shimmer" style={{ height: '75%', width: '100%' }} />
      {/* Info strip (25%) */}
      <div className="p-3 space-y-2" style={{ height: '25%', background: 'var(--surface-2)' }}>
        <div className="h-3 w-3/4 rounded-full shimmer" />
        <div className="h-2 w-1/2 rounded-full shimmer" />
      </div>
    </motion.div>
  )
}

export default function SkeletonGrid({ count = 8 }: { count?: number }) {
  return (
    <div className="poster-rail px-1 pb-4">
      {Array.from({ length: count }, (_, i) => (
        <SkeletonCard key={i} delay={i * 0.04} />
      ))}
    </div>
  )
}
