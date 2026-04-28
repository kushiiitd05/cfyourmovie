import { motion, AnimatePresence } from 'framer-motion'
import { Zap, ChevronDown } from 'lucide-react'
import { useState } from 'react'
import { PIPELINE_META } from '@/api'
import { pipelineKey } from '@/lib/utils'
import { cn } from '@/lib/utils'
import type { RecommendResult } from '@/types'
import type { PipelineColor } from '@/types'

const BADGE_COLOR: Record<PipelineColor, string> = {
  violet:  'bg-white/5 text-accent border-accent/20',
  blue:    'bg-white/5 text-accent border-accent/20',
  emerald: 'bg-white/5 text-accent border-accent/20',
  amber:   'bg-accent/20 text-accent-bright border-accent/30',
  rose:    'bg-white/5 text-accent border-accent/20',
  slate:   'bg-white/5 text-text-muted border-white/10',
}

interface Props {
  result: RecommendResult
}

export default function ExplanationPanel({ result }: Props) {
  const [showExtras, setShowExtras] = useState(false)
  const key = pipelineKey(result.pipeline)
  const meta = PIPELINE_META[key as keyof typeof PIPELINE_META] ?? PIPELINE_META.auto
  const hasExtras = (result.sub_queries?.length ?? 0) > 0 || !!result.hypothetical

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="glass rounded-2xl p-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-full bg-amber-500/10 flex items-center justify-center border border-amber-500/20">
            <Zap size={11} className="text-accent" />
          </div>
          <div>
            <span className="text-[10px] font-bold text-text-secondary uppercase tracking-widest">AI Curation Note</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <span className={cn('text-[9px] font-bold px-1.5 py-0.5 rounded border uppercase tracking-wider', BADGE_COLOR[meta.color])}>
                {result.pipeline}
              </span>
              {result.user_id && (
                <span className="text-[10px] text-zinc-600">User {result.user_id}</span>
              )}
            </div>
          </div>
        </div>

        {hasExtras && (
          <button
            onClick={() => setShowExtras(s => !s)}
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            Details
            <motion.span animate={{ rotate: showExtras ? 180 : 0 }} transition={{ duration: 0.2 }}>
              <ChevronDown size={13} />
            </motion.span>
          </button>
        )}
      </div>

      {/* Explanation text */}
      <p className="text-sm text-slate-400 leading-relaxed">{result.explanation}</p>

      {/* Expandable extras (P5 sub-queries + HyDE) */}
      <AnimatePresence>
        {showExtras && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="mt-4 pt-4 border-t border-white/[0.06] space-y-4">
              {(result.sub_queries?.length ?? 0) > 0 && (
                <div>
                  <p className="text-[10px] text-slate-600 mb-2 uppercase tracking-widest">Sub-queries decomposed</p>
                  <div className="flex flex-wrap gap-2">
                    {result.sub_queries!.map((sq, i) => (
                      <span key={i} className="text-xs bg-white/5 border border-white/10 rounded-full px-3 py-1 text-slate-400">
                        {sq}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {result.hypothetical && (
                <div>
                  <p className="text-[10px] text-slate-600 mb-1.5 uppercase tracking-widest">HyDE hypothesis</p>
                  <p className="text-xs text-slate-500 italic leading-relaxed line-clamp-4">{result.hypothetical}</p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}
