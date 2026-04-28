import * as TabsPrimitive from '@radix-ui/react-tabs'
import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import { PIPELINE_META } from '@/api'
import type { PipelineKey } from '@/types'

interface Props {
  value: PipelineKey
  onChange: (v: PipelineKey) => void
}

const PIPELINE_ORDER: PipelineKey[] = ['P5', 'P4', 'P2', 'P1', 'P3', 'auto']

export default function PipelineSelector({ value, onChange }: Props) {
  const selected = PIPELINE_META[value]

  return (
    <div className="w-full">
      <TabsPrimitive.Root value={value} onValueChange={v => onChange(v as PipelineKey)}>
        <TabsPrimitive.List className="flex flex-wrap gap-1.5 mb-3" aria-label="Pipeline">
          {PIPELINE_ORDER.map(key => {
            const meta = PIPELINE_META[key]
            const isActive = value === key
            return (
              <TabsPrimitive.Trigger
                key={key}
                value={key}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded border text-[10px] font-bold uppercase tracking-widest transition-all duration-200 cursor-pointer',
                  isActive
                    ? 'border-amber-500/50 bg-amber-500/8 text-amber-400'
                    : 'border-white/5 bg-transparent text-zinc-600 hover:text-zinc-400 hover:border-white/10'
                )}
              >
                {/* Badge — Minimal Stark */}
                <span className={cn(
                  'text-[8px] font-black px-1.5 py-0.5 rounded-sm',
                  isActive ? 'bg-amber-500 text-black' : 'bg-zinc-900 text-zinc-600'
                )}>
                  {meta.badge}
                </span>
                <span>{meta.name}</span>
              </TabsPrimitive.Trigger>
            )
          })}
        </TabsPrimitive.List>
      </TabsPrimitive.Root>

      {/* Description of selected pipeline */}
      <motion.p
        key={value}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="text-[10px] uppercase tracking-wider text-zinc-600 font-medium"
      >
        {selected.description}
        {selected.requires_user && (
          <span className="ml-2 text-zinc-500 font-bold opacity-60">· REQUIRES USER ID</span>
        )}
      </motion.p>
    </div>
  )
}
