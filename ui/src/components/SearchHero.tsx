'use client'
import React, { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Search } from 'lucide-react'
import { cn } from '@/lib/utils'

const EXAMPLE_QUERIES = [
  'dark psychological thriller',
  'feel-good 90s comedies',
  'films like Blade Runner but hopeful',
  'crime drama with complex characters',
  'atmospheric horror set in isolated places',
  'underdog sports story with heart',
]

interface Props {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  loading: boolean
}

export default function SearchHero({ value, onChange, onSubmit, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isFocused, setIsFocused] = useState(false)
  const [placeholder, setPlaceholder] = useState(EXAMPLE_QUERIES[0])

  // Cycle placeholder examples
  useEffect(() => {
    if (isFocused || value) return
    let i = 0
    const id = setInterval(() => {
      i = (i + 1) % EXAMPLE_QUERIES.length
      setPlaceholder(EXAMPLE_QUERIES[i])
    }, 4000)
    return () => clearInterval(id)
  }, [isFocused, value])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!value.trim() || loading) return
    onSubmit()
  }

  return (
    <div className="relative w-full mx-auto">
      <motion.form
        onSubmit={handleSubmit}
        className="relative"
        animate={{ scale: isFocused ? 1.01 : 1 }}
        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
      >
        <div
          className={cn(
            'flex items-center w-full rounded-xl border transition-all duration-300',
            isFocused
              ? 'border-amber-500/30 bg-s2 shadow-[0_0_0_1px_rgba(217,119,6,0.15)]'
              : 'border-white/6 bg-s1'
          )}
        >
          {/* Search icon */}
          <div className="pl-5">
            <Search
              size={18}
              className={cn(
                'transition-colors duration-300',
                isFocused ? 'text-white' : 'text-zinc-600'
              )}
            />
          </div>

          <input
            ref={inputRef}
            type="text"
            placeholder={placeholder}
            value={value}
            onChange={e => onChange(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setTimeout(() => setIsFocused(false), 150)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit(e as unknown as React.FormEvent)}
            className="flex-1 py-5 px-4 bg-transparent outline-none text-white placeholder:text-zinc-700 text-base font-medium"
          />

          {/* Submit button — Minimal Stark Style */}
          <AnimatePresence>
            {value && (
              <motion.button
                type="submit"
                disabled={loading}
                initial={{ opacity: 0, x: 10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 10 }}
                whileTap={{ scale: 0.98 }}
                className="px-6 py-2 mr-3 text-xs font-bold uppercase tracking-widest rounded bg-white text-black hover:bg-zinc-200 transition-colors disabled:opacity-50"
              >
                {loading ? 'Searching' : 'Discover'}
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </motion.form>
    </div>
  )
}
