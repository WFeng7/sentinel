import React from 'react'
import { useVLMRAGLoop } from '../hooks/useVLMRAGLoop.js'

export default function VLMRAGTile({ stream, enabled, onToggle }) {
  const { result, loading, error, start, stop } = useVLMRAGLoop(stream, enabled)

  React.useEffect(() => {
    if (enabled) start()
    else stop()
  }, [enabled])

  if (!enabled) {
    return (
      <button
        className="absolute right-2 top-2 z-20 hidden h-8 w-8 items-center justify-center border border-white/20 bg-slate-900/80 text-xs font-semibold text-slate-100 opacity-0 transition group-hover:flex group-hover:opacity-100"
        onClick={() => onToggle(true)}
        type="button"
        title="Enable VLM+RAG analysis"
      >
        AI
      </button>
    )
  }

  const rag = result?.rag
  const ragSources = rag?.supporting_excerpts || []
  const ragSummary = rag?.explanation || ragSources[0]?.text || 'No decision'

  return (
    <div className="absolute right-2 top-2 z-20 flex flex-col gap-1">
      <button
        className="h-8 w-8 border border-white/20 bg-slate-900/80 text-xs font-semibold text-slate-100"
        onClick={() => onToggle(false)}
        type="button"
        title="Disable VLM+RAG analysis"
      >
        ×
      </button>
      {loading && <div className="h-2 w-2 bg-yellow-400 animate-pulse" />}
      {error && <div className="max-w-32 truncate text-xs text-rose-400" title={error}>Error</div>}
      {result && (
        <div className="max-w-56 text-xs text-slate-200">
          <div className="font-semibold">VLM</div>
          <div className="truncate">{result.vlm?.narrative || 'No narrative'}</div>
          <div className="font-semibold mt-1">RAG</div>
          <div className="line-clamp-2 text-slate-200">{ragSummary}</div>
          {ragSources.length > 0 && (
            <div className="mt-1 space-y-1 text-[10px] text-slate-400">
              {ragSources.slice(0, 2).map((excerpt) => (
                <div key={excerpt.document_id} className="truncate">
                  {excerpt.document_id}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
