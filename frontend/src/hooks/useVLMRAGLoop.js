import { useState, useEffect, useRef } from 'react'
import { analyzeWithVLMAndRAG } from '../services/vlmRag.js'

export function useVLMRAGLoop(camera, enabled = false) {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const intervalRef = useRef(null)
  const startRef = useRef(null)

  const run = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await analyzeWithVLMAndRAG(camera)
      setResult(res)
    } catch (e) {
      setError(e.message || 'VLM+RAG failed')
    } finally {
      setLoading(false)
    }
  }

  const start = () => {
    if (!camera || intervalRef.current) return
    enabled = true
    run() // immediate run
    startRef.current = Date.now()
    intervalRef.current = setInterval(() => {
      // Align to 60s cycles from the first run
      const elapsed = Date.now() - startRef.current
      const delay = 60000 - (elapsed % 60000)
      setTimeout(() => run(), delay)
    }, 60000)
  }

  const stop = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    enabled = false
  }

  useEffect(() => {
    if (enabled && camera && !intervalRef.current) {
      start()
    } else if (!enabled && intervalRef.current) {
      stop()
    }
    return () => stop()
  }, [enabled, camera])

  return { result, loading, error, start, stop }
}
