import { useEffect, useMemo, useState, useRef } from 'react'
import { fetchCameraStreams } from '../services/cameras.js'

export default function DashboardPage() {
  const [streams, setStreams] = useState([])
  const [error, setError] = useState('')
  const [dragIndex, setDragIndex] = useState(null)
  const [hiddenKeys, setHiddenKeys] = useState(new Set())
  const [showHidden, setShowHidden] = useState(false)
  const [refreshToken, setRefreshToken] = useState(0)
  const [expandedKey, setExpandedKey] = useState(null)
  const [expandedVisible, setExpandedVisible] = useState(false)
  const orderKey = 'sentinel.cameraOrder.v1'

  const normalize = (item) => {
    if (!item) return null
    if (typeof item === 'string') return { key: item, url: item, label: '' }
    const url = item.url ?? item.stream ?? item.src ?? item.m3u8
    if (!url) return null
    const key = item.id ?? url
    const label = item.label ?? ''
    return { key, url, label }
  }

  const normalizeList = (arr) => (Array.isArray(arr) ? arr.map(normalize).filter(Boolean) : [])

  useEffect(() => {
    let isMounted = true
    const cacheKey = 'sentinel.cameraStreams.v2'
    const cacheTtlMs = 1000 * 60 * 30

    const readCache = () => {
      try {
        const raw = localStorage.getItem(cacheKey)
        if (!raw) return null
        const parsed = JSON.parse(raw)
        if (!parsed?.streams || !Array.isArray(parsed.streams)) return null
        if (Date.now() - (parsed.timestamp ?? 0) > cacheTtlMs) return null
        return parsed.streams
      } catch {
        return null
      }
    }

    const writeCache = (nextStreams) => {
      try {
        localStorage.setItem(
          cacheKey,
          JSON.stringify({ timestamp: Date.now(), streams: nextStreams })
        )
      } catch {}
    }

    const readOrder = () => {
      try {
        const raw = localStorage.getItem(orderKey)
        if (!raw) return []
        const parsed = JSON.parse(raw)
        return Array.isArray(parsed) ? parsed : []
      } catch {
        return []
      }
    }

    const applyOrder = (list, order) => {
      if (!order.length) return list
      const keyed = new Map(list.map((item) => [item.key, item]))
      const ordered = order.map((k) => keyed.get(k)).filter(Boolean)
      const remaining = list.filter((item) => !order.includes(item.key))
      return [...ordered, ...remaining]
    }

    const cachedStreamsRaw = readCache()
    const cachedOrder = readOrder()
    if (cachedStreamsRaw?.length) {
      const normalized = normalizeList(cachedStreamsRaw)
      setStreams(applyOrder(normalized, cachedOrder))
    }
    setError('')

    const loadStreams = (forceRefresh) => {
      const latestOrder = readOrder()
      fetchCameraStreams(50, forceRefresh)
        .then((data) => {
          if (!isMounted) return
          const payload = data?.cameras ?? data?.streams ?? data
          const normalized = normalizeList(payload)
          setStreams(applyOrder(normalized, latestOrder))
          writeCache(normalized)
        })
        .catch((err) => {
          if (!isMounted) return
          setError(err?.message ?? 'Failed to load camera streams')
        })
    }

    if (!cachedStreamsRaw?.length) {
      loadStreams(false)
    }

    const interval = setInterval(() => {
      loadStreams(true)
      setRefreshToken((prev) => prev + 1)
    }, 1000 * 60 * 5)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [])

  useEffect(() => {
    if (!expandedKey) return
    const raf = requestAnimationFrame(() => setExpandedVisible(true))
    return () => cancelAnimationFrame(raf)
  }, [expandedKey])

  useEffect(() => {
    if (!expandedKey) return
    const onKeyDown = (event) => {
      if (event.key === 'Escape') {
        setExpandedVisible(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [expandedKey])

  useEffect(() => {
    if (!expandedKey) return
    if (expandedVisible) return
    const timeout = setTimeout(() => setExpandedKey(null), 260)
    return () => clearTimeout(timeout)
  }, [expandedKey, expandedVisible])

  const visibleStreams = useMemo(() => {
    return showHidden ? streams : streams.filter((s) => !hiddenKeys.has(s.key))
  }, [streams, showHidden, hiddenKeys])

  const expandedStream = expandedKey ? streams.find((s) => s.key === expandedKey) : null
  const expandedLabel = expandedStream?.label || 'Camera'
  const expandedUrl = expandedStream?.url || ''

  return (
    <section
      className="relative h-screen w-screen bg-slate-950"
      style={{ cursor: 'default' }}
    >
      {expandedKey && (
        <div
          className={`fixed inset-0 z-[2147483647] flex items-center justify-center bg-black/70 transition-opacity duration-300 ${
            expandedVisible ? 'opacity-100' : 'opacity-0'
          }`}
          style={{ cursor: 'default' }}
          onClick={() => setExpandedVisible(false)}
          role="presentation"
        >
          <div
            className={`relative h-[92vh] w-[92vw] max-w-[1800px] origin-center overflow-hidden rounded-md border border-white/10 bg-slate-950 shadow-2xl transition-transform duration-300 ease-out ${
              expandedVisible ? 'scale-100' : 'scale-[0.85]'
            }`}
            style={{ cursor: 'default' }}
            onClick={(event) => event.stopPropagation()}
            role="presentation"
          >
            <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-sm border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100 backdrop-blur">
              {expandedLabel}
            </div>
            <iframe
              allow="autoplay; fullscreen"
              className="absolute inset-0 h-full w-full"
              src={`/player.html?src=${encodeURIComponent(expandedUrl)}&controls=1&t=${refreshToken}`}
              title={expandedLabel}
            />
          </div>
        </div>
      )}

      <div className="absolute left-4 top-4 z-50 rounded-md border border-white/10 bg-slate-900/70 px-3 py-2 text-xs uppercase tracking-[0.16em] text-slate-200 backdrop-blur">
        <label className="flex items-center gap-2">
          <input
            checked={showHidden}
            className="h-4 w-4 accent-slate-200"
            onChange={(event) => setShowHidden(event.target.checked)}
            type="checkbox"
          />
          Show hidden cameras
        </label>
      </div>

      {error && (
        <div className="absolute right-4 top-4 z-50 rounded border border-rose-500/60 bg-slate-900/90 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}

      <div className="grid h-full w-full grid-cols-5 grid-rows-10 gap-2 p-2">
        {visibleStreams.map((stream, index) => {
          const streamKey = stream.key
          const isHidden = hiddenKeys.has(streamKey)
          const label = stream.label || `Camera ${index + 1}`
          const isActiveTile = expandedKey === streamKey

          return (
            <div
              className={`group relative border border-slate-600/60 transition-transform duration-350 ease-out ${
                isActiveTile ? '' : 'hover:z-20 hover:scale-[1.12]'
              } ${dragIndex === index ? 'opacity-60' : ''} ${isHidden ? 'opacity-35' : ''}`}
              draggable
              key={streamKey}
              onDragStart={() => setDragIndex(index)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => {
                if (dragIndex === null || dragIndex === index) return
                setStreams((prev) => {
                  const next = [...prev]
                  const [moved] = next.splice(dragIndex, 1)
                  next.splice(index, 0, moved)
                  try {
                    const order = next.map((item) => item.key)
                    localStorage.setItem(orderKey, JSON.stringify(order))
                  } catch {}
                  return next
                })
                setDragIndex(null)
              }}
              onDragEnd={() => setDragIndex(null)}
            >
              <button
                className="absolute inset-0 z-10"
                onClick={() => {
                  if (expandedKey === streamKey) {
                    setExpandedVisible(false)
                  } else {
                    setExpandedKey(streamKey)
                    setExpandedVisible(false)
                  }
                }}
                type="button"
              />
              <button
                className="absolute left-2 top-2 z-20 hidden h-7 w-7 items-center justify-center rounded-sm border border-white/20 bg-slate-900/80 text-xs font-semibold text-slate-100 opacity-0 transition group-hover:flex group-hover:opacity-100"
                onClick={() => {
                  setHiddenKeys((prev) => {
                    const next = new Set(prev)
                    if (next.has(streamKey)) next.delete(streamKey)
                    else next.add(streamKey)
                    return next
                  })
                }}
                type="button"
              >
                {isHidden ? '+' : '-'}
              </button>
              <div className="pointer-events-none absolute left-2 top-11 z-20 hidden rounded-sm border border-white/10 bg-slate-900/80 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100 opacity-0 transition group-hover:block group-hover:opacity-100">
                {label}
              </div>
              <iframe
                allow="autoplay; fullscreen"
                className={`absolute inset-0 z-0 h-full w-full ${
                  isActiveTile ? 'pointer-events-auto' : 'pointer-events-none'
                }`}
                loading="lazy"
                src={`/player.html?src=${encodeURIComponent(stream.url)}${
                  isActiveTile ? '&controls=1' : '&hideCursor=1'
                }`}
                title={label}
              />
            </div>
          )
        })}
      </div>
    </section>
  )
}
