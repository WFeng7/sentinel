import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { fetchCameraStreams, fetchHealth } from '../services/cameras.js'

export default function DashboardPage() {
  const navigate = useNavigate()
  const [streams, setStreams] = useState([])
  const [error, setError] = useState('')
  const [dragIndex, setDragIndex] = useState(null)
  const [hiddenKeys, setHiddenKeys] = useState(new Set())
  const [showHidden, setShowHidden] = useState(false)
  const [refreshToken, setRefreshToken] = useState(0)
  const [expandedKey, setExpandedKey] = useState(null)
  const [expandedVisible, setExpandedVisible] = useState(false)
  const [controlPanelOpen, setControlPanelOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showLabels, setShowLabels] = useState(true)
  const [activeIncident, setActiveIncident] = useState(null)
  const [uptime, setUptime] = useState(100)
  const [uptimeSamples, setUptimeSamples] = useState([true])
  const [healthError, setHealthError] = useState('')
  const orderKey = 'sentinel.cameraOrder.v1'

  const incidents = [
    {
      id: 'incident-1',
      title: 'Motion spike detected',
      status: 'Critical',
      statusColor: 'bg-metadata-4',
      relative: '2m ago',
      timestamp: '2026-01-31 14:22'
    },
    {
      id: 'incident-2',
      title: 'Access door left open',
      status: 'Alert',
      statusColor: 'bg-metadata-2',
      relative: '11m ago',
      timestamp: '2026-01-31 14:13'
    },
    {
      id: 'incident-3',
      title: 'Shift change logged',
      status: 'Info',
      statusColor: 'bg-metadata-3',
      relative: '28m ago',
      timestamp: '2026-01-31 13:56'
    }
  ]

  const expandedStream = expandedKey ? streams.find((s) => s.key === expandedKey) : null
  const expandedLabel = expandedStream?.label || 'Camera'
  const expandedUrl = expandedStream?.url || ''

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
      localStorage.setItem(cacheKey, JSON.stringify({ timestamp: Date.now(), streams: nextStreams }))
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

  const loadStreams = (forceRefresh) => {
    const latestOrder = readOrder()
    setError('')
    if (forceRefresh) {
      try {
        localStorage.removeItem(cacheKey)
      } catch {}
    }
    fetchCameraStreams(50, forceRefresh)
      .then((data) => {
        const payload = data?.cameras ?? data?.streams ?? data
        const normalized = normalizeList(payload)
        setStreams(applyOrder(normalized, latestOrder))
        writeCache(normalized)
      })
      .catch((err) => {
        setError(err?.message ?? 'Failed to load camera streams')
      })
  }

  useEffect(() => {
    let isMounted = true

    const cachedStreamsRaw = readCache()
    const cachedOrder = readOrder()
    if (cachedStreamsRaw?.length) {
      const normalized = normalizeList(cachedStreamsRaw)
      if (isMounted) {
        setStreams(applyOrder(normalized, cachedOrder))
      }
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
    let mounted = true

    const poll = async () => {
      try {
        await fetchHealth()
        if (!mounted) return
        setHealthError('')
        setUptimeSamples((prev) => [...prev, true].slice(-30))
      } catch (err) {
        if (!mounted) return
        setHealthError(err?.message ?? 'Health check failed')
        setUptimeSamples((prev) => [...prev, false].slice(-30))
      }
    }

    poll()
    const sampleInterval = setInterval(poll, 10000)

    return () => {
      mounted = false
      clearInterval(sampleInterval)
    }
  }, [])

  useEffect(() => {
    if (!uptimeSamples.length) return
    const successes = uptimeSamples.filter(Boolean).length
    setUptime(Math.round((successes / uptimeSamples.length) * 1000) / 10)
  }, [uptimeSamples])

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

  const searchResults = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) return visibleStreams
    return visibleStreams.filter((stream) => {
      const label = stream.label || ''
      return label.toLowerCase().includes(query)
    })
  }, [visibleStreams, searchQuery])

  const cameraGridColumns = useMemo(() => {
    const total = searchResults.length
    if (total <= 1) return 1
    if (total <= 2) return 2
    if (total <= 4) return 2
    if (total <= 6) return 3
    if (total <= 9) return 3
    if (total <= 12) return 4
    return 5
  }, [searchResults.length])

  return (
    <section
      className="relative h-screen w-screen bg-slate-950"
      style={{ cursor: 'default' }}
    >
      {expandedKey && (
        <div
          className={`fixed top-0 bottom-0 right-0 z-[2147483647] flex items-center justify-center bg-black/70 transition-opacity duration-300 ${
            controlPanelOpen ? 'left-1/3' : 'left-0'
          } ${expandedVisible ? 'opacity-100' : 'opacity-0'}`}
          style={{ cursor: 'default' }}
          onClick={() => setExpandedVisible(false)}
          role="presentation"
        >
          <div
            className={`relative h-[92vh] w-[96%] max-w-[1800px] origin-center overflow-hidden rounded-md border border-white/10 bg-slate-950 shadow-2xl transition-transform duration-300 ease-out ${
              expandedVisible ? 'scale-100' : 'scale-[0.85]'
            } ${controlPanelOpen ? '' : 'w-[92vw]'}`}
            style={{ cursor: 'default' }}
            onClick={(event) => event.stopPropagation()}
            role="presentation"
          >
            <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-sm border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100 backdrop-blur">
              {expandedLabel}
            </div>
            <button
              className="absolute right-4 top-4 z-10 flex items-center justify-center rounded-sm border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-100 backdrop-blur transition hover:bg-slate-800"
              onClick={() => setExpandedVisible(false)}
              type="button"
              aria-label="Close expanded camera"
            >
              X
            </button>
            <iframe
              allow="autoplay; fullscreen"
              className="absolute inset-0 h-full w-full"
              src={`/player.html?src=${encodeURIComponent(expandedUrl)}&controls=1&t=${refreshToken}`}
              title={expandedLabel}
            />
          </div>
        </div>
      )}

      {error && (
        <div className="absolute right-4 top-4 z-50 rounded border border-rose-500/60 bg-slate-900/90 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}

      <div className="absolute bottom-4 left-4 z-50 flex items-center gap-2">
        <button
          className="flex h-12 w-12 items-center justify-center border border-white/10 bg-slate-900/80 text-xl text-slate-100 shadow-lg transition hover:bg-slate-800"
          onClick={() => setControlPanelOpen((prev) => !prev)}
          type="button"
          aria-label="Toggle control panel"
        >
          {controlPanelOpen ? '←' : '→'}
        </button>
        {controlPanelOpen && (
          <button
            className="flex h-12 w-12 items-center justify-center border border-white/10 bg-slate-900/80 text-xl text-slate-100 shadow-lg transition hover:bg-slate-800"
            onClick={() => navigate('/')}
            type="button"
            aria-label="Return home"
          >
            🏠
          </button>
        )}
      </div>

      <div className="flex h-full w-full">
        <aside
          className={`relative h-full overflow-hidden border-r border-white/10 bg-slate-950/80 text-slate-100 transition-all duration-300 ${
            controlPanelOpen ? 'w-1/3 opacity-100' : 'w-0 opacity-0 pointer-events-none'
          }`}
        >
          {activeIncident && controlPanelOpen && (
            <div className="absolute inset-0 z-30 flex items-start justify-center bg-slate-950/80 p-6">
              <div className="relative w-full rounded-2xl border border-white/10 bg-slate-900/90 p-5 text-slate-100 shadow-2xl">
                <div className="pointer-events-none inline-flex items-center gap-2 rounded-sm border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100">
                  <span className={`h-2 w-2 rounded-full ${activeIncident.statusColor}`} />
                  {activeIncident.status}
                </div>
                <button
                  className="absolute right-4 top-4 flex items-center justify-center rounded-sm border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-100 backdrop-blur transition hover:bg-slate-800"
                  onClick={() => setActiveIncident(null)}
                  type="button"
                  aria-label="Close incident"
                >
                  X
                </button>
                <h3 className="mt-4 text-lg font-semibold text-white">{activeIncident.title}</h3>
                <p className="mt-2 text-sm text-slate-300">
                  Observed at {activeIncident.timestamp}. ({activeIncident.relative})
                </p>
                <div className="mt-4 rounded-lg border border-white/10 bg-slate-950/70 p-3 text-xs text-slate-300">
                  Review the related camera footage and acknowledge the incident once verified.
                </div>
              </div>
            </div>
          )}
            <div className="flex h-full flex-col gap-5 p-6">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Control Panel</div>

            <div className="space-y-3 rounded-md border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-200">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                <span>Filters</span>
                <button
                  className="border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:bg-slate-800"
                  onClick={() => loadStreams(true)}
                  type="button"
                >
                  Refresh URLs
                </button>
              </div>
              <label className="flex items-center gap-2 text-xs text-slate-200">
                <input
                  checked={showHidden}
                  className="h-4 w-4 accent-slate-200"
                  onChange={(event) => setShowHidden(event.target.checked)}
                  type="checkbox"
                />
                Show hidden cameras
              </label>
              <label className="flex items-center gap-2 text-xs text-slate-200">
                <input
                  checked={showLabels}
                  className="h-4 w-4 accent-slate-200"
                  onChange={(event) => setShowLabels(event.target.checked)}
                  type="checkbox"
                />
                Show camera labels
              </label>
              <div className="space-y-2">
                <span className="text-xs uppercase tracking-[0.2em] text-slate-400">Search by label</span>
                <input
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  placeholder="Filter cameras"
                  className="w-full rounded-md border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-600"
                  type="search"
                />
              </div>
            </div>

            <div className="space-y-3 rounded-md border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-200">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Recent Incidents</span>
                  <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <span className="h-2 w-2 bg-metadata-4" />
                      Critical
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="h-2 w-2 bg-metadata-2" />
                      Alert
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="h-2 w-2 bg-metadata-3" />
                      Info
                    </span>
                  </div>
                </div>
                <div className="flex max-w-[220px] flex-col items-end gap-1 border border-white/10 bg-slate-900/80 px-3 py-2 text-xs text-slate-100">
                  <span className="text-[10px] uppercase tracking-[0.2em] text-slate-400">Session uptime</span>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-emerald-200">{uptime.toFixed(1)}%</span>
                    <div className="flex items-center gap-1">
                      {uptimeSamples.map((ok, index) => (
                        <span
                          key={`${index}-${ok ? 'up' : 'down'}`}
                          className={`h-2 w-2 ${ok ? 'bg-emerald-400' : 'bg-rose-400'}`}
                        />
                      ))}
                    </div>
                  </div>
                  {healthError && (
                    <span className="text-[10px] uppercase tracking-[0.16em] text-rose-300">{healthError}</span>
                  )}
                </div>
              </div>
              <div className="space-y-2 text-xs text-slate-300">
                {incidents.map((incident) => (
                  <button
                    key={incident.id}
                    className="flex w-full items-start justify-between gap-3 rounded-md border border-white/10 bg-slate-950/60 px-3 py-2 text-left transition hover:border-white/20 hover:bg-slate-900/70"
                    onClick={() => setActiveIncident(incident)}
                    type="button"
                  >
                    <div className="space-y-1">
                      <span className="block text-sm font-semibold text-slate-100">{incident.title}</span>
                      <span className="block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                        {incident.timestamp}
                      </span>
                    </div>
                    <div className="flex flex-col items-end gap-1 text-[11px] text-slate-400">
                      <span className="flex items-center gap-2">
                        <span className={`h-2 w-2 rounded-full ${incident.statusColor}`} />
                        {incident.status}
                      </span>
                      <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">
                        {incident.relative}
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </aside>

        <div
          className={`relative h-full transition-all duration-300 ${
            controlPanelOpen ? 'w-2/3' : 'w-full'
          }`}
        >
          <div
            className="grid h-full w-full gap-0"
            style={{
              gridTemplateColumns: `repeat(${cameraGridColumns}, minmax(0, 1fr))`,
              gridAutoRows: 'minmax(0, 1fr)'
            }}
          >
            {searchResults.map((stream, index) => {
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
                  <div
                    className={`pointer-events-none absolute bottom-2 left-2 right-2 z-20 rounded-sm border border-white/10 bg-slate-900/80 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100 transition ${
                      showLabels ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                    } ${showLabels ? '' : 'hidden group-hover:block'}`}
                  >
                    <span className="block truncate">{label}</span>
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
        </div>
      </div>
    </section>
  )
}
