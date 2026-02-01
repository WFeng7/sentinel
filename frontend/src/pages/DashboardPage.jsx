import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import { fetchCameraStreams, fetchHealth, fetchLocationVlm, startMotionFirstWorkers, fetchIncidents } from '../services/cameras.js'
import MarkdownRenderer from '../components/MarkdownRenderer.jsx'
import FakeCameraInjector from '../components/FakeCameraInjector.jsx'

const DASHBOARD_LOADING_MESSAGES = [
  'Warming up camera streams',
  'Balancing live feeds across the grid',
  'Syncing edge frames to the dashboard',
  'Hunting for the cleanest frames',
  'Hunting for the cleanest frames',
  'Stabilizing the live view'
]

const EXPANDED_LOADING_MESSAGES = [
  'Locking onto the live feed',
  'Sharpening the large view',
  'Negotiating stream bitrate',
  'Buffering the high-res window'
]

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
  const [activeLocation, setActiveLocation] = useState(null)
  const [hoveredKey, setHoveredKey] = useState(null)
  const [minimizedSections, setMinimizedSections] = useState({
    filters: true,
    incidents: true,
    vlm: true,
    rag: true,
    map: true
  })
  const [uptime, setUptime] = useState(100)
  const [uptimeSamples, setUptimeSamples] = useState([true])
  const [healthError, setHealthError] = useState('')
  const [streamLoadStates, setStreamLoadStates] = useState({})
  const [dashboardMessageIndex, setDashboardMessageIndex] = useState(0)
  const [expandedMessageIndex, setExpandedMessageIndex] = useState(0)
  const [expandedLoading, setExpandedLoading] = useState(false)
  const [showMap, setShowMap] = useState(false)
  const [vlmResult, setVlmResult] = useState(null)
  const [vlmLoading, setVlmLoading] = useState(false)
  const [vlmError, setVlmError] = useState('')
  const [vlmLastRegenerated, setVlmLastRegenerated] = useState(null)
  const [ragResult, setRagResult] = useState(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragError, setRagError] = useState('')

  // Reset loading states on component mount (window reload)
  useEffect(() => {
    setVlmLoading(false)
    setRagLoading(false)
    setVlmError('')
    setRagError('')
    setExpandedLoading(false)
  }, [])
  const [autoStartWorkers, setAutoStartWorkers] = useState(() => {
    try {
      const raw = localStorage.getItem('sentinel.autoStartWorkers.v1')
      if (raw === null) return true
      return raw === 'true'
    } catch {
      return true
    }
  })

  const mapContainerRef = useRef(null)
  const mapRef = useRef(null)
  const autoStartRef = useRef(false)
  const lastIncidentIdRef = useRef(null)
  const notifAudioRef = useRef(null)
  const suppressedIncidentIdsRef = useRef(new Set())
  const processedIncidentIdsRef = useRef(new Set())

  // keep markers around; don't rebuild on each click
  const markersLayerRef = useRef(null) // L.LayerGroup
  const markersByKeyRef = useRef(new Map()) // key -> L.Marker
  const activeKeyRef = useRef(null)
  const boundsKeyRef = useRef('')

  const orderKey = 'sentinel.cameraOrder.v1'

  const [incidents, setIncidents] = useState([])

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
    const lat = item.lat ?? item.latitude ?? null
    const lng = item.lng ?? item.lon ?? item.longitude ?? null
    return { key, url, label, lat, lng }
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
        const total = normalized.length
        const withCoords = normalized.filter((item) => typeof item.lat === 'number' && typeof item.lng === 'number')
        const missing = total - withCoords.length
        if (missing > 0) {
          console.info(`[map] ${missing}/${total} cameras missing lat/lng (not shown on map).`)
        }
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

    // Always refresh in background, even if we have cache
    loadStreams(false)

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
    try {
      localStorage.setItem('sentinel.autoStartWorkers.v1', String(autoStartWorkers))
    } catch {}
  }, [autoStartWorkers])

  useEffect(() => {
    if (!autoStartWorkers) return
    if (autoStartRef.current) return
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    if (!isLocalhost) return
    autoStartRef.current = true

    const maybeNumber = (value) => (value ? Number(value) : undefined)
    const options = {
      max_workers: maybeNumber(import.meta.env.VITE_MULTI_MAX_WORKERS),
      target_fps: maybeNumber(import.meta.env.VITE_MULTI_TARGET_FPS),
      threshold: maybeNumber(import.meta.env.VITE_MULTI_THRESHOLD),
      window_frames: maybeNumber(import.meta.env.VITE_MULTI_WINDOW_FRAMES),
      incident_rate_limit: maybeNumber(import.meta.env.VITE_MULTI_RATE_LIMIT_S),
      enable_vlm: import.meta.env.VITE_MULTI_ENABLE_VLM === 'true' ? true : undefined,
      enable_rag: import.meta.env.VITE_MULTI_ENABLE_RAG === 'true' ? true : undefined
    }

    startMotionFirstWorkers(options).catch((err) => {
      console.warn('[dashboard] Failed to auto-start motion workers', err)
    })
  }, [autoStartWorkers])

  useEffect(() => {
    let mounted = true

    const poll = async () => {
      try {
        const data = await fetchIncidents(20)
        const list = Array.isArray(data?.incidents) ? data.incidents : []
        if (mounted) setIncidents(list)
      } catch {}
    }

    poll()
    const interval = setInterval(poll, 5000)

    return () => {
      mounted = false
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
        setUptimeSamples((prev) => [...prev, true].slice(-5))
      } catch (err) {
        if (!mounted) return
        setHealthError(err?.message ?? 'Health check failed')
        setUptimeSamples((prev) => [...prev, false].slice(-5))
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
    if (!expandedKey) {
      setExpandedLoading(false)
      setExpandedMessageIndex(0)
      return
    }
    setExpandedLoading(true)
    setExpandedMessageIndex(0)
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

  useEffect(() => {
    if (!searchResults.length) return
    setStreamLoadStates((prev) => {
      const next = { ...prev }
      let changed = false
      for (const stream of searchResults) {
        if (!(stream.key in next)) {
          next[stream.key] = false
          changed = true
        }
      }
      return changed ? next : prev
    })
  }, [searchResults])

  const visibleCount = searchResults.length
  const loadedCount = searchResults.filter((stream) => streamLoadStates[stream.key]).length
  const dashboardLoading = visibleCount > 0 && loadedCount !== visibleCount

  useEffect(() => {
    if (!dashboardLoading) {
      setDashboardMessageIndex(0)
      return
    }
    const timer = setInterval(() => {
      setDashboardMessageIndex((prev) => (prev + 1) % DASHBOARD_LOADING_MESSAGES.length)
    }, 2600)
    return () => clearInterval(timer)
  }, [dashboardLoading])

  useEffect(() => {
    if (!expandedLoading) {
      setExpandedMessageIndex(0)
      return
    }
    const timer = setInterval(() => {
      setExpandedMessageIndex((prev) => (prev + 1) % EXPANDED_LOADING_MESSAGES.length)
    }, 2400)
    return () => clearInterval(timer)
  }, [expandedLoading])

  const handleOpenStream = (stream) => {
    setControlPanelOpen(true)
    setExpandedKey(stream.key)
    setExpandedVisible(false)
  }

  const toString = (value) => {
    if (value === null || value === undefined) return ''
    return typeof value === 'string' ? value : String(value)
  }

  const mapVlmToRagInput = (vlmPayload) => {
    const detailed = vlmPayload?.detailed_analysis ?? vlmPayload?.result ?? vlmPayload ?? {}
    const ev = detailed?.event ?? null
    const ragInfo = detailed?.rag ?? null
    const evidence = detailed?.evidence ?? []

    const eventTypeCandidates = []
    if (ev?.type) eventTypeCandidates.push(toString(ev.type))
    if (ev?.category) eventTypeCandidates.push(toString(ev.category?.value ?? ev.category))
    if (Array.isArray(ragInfo?.tags)) eventTypeCandidates.push(...ragInfo.tags.slice(0, 5).map(toString))

    const signals = []
    if (Array.isArray(evidence)) {
      evidence.forEach((item) => {
        if (!item) return
        if (typeof item === 'object') {
          if (item.claim) signals.push(toString(item.claim).slice(0, 200))
          if (Array.isArray(item.signals)) signals.push(...item.signals.slice(0, 3).map(toString))
        } else {
          signals.push(toString(item).slice(0, 200))
        }
      })
    }
    if (Array.isArray(ragInfo?.queries)) signals.push(...ragInfo.queries.slice(0, 3).map(toString))

    return {
      event_type_candidates: eventTypeCandidates,
      signals: signals.slice(0, 10),
      city: 'Providence'
    }
  }

  const formatRelative = (timestamp) => {
    const ts = new Date(timestamp)
    if (Number.isNaN(ts.getTime())) return ''
    const diffSeconds = Math.max(0, Math.floor((Date.now() - ts.getTime()) / 1000))
    if (diffSeconds < 60) return `${diffSeconds}s ago`
    const minutes = Math.floor(diffSeconds / 60)
    if (minutes < 60) return `${minutes}m ago`
    const hours = Math.floor(minutes / 60)
    if (hours < 24) return `${hours}h ago`
    const days = Math.floor(hours / 24)
    return `${days}d ago`
  }

  const scoreToStatus = (score) => {
    const numeric = typeof score === 'number' ? score : Number(score)
    if (numeric >= 6) return { status: 'Critical', statusColor: 'bg-metadata-4' }
    if (numeric >= 4) return { status: 'Alert', statusColor: 'bg-metadata-2' }
    return { status: 'Info', statusColor: 'bg-metadata-3' }
  }

  const buildIncidentRow = (incident) => {
    const events = incident?.events
    const eventList = events && typeof events === 'object' ? Object.keys(events) : []
    const label = incident?.label || incident?.camera_id || 'Camera'
    const title = eventList.length ? `${label} — ${eventList.join(', ')}` : `${label} incident`
    const { status, statusColor } = scoreToStatus(incident?.score)
    const timestamp = incident?.timestamp || ''
    const relative = formatRelative(timestamp) || 'just now'
    return {
      id: incident?.id || `${incident?.camera_id}-${timestamp}`,
      title,
      status,
      statusColor,
      relative,
      timestamp,
      camera_id: incident?.camera_id,
      label: incident?.label,
      events: incident?.events,
      score: incident?.score
    }
  }

  const incidentRows = useMemo(() => {
    const suppressed = suppressedIncidentIdsRef.current
    const rows = []
    const bucketLatest = new Map()
    const windowMs = 5 * 60 * 1000

    for (const incident of incidents) {
      if (suppressed.has(incident?.id)) continue
      const row = buildIncidentRow(incident)
      const events = row.events && typeof row.events === 'object' ? Object.keys(row.events).sort().join(',') : ''
      const bucketKey = `${row.camera_id || ''}|${events}`
      const ts = new Date(row.timestamp).getTime()
      if (Number.isNaN(ts)) {
        rows.push(row)
        continue
      }
      const existing = bucketLatest.get(bucketKey)
      if (!existing) {
        bucketLatest.set(bucketKey, { row, ts })
        continue
      }
      if (Math.abs(ts - existing.ts) <= windowMs) {
        if (ts > existing.ts) {
          bucketLatest.set(bucketKey, { row, ts })
        }
      } else {
        rows.push(existing.row)
        bucketLatest.set(bucketKey, { row, ts })
      }
    }

    for (const { row } of bucketLatest.values()) {
      rows.push(row)
    }

    const seen = new Set()
    return rows.filter((row) => {
      if (seen.has(row.id)) return false
      seen.add(row.id)
      return true
    })
  }, [incidents])

  useEffect(() => {
    if (!incidents.length) return
    if (dashboardLoading) return
    const newest = incidents[0]
    if (!newest?.id) return
    if (processedIncidentIdsRef.current.has(newest.id)) return
    processedIncidentIdsRef.current.add(newest.id)

    const cameraId = newest.camera_id
    const stream = streams.find((item) => item?.key === cameraId)
    if (!stream?.url) {
      const row = buildIncidentRow(newest)
      lastIncidentIdRef.current = newest.id
      setActiveIncident(row)
      setControlPanelOpen(true)
      if (typeof notifAudioRef.current?.play === 'function') {
        notifAudioRef.current.currentTime = 0
        notifAudioRef.current.play().catch(() => {})
      }
      return
    }

    const label = stream.label || newest.label || `Camera ${cameraId}`
    fetchLocationVlm({
      cameraId,
      label,
      streamUrl: stream.url
    })
      .then((data) => {
        const detailed = data?.detailed_analysis ?? data?.result ?? data ?? {}
        const severity = detailed?.event?.severity?.value ?? detailed?.event?.severity ?? ''
        if (String(severity).toLowerCase() === 'none') {
          suppressedIncidentIdsRef.current.add(newest.id)
          return
        }
        const row = buildIncidentRow(newest)
        lastIncidentIdRef.current = newest.id
        setActiveIncident(row)
        setControlPanelOpen(true)
        if (typeof notifAudioRef.current?.play === 'function') {
          notifAudioRef.current.currentTime = 0
          notifAudioRef.current.play().catch(() => {})
        }
      })
      .catch(() => {
        const row = buildIncidentRow(newest)
        lastIncidentIdRef.current = newest.id
        setActiveIncident(row)
        setControlPanelOpen(true)
        if (typeof notifAudioRef.current?.play === 'function') {
          notifAudioRef.current.currentTime = 0
          notifAudioRef.current.play().catch(() => {})
        }
      })
  }, [incidents, streams, dashboardLoading])

  const runRAGFromVLM = async (vlmPayload) => {
    if (!vlmPayload) return
    setRagLoading(true)
    setRagError('')
    setRagResult(null)
    try {
      const ragInput = mapVlmToRagInput(vlmPayload)
      const res = await fetch(`${import.meta.env.VITE_DEPLOYMENT_API_URL ?? 'http://localhost:8000'}/rag/decide`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(ragInput)
      })
      if (!res.ok) throw new Error('Request failed')
      const data = await res.json()
      setRagResult(data)
    } catch (e) {
      setRagError(e.message || 'RAG failed')
    } finally {
      setRagLoading(false)
    }
  }

  const runVLM = async (stream, label) => {
    if (!stream) return
    const safeLabel = label || stream.label || `Camera ${stream.key}`
    setVlmLoading(true)
    setVlmError('')
    setVlmResult(null)
    setRagLoading(true)
    setRagResult(null)
    setRagError('')
    try {
      const data = await fetchLocationVlm({
        cameraId: stream.key,
        label: safeLabel,
        streamUrl: stream.url
      })
      setVlmResult(data)
      setVlmLastRegenerated(Date.now())
      await runRAGFromVLM(data)
    } catch (e) {
      setVlmError(e.message || 'VLM failed')
    } finally {
      setVlmLoading(false)
    }
  }

  const regenerateVLM = async () => {
    console.log('[Regenerate] Starting regeneration...')
    if (!activeLocation || !streams) {
      console.log('[Regenerate] No active location or streams')
      return
    }
    
    // Find the stream for the active location
    const stream = streams.find(s => s.key === activeLocation)
    if (!stream) {
      console.log('[Regenerate] No stream found for active location')
      return
    }
    
    console.log('[Regenerate] Found stream:', stream.key)
    
    // Just call the same runVLM function but with a "regenerate" flag
    const safeLabel = stream.label || `Camera ${stream.key}`
    setVlmLoading(true)
    setVlmError('')
    setVlmResult(null)
    setRagLoading(true)
    setRagResult(null)
    setRagError('')
    
    // Add timeout to prevent getting stuck
    const timeoutId = setTimeout(() => {
      setVlmLoading(false)
      setRagLoading(false)
      setVlmError('VLM regeneration timed out. Please try again.')
    }, 15000) // 15 second timeout
    
    try {
      console.log('[Regenerate] Calling fetchLocationVlm...')
      // Use the same endpoint but with a flag for faster processing
      const data = await fetchLocationVlm({
        cameraId: stream.key,
        label: safeLabel,
        streamUrl: stream.url,
        regenerate: true  // Add flag for faster processing
      })
      console.log('[Regenerate] Got VLM data:', data)
      clearTimeout(timeoutId)
      setVlmResult(data)
      setVlmLastRegenerated(Date.now())
      await runRAGFromVLM(data)
    } catch (e) {
      console.error('[Regenerate] Error:', e)
      clearTimeout(timeoutId)
      setVlmError(e.message || 'VLM regeneration failed')
    } finally {
      clearTimeout(timeoutId)
      setVlmLoading(false)
      setRagLoading(false)
      console.log('[Regenerate] Finished regeneration')
    }
  }

  const toggleMinimized = (section) => {
    setMinimizedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const handleCameraSelect = async (stream, label) => {
    setActiveLocation({ key: stream.key, label, url: stream.url })
    await runVLM(stream, label)
  }

  // Create map ONCE (dark tiles). Keep container mounted; don't destroy on hide.
  useEffect(() => {
    if (!mapContainerRef.current) return
    if (mapRef.current) return

    const mapInstance = L.map(mapContainerRef.current, {
      center: [41.823094, -71.413391],
      zoom: 20,
      zoomControl: true,
      scrollWheelZoom: true
    })

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }).addTo(mapInstance)

    markersLayerRef.current = L.layerGroup().addTo(mapInstance)
    mapRef.current = mapInstance

    return () => {
      try {
        markersByKeyRef.current.clear()
      } catch {}
      try {
        mapInstance.remove()
      } catch {}
      mapRef.current = null
      markersLayerRef.current = null
      activeKeyRef.current = null
    }
  }, [])

  // Incremental marker sync: build/update/remove without full rebuilds
  useEffect(() => {
    if (!mapRef.current || !markersLayerRef.current) return

    const layer = markersLayerRef.current
    const byKey = markersByKeyRef.current

    // flat pink pins (#f27fa5), no glow
    const baseStyle =
      'width:16px;height:16px;border-radius:9999px;background:#f27fa5;border:2px solid rgba(255,255,255,0.95);'
    const activeStyle =
      'width:22px;height:22px;border-radius:9999px;background:#f27fa5;border:2px solid rgba(255,255,255,1);'

    const nextKeys = new Set()

    const coordCounts = new Map()
    const jitterStep = 0.000018
    const boundsPoints = []

    visibleStreams.forEach((stream, index) => {
      if (typeof stream.lat !== 'number' || typeof stream.lng !== 'number') return

      const key = stream.key
      nextKeys.add(key)

      const label = stream.label || `Camera ${index + 1}`
      const isActive = (activeLocation?.key ?? null) === key

      const coordKey = `${stream.lat.toFixed(5)},${stream.lng.toFixed(5)}`
      const seenCount = coordCounts.get(coordKey) ?? 0
      coordCounts.set(coordKey, seenCount + 1)
      const angle = seenCount * 1.1
      const jitter = seenCount === 0 ? 0 : jitterStep * seenCount
      const lat = stream.lat + Math.cos(angle) * jitter
      const lng = stream.lng + Math.sin(angle) * jitter
      boundsPoints.push([lat, lng])

      let marker = byKey.get(key)
      if (!marker) {
        const icon = L.divIcon({
          className: '',
          html: `<div data-pin="1" style="${isActive ? activeStyle : baseStyle}"></div>`,
          iconSize: [22, 22],
          iconAnchor: [11, 11]
        })

        marker = L.marker([lat, lng], { icon })

        marker.on('click', () => {
          handleOpenStream(stream)
          handleCameraSelect(stream, label)
        })
        marker.on('mouseover', () => setHoveredKey(key))
        marker.on('mouseout', () => setHoveredKey(null))

        marker.addTo(layer)
        byKey.set(key, marker)
        return
      }

      const ll = marker.getLatLng()
      if (ll.lat !== lat || ll.lng !== lng) {
        marker.setLatLng([lat, lng])
      }
    })

    for (const [key, marker] of byKey.entries()) {
      if (!nextKeys.has(key)) {
        try {
          marker.remove()
        } catch {}
        byKey.delete(key)
      }
    }

    if (boundsPoints.length > 0) {
      const boundsKey = boundsPoints
        .map(([lat, lng]) => `${lat.toFixed(5)},${lng.toFixed(5)}`)
        .join('|')
      if (boundsKeyRef.current !== boundsKey) {
        boundsKeyRef.current = boundsKey
        const bounds = L.latLngBounds(boundsPoints)
        mapRef.current.fitBounds(bounds, {
          paddingTopLeft: [220, 40],
          paddingBottomRight: [20, 20],
          maxZoom: 15
        })
      }
    }
  }, [visibleStreams]) // IMPORTANT: NOT dependent on activeLocation

  // Update only the active marker style (fast)
  useEffect(() => {
    if (!markersLayerRef.current) return

    const byKey = markersByKeyRef.current
    const prevKey = activeKeyRef.current
    const nextKey = activeLocation?.key ?? null

    if (prevKey === nextKey) return

    const baseStyle =
      'width:16px;height:16px;border-radius:9999px;background:#f27fa5;border:2px solid rgba(255,255,255,0.95);'
    const activeStyle =
      'width:22px;height:22px;border-radius:9999px;background:#f27fa5;border:2px solid rgba(255,255,255,1);'

    const setMarkerStyle = (key, style) => {
      const marker = byKey.get(key)
      if (!marker) return
      const icon = L.divIcon({
        className: '',
        html: `<div data-pin="1" style="${style}"></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11]
      })
      marker.setIcon(icon)
    }

    if (prevKey) setMarkerStyle(prevKey, baseStyle)
    if (nextKey) setMarkerStyle(nextKey, activeStyle)
    activeKeyRef.current = nextKey
  }, [activeLocation])

  useEffect(() => {
    if (!mapRef.current) return
    const mapInstance = mapRef.current
    const timer = setTimeout(() => {
      mapInstance.invalidateSize()
    }, 250)
    return () => clearTimeout(timer)
  }, [showMap, controlPanelOpen])

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
    <section className="relative h-screen w-screen overflow-hidden bg-slate-950" style={{ cursor: 'default' }}>
      <audio ref={notifAudioRef} src="/notif.mp3" preload="auto" />
      <FakeCameraInjector streams={streams} setStreams={setStreams} />
      <style>{`
        iframe {
          scrollbar-width: none;
          -ms-overflow-style: none;
        }
        iframe::-webkit-scrollbar {
          display: none;
        }
        .sentinel-spinner {
          width: var(--spinner-size, 34px);
          height: var(--spinner-size, 34px);
          border-radius: 9999px;
          border: 2px solid rgba(148, 163, 184, 0.3);
          border-top-color: rgba(248, 250, 252, 0.9);
          animation: sentinel-spin 0.9s linear infinite;
        }
        .sentinel-sheen {
          position: relative;
          overflow: hidden;
        }
        .sentinel-sheen::after {
          content: '';
          position: absolute;
          top: -60%;
          bottom: -60%;
          left: -60%;
          width: 60%;
          background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.12), transparent);
          animation: sentinel-sheen 2s ease-in-out infinite;
        }
        @keyframes sentinel-spin {
          to { transform: rotate(360deg); }
        }
        @keyframes sentinel-sheen {
          0% { transform: translateX(-120%); }
          100% { transform: translateX(220%); }
        }
      `}</style>

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
            className={`relative h-[92vh] w-[96%] max-w-[1800px] origin-center overflow-hidden border border-white/10 bg-slate-950 shadow-2xl transition-transform duration-300 ease-out ${
              expandedVisible ? 'scale-100' : 'scale-[0.85]'
            } ${controlPanelOpen ? '' : 'w-[92vw]'}`}
            style={{ cursor: 'default' }}
            onClick={(event) => event.stopPropagation()}
            role="presentation"
          >
            <div className="pointer-events-none absolute left-4 top-4 z-10 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100 backdrop-blur">
              {expandedLabel}
            </div>
            <button
              className="absolute right-4 top-4 z-10 flex items-center justify-center border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-100 backdrop-blur transition hover:bg-slate-800"
              onClick={() => setExpandedVisible(false)}
              type="button"
              aria-label="Close expanded camera"
            >
              X
            </button>
            {expandedLoading && (
              <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-slate-950/80">
                <div className="sentinel-sheen flex flex-col items-center gap-3 border border-white/10 bg-slate-900/80 px-6 py-5 text-slate-100 shadow-2xl">
                  <div className="sentinel-spinner" />
                  <div className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Loading live view</div>
                  <div className="text-sm text-slate-200">{EXPANDED_LOADING_MESSAGES[expandedMessageIndex]}</div>
                </div>
              </div>
            )}
            <iframe
              allow="autoplay; fullscreen"
              className="absolute inset-0 h-full w-full"
              src={`/player.html?src=${encodeURIComponent(expandedUrl)}&controls=1&t=${refreshToken}`}
              title={expandedLabel}
              onLoad={() => setExpandedLoading(false)}
            />
            {null}
          </div>
        </div>
      )}

      {error && (
        <div className="absolute right-4 top-4 z-50 rounded border border-rose-500/60 bg-slate-900/90 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}

      {dashboardLoading && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-950/70">
          <div className="sentinel-sheen flex flex-col items-center gap-3 border border-white/10 bg-slate-900/80 px-6 py-5 text-slate-100 shadow-2xl">
            <div className="sentinel-spinner" />
            <div className="text-[11px] uppercase tracking-[0.3em] text-slate-400">Loading dashboard</div>
            <div className="text-sm text-slate-200">{DASHBOARD_LOADING_MESSAGES[dashboardMessageIndex]}</div>
            <div className="text-[10px] uppercase tracking-[0.24em] text-slate-500">
              {loadedCount} of {visibleCount} feeds live
            </div>
          </div>
        </div>
      )}

      {!dashboardLoading && searchResults.length === 0 && !error && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-slate-950/70">
          <div className="flex flex-col items-center gap-3 border border-white/10 bg-slate-900/80 px-6 py-5 text-slate-100 shadow-2xl">
            <div className="text-[11px] uppercase tracking-[0.3em] text-slate-400">No cameras found</div>
            <div className="text-sm text-slate-200">Check your camera sources or refresh</div>
            <button
              className="mt-2 border border-white/10 bg-slate-800 px-4 py-2 text-xs text-slate-100 transition hover:bg-slate-700"
              onClick={() => loadStreams(true)}
              type="button"
            >
              Refresh Cameras
            </button>
          </div>
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

      <div className="flex h-full w-full overflow-hidden">
        <aside
          className={`relative h-full overflow-y-auto border-r border-white/10 bg-slate-950/80 text-slate-100 transition-all duration-300 ${
            controlPanelOpen ? 'w-1/3 opacity-100' : 'w-0 opacity-0 pointer-events-none'
          }`}
        >
          {activeIncident && controlPanelOpen && (
            <div className="absolute inset-0 z-30 flex items-start justify-center bg-slate-950/80 p-6">
              <div className="relative w-full border border-white/10 bg-slate-900/90 p-5 text-slate-100 shadow-2xl">
                <div className="pointer-events-none inline-flex items-center gap-2 border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100">
                  <span className={`h-2 w-2 rounded-full ${activeIncident.statusColor}`} />
                  {activeIncident.status}
                </div>
                <button
                  className="absolute right-4 top-4 flex items-center justify-center border border-white/10 bg-slate-900/70 px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-100 backdrop-blur transition hover:bg-slate-800"
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
                <div className="mt-4 border border-white/10 bg-slate-950/70 p-3 text-xs text-slate-300">
                  Review the related camera footage and acknowledge the incident once verified.
                </div>
              </div>
            </div>
          )}

          <div className="flex h-full flex-col gap-5 p-6">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Control Panel</div>

            <div className="space-y-3 border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-200">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                <span>Filters</span>
                <div className="flex items-center gap-2">
                  <button
                    className="inline-flex h-6 items-center border border-white/10 bg-slate-900/70 px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-100 transition hover:bg-slate-800"
                    onClick={() => loadStreams(true)}
                    type="button"
                  >
                    Refresh URLs
                  </button>
                  <button
                    className="flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
                    onClick={() => toggleMinimized('filters')}
                    type="button"
                    aria-label="Toggle filters"
                  >
                    {minimizedSections.filters ? '+' : '-'}
                  </button>
                </div>
              </div>
              
              {!minimizedSections.filters && (
                <div className="space-y-3">
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
                      checked={autoStartWorkers}
                      className="h-4 w-4 accent-slate-200"
                      onChange={(event) => {
                        autoStartRef.current = false
                        setAutoStartWorkers(event.target.checked)
                      }}
                      type="checkbox"
                    />
                    Auto-start motion workers (localhost)
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
                      className="w-full border border-white/10 bg-slate-950/80 px-3 py-2 text-sm text-slate-100 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-slate-600"
                      type="search"
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-3 border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-200">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                <span>Recent Incidents</span>
                <button
                  className="flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
                  onClick={() => toggleMinimized('incidents')}
                  type="button"
                  aria-label="Toggle incidents"
                >
                  {minimizedSections.incidents ? '+' : '-'}
                </button>
              </div>
              
              {!minimizedSections.incidents && (
                <div className="space-y-2">
                  <div className="flex items-start justify-between gap-4">
                    <div>
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
                  </div>
                  <div className="space-y-2 text-xs text-slate-300">
                    {incidentRows.length ? (
                      incidentRows.map((incident) => (
                        <button
                          key={incident.id}
                          className="flex w-full items-start justify-between gap-3 border border-white/10 bg-slate-950/60 px-3 py-2 text-left transition hover:border-white/20 hover:bg-slate-900/70"
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
                            <span className="text-[10px] uppercase tracking-[0.2em] text-slate-500">{incident.relative}</span>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="border border-white/10 bg-slate-950/50 px-3 py-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                        No incidents yet
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {activeLocation && (vlmLoading || vlmResult || vlmError) && (
              <div className="border border-white/10 bg-slate-900/60 p-4 text-xs text-slate-200">
                <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                  <div className="flex items-center gap-2">
                    <span>VLM Insight</span>
                    {vlmLastRegenerated && (
                      <span className="text-[10px] text-slate-500">
                        Regenerated {new Date(vlmLastRegenerated).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      className={`flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold transition ${
                        vlmLoading 
                          ? 'text-slate-400 cursor-not-allowed' 
                          : 'text-slate-200 hover:bg-slate-800'
                      }`}
                      onClick={() => regenerateVLM()}
                      disabled={vlmLoading}
                      type="button"
                      aria-label={vlmLoading ? "Regenerating VLM analysis..." : "Regenerate VLM analysis"}
                      title={vlmLoading ? "Regenerating VLM analysis..." : "Regenerate VLM analysis"}
                    >
                      ↻
                    </button>
                    <button
                      className="flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
                      onClick={() => toggleMinimized('vlm')}
                      type="button"
                      aria-label="Toggle VLM insight"
                    >
                      {minimizedSections.vlm ? '+' : '-'}
                    </button>
                  </div>
                </div>
                {!minimizedSections.vlm && (
                  <>
                    {vlmLoading && (
                      <div className="mt-2 flex items-center gap-2 text-slate-300">
                        <div className="sentinel-spinner" style={{ '--spinner-size': '16px' }} />
                        <span>Regenerating VLM analysis with fresh frames…</span>
                      </div>
                    )}
                    {vlmError && !vlmLoading && <div className="mt-2 text-rose-300">{vlmError}</div>}
                    {vlmResult && !vlmLoading && (
                      <div className="mt-2 max-h-64 overflow-y-auto space-y-3 text-slate-200">
                        {vlmResult.summary && (
                          <div className="space-y-2">
                            <MarkdownRenderer
                              content={vlmResult.summary}
                              className="text-sm leading-relaxed"
                            />
                          </div>
                        )}
                        {vlmResult.detailed_analysis && (
                          <div className="space-y-2 border-t border-white/10 pt-2">
                            {vlmResult.detailed_analysis.event && (
                              <div className="flex flex-wrap items-center gap-2 text-xs">
                                <span className="px-2 py-1 rounded bg-slate-800 text-slate-300 font-medium">
                                  {vlmResult.detailed_analysis.event.type}
                                </span>
                                <span className={`px-2 py-1 rounded font-medium ${
                                  vlmResult.detailed_analysis.event.severity === 'critical' ? 'bg-red-900/60 text-red-200' :
                                  vlmResult.detailed_analysis.event.severity === 'high' ? 'bg-orange-900/60 text-orange-200' :
                                  vlmResult.detailed_analysis.event.severity === 'medium' ? 'bg-yellow-900/60 text-yellow-200' :
                                  vlmResult.detailed_analysis.event.severity === 'low' ? 'bg-blue-900/60 text-blue-200' :
                                  'bg-slate-800 text-slate-400'
                                }`}>
                                  {vlmResult.detailed_analysis.event.severity}
                                </span>
                                <span className="text-slate-400">
                                  {Math.round((vlmResult.detailed_analysis.event.confidence || 0) * 100)}% confidence
                                </span>
                              </div>
                            )}
                            {vlmResult.detailed_analysis.actors && vlmResult.detailed_analysis.actors.length > 0 && (
                              <div className="text-xs text-slate-400">
                                <span className="font-medium text-slate-300">Actors:</span> {vlmResult.detailed_analysis.actors.length} detected
                                <div className="mt-1 flex flex-wrap gap-1">
                                  {vlmResult.detailed_analysis.actors.slice(0, 4).map((actor, idx) => (
                                    <span key={idx} className="px-2 py-0.5 bg-slate-800 rounded text-slate-300">
                                      {actor.class || 'unknown'} {actor.track_id ? `(#${actor.track_id})` : ''}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {activeLocation && (ragLoading || ragResult || ragError || vlmLoading) && (
              <div className="border border-white/10 bg-slate-900/60 p-4 text-xs text-slate-200">
                <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                  <span>RAG Decision</span>
                  <button
                    className="flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
                    onClick={() => toggleMinimized('rag')}
                    type="button"
                    aria-label="Toggle RAG decision"
                  >
                    {minimizedSections.rag ? '+' : '-'}
                  </button>
                </div>
                {!minimizedSections.rag && (
                  <div className="mt-2 max-h-64 overflow-y-auto">
                    {ragLoading && (
                      <div className="flex items-center gap-2 text-slate-300">
                        <div className="sentinel-spinner" style={{ '--spinner-size': '16px' }} />
                        <span>Running policy checks…</span>
                      </div>
                    )}
                    {ragError && !ragLoading && <div className="text-rose-300">{ragError}</div>}
                    {ragResult && !ragLoading && (
                      <>
                        <div className="font-semibold mb-1">Decision</div>
                        <div className="mb-2 text-slate-300">
                          {ragResult.explanation || ragResult.supporting_excerpts?.[0]?.text || 'No decision'}
                        </div>
                        {ragResult.action && (
                          <div className="text-slate-400">Action: {ragResult.action}</div>
                        )}
                        {Array.isArray(ragResult.supporting_excerpts) && ragResult.supporting_excerpts.length > 0 && (
                          <div className="mt-3 border-t border-white/10 pt-2 text-slate-300">
                            <div className="font-semibold mb-1">Sources</div>
                            <div className="space-y-1 text-xs">
                              {ragResult.supporting_excerpts.slice(0, 4).map((excerpt) => (
                                <div key={excerpt.document_id}>
                                  <span className="text-slate-400">{excerpt.document_id}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                )}
              </div>
            )}

            <div className="space-y-3 border border-white/10 bg-slate-900/60 p-4 text-sm text-slate-200">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                <span>Show map</span>
                <button
                  className="flex h-6 w-6 items-center justify-center border border-white/10 bg-slate-900/70 text-[11px] font-semibold text-slate-200 transition hover:bg-slate-800"
                  onClick={() => toggleMinimized('map')}
                  type="button"
                  aria-label="Toggle map"
                >
                  {minimizedSections.map ? '+' : '-'}
                </button>
              </div>

              {/* keep container mounted so map stays alive */}
              {!minimizedSections.map && (
                <div
                  className="border border-white/10 bg-slate-950/80 transition-all duration-200"
                >
                  <div
                    ref={mapContainerRef}
                    className="aspect-square overflow-hidden border border-white/10"
                    style={{ background: '#0b1020' }}
                  /> 
                </div>
              )}
            </div>
            <div className="border border-white/10 bg-slate-900/60 p-4 text-xs text-slate-200">
              <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.2em] text-slate-400">
                <span>Session uptime</span>
              </div>
              <div className="mt-2 flex items-center justify-between gap-2">
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
                <span className="mt-2 block text-[10px] uppercase tracking-[0.16em] text-rose-300">{healthError}</span>
              )}
            </div>
          </div>
        </aside>

        <div className={`relative h-full overflow-hidden transition-all duration-300 ${controlPanelOpen ? 'w-2/3' : 'w-full'}`}>
          <div
            className="grid h-full w-full gap-0 overflow-hidden"
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
              const isHoveredTile = hoveredKey === streamKey

              return (
                <div
                  className={`group relative overflow-hidden border border-slate-600/60 transition-transform duration-350 ease-out ${
                    isActiveTile ? '' : 'hover:z-20 hover:scale-[1.12]'
                  } ${isHoveredTile ? 'z-20 scale-[1.12] shadow-[0_0_0_2px_rgba(248,250,252,0.35)]' : ''} ${
                    dragIndex === index ? 'opacity-60' : ''
                  } ${isHidden ? 'opacity-35' : ''}`}
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
                        handleCameraSelect(stream, label)
                      }
                    }}
                    type="button"
                  />
                  <button
                    className="absolute left-2 top-2 z-20 hidden px-2 py-1 items-center justify-center border border-white/20 bg-slate-900/80 text-[10px] font-semibold text-slate-100 opacity-0 transition group-hover:flex group-hover:opacity-100"
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

                  {/* Labels: NO border, NO hover behavior */}
                  {showLabels && (
                    <div className="pointer-events-none absolute bottom-2 left-2 right-2 z-20 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-slate-100">
                      <span className="block truncate">{label}</span>
                    </div>
                  )}

                  <iframe
                    allow="autoplay; fullscreen"
                    className={`absolute inset-0 z-0 h-full w-full ${isActiveTile ? 'pointer-events-auto' : 'pointer-events-none'}`}
                    loading="lazy"
                    src={`/player.html?src=${encodeURIComponent(stream.url)}${
                      isActiveTile ? '&controls=1' : '&hideCursor=1'
                    }`}
                    title={label}
                    onLoad={() =>
                      setStreamLoadStates((prev) => ({
                        ...prev,
                        [streamKey]: true
                      }))
                    }
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
