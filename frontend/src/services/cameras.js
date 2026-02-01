export const API_BASE = import.meta.env.VITE_DEPLOYMENT_API_URL ?? 'http://localhost:8000'

export async function startTracking(streamUrl) {
  const response = await fetch(`${API_BASE}/tracking/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ stream_url: streamUrl })
  })
  if (!response.ok) {
    throw new Error('Failed to start tracking')
  }
  return response.json()
}

export async function stopTracking() {
  const response = await fetch(`${API_BASE}/tracking/stop`, { method: 'POST' })
  if (!response.ok) {
    throw new Error('Failed to stop tracking')
  }
  return response.json()
}

export async function fetchTrackingStatus() {
  const response = await fetch(`${API_BASE}/tracking/status`)
  if (!response.ok) {
    throw new Error('Failed to fetch tracking status')
  }
  return response.json()
}

export async function fetchHealth() {
  const response = await fetch(`${API_BASE}/health`)
  if (!response.ok) {
    throw new Error('Health check failed')
  }
  return response.json()
}

export async function startMotionFirstWorkers(options = {}) {
  const url = new URL(`${API_BASE}/workers/motion-first/start`)
  Object.entries(options).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value))
    }
  })
  const response = await fetch(url.toString(), { method: 'POST' })
  if (!response.ok) {
    throw new Error('Failed to start motion-first workers')
  }
  return response.json()
}

function normalizeCameraEntry(entry, index) {
  if (typeof entry === 'string') {
    return { url: entry, label: `Camera ${index + 1}` }
  }

  if (entry && typeof entry === 'object') {
    return {
      url: entry.url ?? entry.stream ?? entry.src ?? '',
      label: entry.label ?? entry.name ?? entry.location ?? `Camera ${index + 1}`,
      lat: entry.lat ?? entry.latitude ?? null,
      lng: entry.lng ?? entry.lon ?? entry.longitude ?? null
    }
  }

  return { url: '', label: `Camera ${index + 1}` }
}

export async function fetchCameraStreams(limit = 50, refresh = false) {
  const url = new URL(`${API_BASE}/cameras`)
  url.searchParams.set('limit', String(limit))
  if (refresh) {
    url.searchParams.set('refresh', 'true')
  }
  const response = await fetch(url.toString())
  if (!response.ok) {
    throw new Error('Failed to load camera streams')
  }
  const data = await response.json()
  const list = Array.isArray(data.cameras) ? data.cameras : Array.isArray(data.streams) ? data.streams : []
  return list.map((entry, index) => normalizeCameraEntry(entry, index)).filter((entry) => entry.url)
}

export async function fetchLocationVlm({ cameraId, label, streamUrl }) {
  const response = await fetch(`${API_BASE}/vlm/location`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_id: cameraId,
      label,
      stream_url: streamUrl
    })
  })
  if (!response.ok) {
    throw new Error('Failed to fetch VLM analysis')
  }
  return response.json()
}

export async function fetchIncidents(limit = 20) {
  const url = new URL(`${API_BASE}/incidents`)
  url.searchParams.set('limit', String(limit))
  const response = await fetch(url.toString())
  if (!response.ok) {
    throw new Error('Failed to fetch incidents')
  }
  return response.json()
}
