const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

function normalizeCameraEntry(entry, index) {
  if (typeof entry === 'string') {
    return { url: entry, label: `Camera ${index + 1}` }
  }

  if (entry && typeof entry === 'object') {
    return {
      url: entry.url ?? entry.stream ?? entry.src ?? '',
      label: entry.label ?? entry.name ?? entry.location ?? `Camera ${index + 1}`
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
