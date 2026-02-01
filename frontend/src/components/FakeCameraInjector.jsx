import { useEffect } from 'react'

export default function FakeCameraInjector({ streams, setStreams }) {
  useEffect(() => {
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    if (!isLocalhost) return

    const baseUrl = import.meta.env.VITE_DEPLOYMENT_API_URL ?? 'http://localhost:8000'
    const fakeKey = 'fake-2026-01-3015-25-54'
    const fakeCamera = {
      key: fakeKey,
      url: `${baseUrl}/fake-camera/2026-01-3015-25-54.mov`,
      label: 'Fake Camera (2026-01-3015-25-54.mov)',
      lat: 41.823094,
      lng: -71.413391
    }

    setStreams((prev) => {
      if (!Array.isArray(prev)) return [fakeCamera]
      if (prev.some((item) => item?.key === fakeKey)) return prev
      return [...prev, fakeCamera]
    })
  }, [streams, setStreams])

  return null
}
