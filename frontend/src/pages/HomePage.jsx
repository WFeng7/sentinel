import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Globe from 'react-globe.gl'
import ParticleField from '../components/ParticleField.jsx'
import { Highlighter } from '../components/ui/highlighter.jsx'

export default function HomePage() {
  const navigate = useNavigate()
  const globeRef = useRef(null)
  const navigatedRef = useRef(false)
  const [scrollY, setScrollY] = useState(0)
  const [isScrolling, setIsScrolling] = useState(false)
  const scrollTimeoutRef = useRef(null)
  const virtualScrollRef = useRef(0)

  const ROTATE_SPEED_IDLE = 0.35
  const ROTATE_SPEED_SCROLL = 1.8
  const ROTATE_SPEED_LOAD = 0.5
  const scrollPixels = 1200
  const target = useMemo(() => ({ lat: 41.824, lng: -71.4128 }), [])

  const clamp01 = (x) => Math.min(1, Math.max(0, x))
  const easeInOutCubic = (t) =>
    t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

  useEffect(() => {
    const onWheel = (event) => {
      if (window.scrollY > 2) return
      event.preventDefault()

      const delta = event.deltaY
      const next = Math.min(scrollPixels, Math.max(0, virtualScrollRef.current + delta))
      virtualScrollRef.current = next
      setScrollY(next)
      setIsScrolling(true)

      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current)
      scrollTimeoutRef.current = setTimeout(() => setIsScrolling(false), 160)

      if (next >= scrollPixels && !navigatedRef.current) {
        navigatedRef.current = true
        navigate('/dashboard')
      }
    }

    window.addEventListener('wheel', onWheel, { passive: false })
    return () => {
      window.removeEventListener('wheel', onWheel)
    }
  }, [navigate])

  const zoomProgress = useMemo(() => {
    const raw = clamp01(scrollY / scrollPixels)
    return easeInOutCubic(raw)
  }, [scrollY])

  useEffect(() => {
    const globe = globeRef.current
    if (!globe) return

    const controls = globe.controls()
    controls.autoRotate = !isScrolling
    controls.autoRotateSpeed = isScrolling ? ROTATE_SPEED_SCROLL : ROTATE_SPEED_IDLE
    controls.enableZoom = false
    controls.enablePan = false

    const startAltitude = 2.6
    const endAltitude = 0.25
    const altitude = startAltitude + (endAltitude - startAltitude) * zoomProgress
    const spinLng = target.lng + 360 * zoomProgress
    globe.pointOfView({ lat: target.lat, lng: spinLng, altitude }, 0)
  }, [zoomProgress, target, isScrolling])

  return (
    <div className="h-screen w-screen overflow-hidden bg-black">
      <div className="relative h-screen w-screen">
        <div className="absolute inset-0 z-[1]">
          <Globe
            ref={globeRef}
            backgroundColor="#000000"
            globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
            bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
            showAtmosphere
            atmosphereAltitude={0.22}
            atmosphereColor="#7dd3fc"
            onGlobeReady={() => {
              const globe = globeRef.current
              if (!globe) return
              if (typeof globe.renderer === 'function') {
                const renderer = globe.renderer()
                if (renderer?.setPixelRatio) {
                  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
                }
              }
              globe.pointOfView({ lat: target.lat, lng: target.lng, altitude: 2.6 }, 0)
              const controls = globe.controls()
              controls.autoRotate = true
              controls.autoRotateSpeed = ROTATE_SPEED_LOAD
              controls.enableZoom = false
              controls.enablePan = false
            }}
          />
        </div>
        <div className="absolute inset-0 z-[2]">
          <ParticleField />
        </div>
        <div className="pointer-events-none absolute inset-0 z-[3] flex flex-col items-center justify-center px-6 text-center">
          <h2 className="max-w-4xl text-4xl font-light leading-tight tracking-tight text-white md:text-5xl lg:text-6xl">
            What looks like a{" "}
            <Highlighter action="highlight" color="#263265" animationDuration={1100} strokeWidth={2} padding={4}>
              quiet road
            </Highlighter>
            <br />
            <span className="font-montecarlo text-[clamp(3.25rem,6.8vw,8.5rem)]">
              <Highlighter action="underline" color="#f7c948" animationDuration={1200} strokeWidth={3} padding={2}>
                is also a flow of detection
              </Highlighter>
            </span>
            .
          </h2>
          <p className="mt-16 max-w-md text-base text-slate-200 md:text-lg tracking-tight">
            Sentinel monitors Providence&apos;s roads with SoTA computer vision models to alert emergency
            providers about accidents within milliseconds.
          </p>
        </div>
      </div>
    </div>
  )
}
