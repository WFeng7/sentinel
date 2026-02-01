import { useRef, useState, useEffect, useMemo } from 'react'
import Globe from 'react-globe.gl'
import ParticleField from '../components/ParticleField.jsx'
import { Highlighter } from '../components/ui/highlighter.jsx'

export default function HomePage() {
  const productDemoRef = useRef(null)

  const ROTATE_SPEED_IDLE = -1.15
  const ROTATE_SPEED_SCROLL = 1.8
  const ROTATE_SPEED_LOAD = -0.9

  const [animationProgress, setAnimationProgress] = useState(0)
  const [footerProgress, setFooterProgress] = useState(0)
  const [isLocked, setIsLocked] = useState(false)

  const lockedScrollY = useRef(null)
  const unlockOffset = useRef(0)
  const progressRef = useRef(0)
  const globeRef = useRef(null)

  const [scrollY, setScrollY] = useState(0)
  const [isScrolling, setIsScrolling] = useState(false)
  const scrollTimeoutRef = useRef(null)
  const virtualScrollRef = useRef(0)
  const [isPinned, setIsPinned] = useState(false)
  const zoomHoldRef = useRef(0)

  const [showComments, setShowComments] = useState(true)
  const [dismissedComments, setDismissedComments] = useState(() => new Set())

  const hideComment = (id) => {
    setDismissedComments((prev) => {
      const next = new Set(prev)
      next.add(id)
      return next
    })
  }

  const restoreComments = () => {
    setDismissedComments(new Set())
  }

  const ANIMATION_COMPLETE = 0.92
  const WHEEL_SPEED = 0.0007
  const scrollPixels = 1200

  const target = useMemo(() => ({ lat: 41.824, lng: -71.4128 }), [])

  const clamp01 = (x) => Math.min(1, Math.max(0, x))
  const easeInOutCubic = (t) =>
    t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2

  useEffect(() => {
    progressRef.current = animationProgress
  }, [animationProgress])

  useEffect(() => {
    const onScroll = () => setScrollY(window.scrollY || 0)

    const onWheel = (event) => {
      const atTop = window.scrollY < 8
      const shouldLock = atTop && virtualScrollRef.current < scrollPixels
      if (!shouldLock) return

      event.preventDefault()

      const delta = event.deltaY
      const next = Math.min(scrollPixels, Math.max(0, virtualScrollRef.current + delta))
      virtualScrollRef.current = next
      setScrollY(next)

      setIsScrolling(true)
      if (next >= scrollPixels) {
        setIsPinned(true)
        zoomHoldRef.current = scrollPixels
      } else if (next <= 0) {
        setIsPinned(false)
        zoomHoldRef.current = 0
      }

      if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current)
      scrollTimeoutRef.current = setTimeout(() => setIsScrolling(false), 160)
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('wheel', onWheel, { passive: false })
    onScroll()

    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('wheel', onWheel)
    }
  }, [isPinned])

  const zoomProgress = useMemo(() => {
    const raw = clamp01(scrollY / scrollPixels)
    return easeInOutCubic(raw)
  }, [scrollY])

  const zoomProgressEffective = useMemo(() => {
    if (isPinned && window.scrollY > 8) {
      return easeInOutCubic(zoomHoldRef.current / scrollPixels)
    }
    return zoomProgress
  }, [zoomProgress, isPinned])

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
    const altitude = startAltitude + (endAltitude - startAltitude) * zoomProgressEffective
    const spinLng = target.lng + 360 * zoomProgressEffective

    globe.pointOfView({ lat: target.lat, lng: spinLng, altitude }, 0)
  }, [zoomProgressEffective, target, isScrolling])

  useEffect(() => {
    const section = productDemoRef.current
    if (!section) return

    const handleScroll = () => {
      const rect = section.getBoundingClientRect()
      const viewportHeight = window.innerHeight
      const shouldLock = rect.top <= 0 && rect.bottom > 0 && progressRef.current < ANIMATION_COMPLETE

      if (shouldLock && !isLocked) {
        setIsLocked(true)
        setAnimationProgress(0)
        setFooterProgress(0)
        return
      }

      if (isLocked) return

      if (rect.bottom <= viewportHeight && rect.bottom >= 0) {
        setFooterProgress(1 - rect.bottom / viewportHeight)
      } else if (rect.bottom < 0) {
        setFooterProgress(1)
      } else {
        setFooterProgress(0)
      }
    }

    const handleWheel = (e) => {
      if (!isLocked) return

      e.preventDefault()
      const delta = e.deltaY * WHEEL_SPEED
      const nextProgress = Math.min(1, Math.max(0, progressRef.current + delta))
      const scrollingUp = e.deltaY < 0
      const wouldComplete = nextProgress >= ANIMATION_COMPLETE
      const wouldExitUp = nextProgress <= 0.05

      if (scrollingUp && wouldExitUp) {
        setIsLocked(false)
        unlockOffset.current = 0
        setAnimationProgress(0)
        return
      }

      if (!scrollingUp && wouldComplete) {
        setIsLocked(false)
        setAnimationProgress(ANIMATION_COMPLETE)
        unlockOffset.current = 220
        return
      }

      setAnimationProgress(nextProgress)
    }

    handleScroll()
    window.addEventListener('scroll', handleScroll, { passive: true })
    window.addEventListener('wheel', handleWheel, { passive: false })

    return () => {
      window.removeEventListener('scroll', handleScroll)
      window.removeEventListener('wheel', handleWheel)
    }
  }, [isLocked, animationProgress])

  useEffect(() => {
    if (!isLocked) return
    const scrollY = window.scrollY
    lockedScrollY.current = scrollY

    const original = {
      position: document.body.style.position,
      top: document.body.style.top,
      width: document.body.style.width,
      overflow: document.body.style.overflow,
    }

    document.body.style.position = 'fixed'
    document.body.style.top = `-${scrollY}px`
    document.body.style.width = '100%'
    document.body.style.overflow = 'hidden'

    return () => {
      document.body.style.position = original.position
      document.body.style.top = original.top
      document.body.style.width = original.width
      document.body.style.overflow = original.overflow
      if (lockedScrollY.current !== null) {
        const nextY = lockedScrollY.current + (unlockOffset.current || 0)
        window.scrollTo(0, nextY)
      }
      lockedScrollY.current = null
      unlockOffset.current = 0
    }
  }, [isLocked])

  return (
    <div className="w-screen min-h-screen overflow-x-hidden bg-black text-slate-200">
      <div className="relative w-full min-h-screen flex-1 bg-black overflow-hidden">
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

        <div className="pointer-events-none absolute inset-0 z-[3] bg-[radial-gradient(ellipse_at_center,rgba(0,0,0,0)_40%,rgba(0,0,0,0.65)_100%)]" />

        <div className="absolute inset-0 z-10 flex flex-col justify-between pointer-events-none">
          <div className="flex justify-between items-start p-8 md:p-12" />
          <div className="absolute inset-x-0 top-[75%] flex -translate-y-1/2 justify-center px-6 text-center">
            <p className="max-w-md text-base text-slate-200 md:text-lg tracking-tight">
              Sentinel monitors Providence&apos;s roads with SoTA computer vision models to alert
              emergency providers about accidents within milliseconds.
            </p>
          </div>
          <div className="px-8 pb-32 md:px-16 md:pb-40 text-center">
            <h2 className="mx-auto max-w-4xl text-4xl font-light leading-tight tracking-tight text-white md:text-5xl lg:text-6xl">
              What looks like a{' '}
              <Highlighter action="highlight" color="#263265" animationDuration={1100} strokeWidth={2} padding={4}>
                quiet road
              </Highlighter>
              <br />
              <span className="font-montecarlo text-5xl text-emerald-200 md:text-6xl lg:text-7xl">
                <Highlighter action="underline" color="#f7c948" animationDuration={1200} strokeWidth={3} padding={2}>
                  is also a flow of detection
                </Highlighter>
                .
              </span>
            </h2>
          </div>
          <div className="px-8 pb-8 md:px-16" />
        </div>
      </div>

      <section ref={productDemoRef} className="relative w-full min-h-[100vh] bg-black py-10 md:py-14">
        <div className="absolute inset-0">
          <ParticleField />
        </div>

        <div className="pointer-events-auto absolute left-6 top-6 z-50 flex items-center gap-2">
          <button
            className="rounded-md border border-slate-600/70 bg-black/60 px-3 py-2 text-xs text-slate-100 hover:bg-black/80"
            onClick={() => setShowComments((v) => !v)}
          >
            {showComments ? 'Hide comments' : 'Show comments'}
          </button>
          <button
            className="rounded-md border border-slate-600/70 bg-black/60 px-3 py-2 text-xs text-slate-100 hover:bg-black/80"
            onClick={restoreComments}
          >
            Restore all
          </button>
        </div>

        <div className="sticky top-0 left-0 flex h-screen w-full items-center justify-center transition-opacity duration-700">
          <div
            className="relative w-[86vw] max-w-6xl overflow-hidden rounded-2xl border border-slate-500/60 bg-black/20 shadow-[0_0_40px_rgba(0,0,0,0.45)]"
            style={{ aspectRatio: '16 / 9', opacity: 1 - footerProgress }}
          >
            <div
              className="absolute inset-0 bg-center bg-cover bg-no-repeat"
              style={{ backgroundImage: "url('/images/home.jpg')" }}
            />
            <div className="absolute inset-0 bg-gradient-to-br from-black/35 via-transparent to-black/55" />

            <div
              className="absolute border-2 border-metadata-4 rounded-lg bg-metadata-4/15 transition-opacity duration-300"
              style={{
                left: '44%',
                top: '20%',
                width: '22%',
                height: '40%',
                opacity: (() => {
                  const p = animationProgress
                  if (p < 0.06) return 0
                  if (p < 0.18) return (p - 0.06) / 0.12
                  if (p < 0.42) return 1
                  if (p < 0.58) return 1 - (p - 0.42) / 0.16
                  return 0
                })(),
              }}
            >
              <span className="absolute -top-10 left-0 text-sm font-semibold text-red-400">
                Emergency response
              </span>
            </div>

            <div
              className="absolute border-2 border-metadata-2 rounded-lg bg-metadata-2/15 transition-opacity duration-300"
              style={{
                left: '36%',
                top: '54%',
                width: '32%',
                height: '38%',
                opacity: (() => {
                  const p = animationProgress
                  if (p < 0.42) return 0
                  if (p < 0.58) return (p - 0.42) / 0.16
                  if (p < 0.75) return 1
                  if (p < 0.92) return 1 - (p - 0.75) / 0.17
                  return 0
                })(),
              }}
            >
              <span className="absolute -top-10 left-0 text-sm font-semibold text-amber-400">
                Incident / stopped vehicle
              </span>
            </div>
          </div>

          {showComments && (
            <div className="pointer-events-none absolute inset-0 z-40">
              {[
                { id: 'a', side: 'right', top: '18%', start: 0.08, end: 0.28 },
                { id: 'b', side: 'left', top: '36%', start: 0.28, end: 0.48 },
                { id: 'c', side: 'right', top: '58%', start: 0.48, end: 0.68 },
                { id: 'd', side: 'left', top: '76%', start: 0.68, end: 0.88 },
              ]
                .filter((card) => !dismissedComments.has(card.id))
                .map((card) => {
                  const progress = Math.min(
                    1,
                    Math.max(0, (animationProgress - card.start) / (card.end - card.start))
                  )
                  const opacity = progress
                  const offset = (1 - progress) * 30
                  const isRight = card.side === 'right'

                  return (
                    <div
                      key={card.id}
                      className={`absolute z-40 hidden items-center md:flex ${isRight ? 'right-[4vw]' : 'left-[4vw]'}`}
                      style={{
                        top: card.top,
                        opacity,
                        transform: `translateX(${isRight ? offset : -offset}px)`,
                      }}
                    >
                      {isRight && (
                        <div className="h-0 w-0 border-y-[10px] border-y-transparent border-r-[14px] border-r-slate-300/80" />
                      )}
                      <div className="pointer-events-auto relative flex items-center gap-3 rounded-md border border-slate-400/60 bg-slate-900/70 px-4 py-3 text-xs uppercase tracking-[0.24em] text-slate-100 shadow-lg">
                        <span>Alert cue</span>
                        <span className="h-1 w-6 bg-slate-200/80" />
                        <button
                          onClick={() => hideComment(card.id)}
                          className="absolute -top-2 -right-2 h-5 w-5 rounded-full bg-black text-xs text-slate-200 hover:bg-slate-800"
                        >
                          ×
                        </button>
                      </div>
                      {!isRight && (
                        <div className="h-0 w-0 border-y-[10px] border-y-transparent border-l-[14px] border-l-slate-300/80" />
                      )}
                    </div>
                  )
                })}
            </div>
          )}
        </div>
      </section>

      <footer className="h-0 overflow-hidden bg-black" />
    </div>
  )
}
