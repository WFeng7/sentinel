import { useRef, useState, useEffect } from 'react'
import ParticleField from '../components/ParticleField.jsx'

export default function HomePage() {
  const productDemoRef = useRef(null)
  const [animationProgress, setAnimationProgress] = useState(0)
  const [footerProgress, setFooterProgress] = useState(0)
  const [isLocked, setIsLocked] = useState(false)
  const lockedScrollY = useRef(null)
  const unlockOffset = useRef(0)
  const progressRef = useRef(0)

  // Animation thresholds (unlock when yellow box has faded) - spaced out for longer animation
  const ANIMATION_COMPLETE = 0.92
  const WHEEL_SPEED = 0.0007

  useEffect(() => {
    progressRef.current = animationProgress
  }, [animationProgress])

  useEffect(() => {
    const section = productDemoRef.current
    if (!section) return

    const handleScroll = () => {
      const rect = section.getBoundingClientRect()
      const viewportHeight = window.innerHeight
      
      // Check if we should lock: section top at/above viewport top and animation not complete
      const shouldLock = rect.top <= 0 && rect.bottom > 0 && progressRef.current < ANIMATION_COMPLETE
      
      if (shouldLock && !isLocked) {
        setIsLocked(true)
        setAnimationProgress(0)
        setFooterProgress(0)
        return
      }
      
      if (isLocked) return
      
      // Footer fade: when demo section bottom enters viewport from below, start transition
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
      
      // Scrolling up: when animation would reverse to start, unlock and let scroll through
      // so user can return to hero (avoids blank/stuck state)
      if (scrollingUp && wouldExitUp) {
        setIsLocked(false)
        unlockOffset.current = 0
        setAnimationProgress(0)
        return
      }
      
      // Scrolling down: when animation completes, unlock and give immediate scroll
      // so there's no friction when leaving the section
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
    <div className="w-screen min-h-screen flex flex-col gap-8 overflow-x-hidden overflow-y-auto pb-16 text-slate-200 md:gap-12 md:pb-24">
      {/* Hero Section - image background + particles + overlay */}
      <div className="relative w-full min-h-screen flex-1 bg-[#050505] overflow-hidden">
        <div
          className="absolute inset-0 z-0 bg-cover bg-center"
          style={{ backgroundImage: "url('/images/home.jpg')" }}
        />
        <div className="absolute inset-0 z-[1] bg-gradient-to-b from-slate-950/50 via-slate-950/30 to-slate-950/70" />
        <div className="absolute inset-0 z-[2]">
          <ParticleField />
        </div>
        {/* Overlay - text only */}
        <div className="absolute inset-0 z-10 flex flex-col justify-between pointer-events-none">
          <div className="flex justify-between items-start p-8 md:p-12">
            <div className="text-xs uppercase tracking-[0.4em] text-slate-400">Sentinel</div>
          </div>
          <div className="px-8 pb-32 md:px-16 md:pb-40">
            <h2 className="max-w-3xl text-4xl font-light leading-tight text-white md:text-5xl lg:text-6xl">
              What looks like a quiet road
              <br />
              <span className="text-slate-400">is also a flow of detection.</span>
            </h2>
            <p className="mt-8 text-lg text-slate-400 max-w-xl">
              We build technology to surface incidents from live camera feeds in real time.
            </p>
          </div>
          <div className="px-8 pb-8 md:px-16">
            <p className="text-xs uppercase tracking-[0.3em] text-slate-500">Scroll to discover</p>
          </div>
        </div>
      </div>

      {/* Product Demo Section - fixed highway background + scroll-triggered highlights */}
      <section
        ref={productDemoRef}
        className="relative w-full min-h-[100vh] bg-slate-950"
      >
        {/* Fixed background image (stays in place while scrolling) */}
        <div
          className="sticky top-0 left-0 w-full h-screen bg-center bg-cover bg-no-repeat transition-opacity duration-700 pointer-events-none"
          style={{
            backgroundImage: "url('/images/product-demo-bg.png')",
            backgroundAttachment: 'fixed',
            opacity: 1 - footerProgress,
          }}
        >
          {/* Highlight 1: Firetruck - appears first, then fades as yellow appears */}
          <div
            className="absolute border-4 border-red-500 rounded-lg bg-red-500/10 transition-opacity duration-300"
            style={{
              left: '49%',
              top: '27%',
              width: '12%',
              height: '30%',
              opacity: (() => {
                const p = animationProgress
                // Stage 1: Red box appears (0.06 -> 0.18)
                if (p < 0.06) return 0
                if (p < 0.18) return (p - 0.06) / 0.12
                // Stage 2: Red box stays visible (0.18 -> 0.42)
                if (p < 0.42) return 1
                // Stage 3: Red box fades as yellow appears (0.42 -> 0.58)
                if (p < 0.58) return 1 - ((p - 0.42) / 0.16)
                return 0
              })(),
            }}
          >
            <span className="absolute -top-10 left-0 text-sm font-semibold text-red-400 drop-shadow-lg">
              Emergency response
            </span>
          </div>
          {/* Highlight 2: Stopped car - appears as red fades */}
          <div
            className="absolute border-4 border-amber-400 rounded-lg bg-amber-400/10 transition-opacity duration-300"
            style={{
              left: '44%',
              top: '59%',
              width: '18%',
              height: '35%',
              opacity: (() => {
                const p = animationProgress
                // Stage 1: Yellow box appears as red fades (0.42 -> 0.58)
                if (p < 0.42) return 0
                if (p < 0.58) return (p - 0.42) / 0.16
                // Stage 2: Yellow box stays visible (0.58 -> 0.75)
                if (p < 0.75) return 1
                // Stage 3: Yellow box fades out (0.75 -> 0.92)
                if (p < 0.92) return 1 - ((p - 0.75) / 0.17)
                return 0
              })(),
            }}
          >
            <span className="absolute -top-10 left-0 text-sm font-semibold text-amber-400 drop-shadow-lg">
              Incident / stopped vehicle
            </span>
          </div>
        </div>
      </section>

      {/* Footer - Walkthrough (fades in as demo fades out) */}
      <footer
        className="relative w-full min-h-screen flex flex-col items-center justify-center text-center bg-slate-950 py-20 md:py-32 transition-opacity duration-700"
        style={{ opacity: footerProgress }}
      >
        <div className="max-w-2xl mx-auto px-8">
          <div className="text-xs uppercase tracking-[0.25em] text-slate-500">Book a walkthrough</div>
          <h3 className="mt-6 text-3xl font-semibold text-white md:text-4xl leading-tight">
            See your network in real time
          </h3>
          <p className="mt-8 text-base leading-relaxed text-slate-300 md:text-lg">
            We will map your live feeds, configure alert policies, and surface critical events within days.
          </p>
          <div className="mt-12 flex flex-wrap justify-center gap-4">
            <a
              href="#"
              className="rounded-full bg-white text-slate-900 px-8 py-4 text-sm font-semibold hover:bg-slate-100 transition-colors"
            >
              Book a demo
            </a>
            <a
              href="#"
              className="rounded-full border border-slate-600 text-slate-300 px-8 py-4 text-sm font-semibold hover:border-slate-500 hover:text-white transition-colors"
            >
              Talk to sales
            </a>
          </div>
        </div>
      </footer>
    </div>
  )
}
