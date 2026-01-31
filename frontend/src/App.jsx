import { useEffect, useState } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'
import { fetchCameraStreams } from './services/cameras.js'

const pageMeta = {
  '/': {
    title: 'Sentinel | Home',
    eyebrow: 'Sentinel',
    heading: 'Sentinel | Home',
    description:
      'Frontend runs on Vite. Backend runs on FastAPI. Use the root dev script to launch both.'
  },
  '/dashboard': {
    title: 'Sentinel | Dashboard',
    eyebrow: 'Sentinel',
    heading: 'Sentinel | Dashboard',
    description: 'Monitor system activity, review alerts, and track project health in one place.'
  }
}

const navLinks = [
  { label: 'Home', to: '/' },
  { label: 'About', to: '/about' },
  { label: 'Docs', to: '/docs' }
]

function Header({ condensed }) {
  return (
    <header className={`sticky z-30 ${condensed ? 'top-2' : 'top-6'} transition-all duration-300`}>
      <div
        className={`mx-auto flex w-full items-center justify-center px-4 transition-all duration-300 ${
          condensed ? 'max-w-2xl' : 'max-w-6xl'
        }`}
      >
        <div
          className={`flex w-full items-center justify-between rounded-full border border-white/10 bg-slate-800/70 px-4 py-3 text-slate-100 shadow-lg shadow-slate-950/30 backdrop-blur ${
            condensed ? 'gap-4' : 'gap-8'
          }`}
        >
          <Link className="flex items-center gap-3" to="/">
            <span className="flex h-9 w-9 items-center justify-center rounded-full bg-white/90 text-xs font-semibold text-slate-900">
              S
            </span>
            <span className="text-sm font-semibold uppercase tracking-[0.18em]">Sentinel</span>
          </Link>

          <nav
            className={`hidden flex-1 items-center justify-center gap-7 text-sm text-slate-200 transition-all duration-300 md:flex ${
              condensed
                ? 'pointer-events-none -translate-y-2 opacity-0'
                : 'pointer-events-auto translate-y-0 opacity-100'
            }`}
          >
            {navLinks.map((link) => (
              <Link className="text-slate-200/80 transition hover:text-white" key={link.label} to={link.to}>
                {link.label}
              </Link>
            ))}
          </nav>

          <Link
            className="rounded-full border border-white/10 bg-white/90 px-5 py-2 text-sm font-semibold text-slate-900 transition hover:bg-white"
            to="/dashboard"
          >
            Dashboard
          </Link>
        </div>
      </div>
    </header>
  )
}

function HomeContent() {
  return (
    <section className="mt-10 space-y-6">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
        <h2 className="text-lg font-semibold text-white">Placeholder content</h2>
        <p className="mt-2 text-sm text-slate-300">
          This page is intentionally taller now so it scrolls on mobile. Replace these sections with real content
          when ready.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Section</div>
          <h3 className="mt-3 text-base font-semibold text-white">Road monitoring</h3>
          <div className="mt-4 h-40 rounded-xl border border-slate-800 bg-slate-950" />
        </div>
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Section</div>
          <h3 className="mt-3 text-base font-semibold text-white">Alerts</h3>
          <div className="mt-4 h-40 rounded-xl border border-slate-800 bg-slate-950" />
        </div>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8 text-slate-200">
        <h3 className="text-base font-semibold text-white">Extra padding</h3>
        <p className="mt-2 text-sm text-slate-300">
          Spacer content below to ensure consistent scrolling behavior across devices.
        </p>
        <div className="mt-6 h-[60vh] rounded-xl border border-slate-800 bg-slate-950" />
      </div>
    </section>
  )
}

function BasicPage({ title, children }) {
  return (
    <section className="mt-10">
      <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-8">
        <h1 className="text-2xl font-semibold text-white">{title}</h1>
        <div className="mt-4 text-sm text-slate-300">{children}</div>
      </div>
    </section>
  )
}

function DashboardGrid() {
  const [streams, setStreams] = useState([])
  const [error, setError] = useState('')
  const [dragIndex, setDragIndex] = useState(null)
  const orderKey = 'sentinel.cameraOrder.v1'

  useEffect(() => {
    let isMounted = true
    const cacheKey = 'sentinel.cameraStreams.v1'
    const cacheTtlMs = 1000 * 60 * 30

    const getStreamKey = (stream) => stream?.url ?? stream?.stream ?? stream?.src ?? stream

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
      const keyed = new Map(list.map((item) => [getStreamKey(item), item]))
      const ordered = order.map((key) => keyed.get(key)).filter(Boolean)
      const remaining = list.filter((item) => !order.includes(getStreamKey(item)))
      return [...ordered, ...remaining]
    }

    const writeCache = (nextStreams) => {
      try {
        localStorage.setItem(
          cacheKey,
          JSON.stringify({ timestamp: Date.now(), streams: nextStreams })
        )
      } catch {
        // ignore cache writes
      }
    }

    const cachedStreams = readCache()
    const cachedOrder = readOrder()
    if (cachedStreams?.length) {
      setStreams(applyOrder(cachedStreams, cachedOrder))
    }
    setError('')

    if (!cachedStreams?.length) {
      fetchCameraStreams(50, false)
        .then((data) => {
          if (!isMounted) return
          setStreams(applyOrder(data, cachedOrder))
          writeCache(data)
        })
        .catch((err) => {
          if (!isMounted) return
          setError(err?.message ?? 'Failed to load camera streams')
          setStatus('error')
        })
    }

    return () => {
      isMounted = false
    }
  }, [])

  return (
    <section className="h-screen w-screen bg-slate-950">
      {status === 'error' && (
        <div className="absolute left-4 top-4 z-10 rounded border border-rose-500/60 bg-slate-900/90 px-3 py-2 text-xs text-rose-200">
          {error}
        </div>
      )}
      <div className="grid h-full w-full grid-cols-5 grid-rows-10 gap-0">
        {streams.map((stream, index) => (
          <div
            className={`relative border border-slate-600/60 ${dragIndex === index ? 'opacity-60' : ''}`}
            draggable
            key={stream.url ?? stream}
            onDragStart={() => setDragIndex(index)}
            onDragOver={(event) => event.preventDefault()}
            onDrop={() => {
              if (dragIndex === null || dragIndex === index) return
              setStreams((prev) => {
                const next = [...prev]
                const [moved] = next.splice(dragIndex, 1)
                next.splice(index, 0, moved)
                try {
                  const order = next.map((item) => item.url ?? item)
                  localStorage.setItem(orderKey, JSON.stringify(order))
                } catch {
                  // ignore persistence errors
                }
                return next
              })
              setDragIndex(null)
            }}
            onDragEnd={() => setDragIndex(null)}
          >
            <iframe
              allow="autoplay; fullscreen"
              className="absolute inset-0 h-full w-full"
              loading="lazy"
              src={`/player.html?src=${encodeURIComponent(stream.url ?? stream)}`}
              title={stream.label ?? `Camera feed ${index + 1}`}
            />
          </div>
        ))}
      </div>
    </section>
  )
}

function PageShell({ pathKey, condensed, wide = false, children }) {
  const content = pageMeta[pathKey]
  const containerWidth = wide ? 'max-w-none' : 'max-w-4xl'

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Header condensed={condensed} />
      <div className={`mx-auto ${containerWidth} px-6 pb-16 pt-32`}>
        <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-8 shadow-2xl shadow-slate-900/40">
          <p className="text-xs uppercase tracking-[0.3em] text-slate-400">{content.eyebrow}</p>
          <h1 className="mt-4 text-3xl font-semibold text-white">{content.heading}</h1>
          <p className="mt-3 text-slate-300">{content.description}</p>
          <div className="mt-6 flex flex-wrap gap-3 text-sm">
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">
              Vite
            </span>
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">
              React
            </span>
            <span className="rounded-full border border-slate-700 bg-slate-900 px-3 py-1 text-slate-200">
              Tailwind
            </span>
          </div>
        </div>
        {children}
      </div>
    </main>
  )
}

export default function App() {
  const location = useLocation()
  const [condensed, setCondensed] = useState(false)

  useEffect(() => {
    const meta = pageMeta[location.pathname] ?? pageMeta['/']
    document.title = meta.title
  }, [location.pathname])

  useEffect(() => {
    const onScroll = () => setCondensed(window.scrollY > 40)
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return (
    <Routes>
      <Route
        path="/"
        element={
          <PageShell pathKey="/" condensed={condensed}>
            <HomeContent />
          </PageShell>
        }
      />
      <Route
        path="/about"
        element={
          <PageShell pathKey="/" condensed={condensed}>
            <BasicPage title="About">Placeholder About page.</BasicPage>
          </PageShell>
        }
      />
      <Route
        path="/docs"
        element={
          <PageShell pathKey="/" condensed={condensed}>
            <BasicPage title="Docs">Placeholder Docs page.</BasicPage>
          </PageShell>
        }
      />
      <Route
        path="/dashboard"
        element={
          <DashboardGrid />
        }
      />
    </Routes>
  )
}
