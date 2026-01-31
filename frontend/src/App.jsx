import { useEffect, useState } from 'react'
import { Link, Route, Routes, useLocation } from 'react-router-dom'

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
  { label: 'Product', href: '#' },
  { label: 'Security', href: '#' },
  { label: 'Company', href: '#' },
  { label: 'Docs', href: '#' }
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
              <a
                className="text-slate-200/80 transition hover:text-white"
                href={link.href}
                key={link.label}
              >
                {link.label}
              </a>
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

function PageShell({ pathKey, condensed }) {
  const content = pageMeta[pathKey]

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Header condensed={condensed} />
      <div className="mx-auto max-w-4xl px-6 pb-16 pt-32">
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
      <Route path="/" element={<PageShell pathKey="/" condensed={condensed} />} />
      <Route path="/dashboard" element={<PageShell pathKey="/dashboard" condensed={condensed} />} />
    </Routes>
  )
}
