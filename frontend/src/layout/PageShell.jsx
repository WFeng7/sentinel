import { Link } from 'react-router-dom'

export const pageMeta = {
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

export default function PageShell({ pathKey, condensed, wide = false, children }) {
  const content = pageMeta[pathKey]

  if (wide) {
    return (
      <main className="bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
        <div>{children}</div>
        <div className="fixed top-0 left-0 right-0 z-40 pt-4 md:pt-6">
          <Header condensed={condensed} />
        </div>
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <Header condensed={condensed} />
      <div className="mx-auto max-w-4xl px-8 pb-32 pt-32 md:px-12 md:pt-40">
        {children}
      </div>
    </main>
  )
}
