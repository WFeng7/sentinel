import { useEffect, useState } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import PageShell, { pageMeta } from './layout/PageShell.jsx'
import AboutPage from './pages/AboutPage.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
import DocsPage from './pages/DocsPage.jsx'
import HomePage from './pages/HomePage.jsx'

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
            <HomePage />
          </PageShell>
        }
      />
      <Route
        path="/about"
        element={
          <PageShell pathKey="/" condensed={condensed}>
            <AboutPage />
          </PageShell>
        }
      />
      <Route
        path="/docs"
        element={
          <PageShell pathKey="/" condensed={condensed}>
            <DocsPage />
          </PageShell>
        }
      />
      <Route
        path="/dashboard"
        element={
          <DashboardPage />
        }
      />
    </Routes>
  )
}
