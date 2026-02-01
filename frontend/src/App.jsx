import { useEffect, useState } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import PageShell, { pageMeta } from './layout/PageShell.jsx'
import DashboardPage from './pages/DashboardPage.jsx'
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
          <PageShell pathKey="/" condensed={condensed} wide>
            <HomePage />
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
