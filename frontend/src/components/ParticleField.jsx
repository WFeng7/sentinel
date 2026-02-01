import { useEffect, useRef } from 'react'

export default function ParticleField({ className = '' }) {
  const canvasRef = useRef(null)
  const animationRef = useRef(0)
  const particlesRef = useRef([])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const spawnParticles = (count) => {
      const next = []
      for (let i = 0; i < count; i += 1) {
        next.push({
          x: Math.random(),
          y: Math.random(),
          radius: 0.8 + Math.random() * 2.5,
          speed: 0.04 + Math.random() * 0.2,
          drift: -0.4 + Math.random() * 0.8,
          opacity: 0.25 + Math.random() * 0.5,
        })
      }
      particlesRef.current = next
    }

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = Math.max(1, Math.floor(rect.width * ratio))
      canvas.height = Math.max(1, Math.floor(rect.height * ratio))
      spawnParticles(Math.floor((rect.width * rect.height) / 6000))
    }

    const render = () => {
      const { width, height } = canvas.getBoundingClientRect()
      const ratio = Math.min(window.devicePixelRatio || 1, 2)
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
      ctx.clearRect(0, 0, width, height)
      ctx.globalCompositeOperation = 'lighter'

      const gradient = ctx.createRadialGradient(
        width * 0.7,
        height * 0.2,
        width * 0.05,
        width * 0.7,
        height * 0.2,
        width * 0.55
      )
      gradient.addColorStop(0, 'rgba(120, 180, 255, 0.15)')
      gradient.addColorStop(0.5, 'rgba(80, 140, 255, 0.05)')
      gradient.addColorStop(1, 'rgba(0, 0, 0, 0)')
      ctx.fillStyle = gradient
      ctx.fillRect(0, 0, width, height)

      particlesRef.current.forEach((particle) => {
        particle.y -= particle.speed * 0.012
        particle.x += particle.drift * 0.003

        if (particle.y < -0.1) particle.y = 1.1
        if (particle.x < -0.1) particle.x = 1.1
        if (particle.x > 1.1) particle.x = -0.1

        const px = particle.x * width
        const py = particle.y * height

        ctx.beginPath()
        ctx.arc(px, py, particle.radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(140, 200, 255, ${particle.opacity})`
        ctx.fill()
      })

      ctx.globalCompositeOperation = 'source-over'
      animationRef.current = window.requestAnimationFrame(render)
    }

    resize()
    render()
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      window.cancelAnimationFrame(animationRef.current)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={`absolute inset-0 w-full h-full ${className}`}
      aria-hidden="true"
      style={{ display: 'block' }}
    />
  )
}
