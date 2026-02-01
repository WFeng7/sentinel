export const API_BASE = import.meta.env.VITE_DEPLOYMENT_API_URL ?? 'http://localhost:8000'

/**
 * Capture 3 frames (first, middle, last) from an HLS video over 10 seconds.
 * Returns base64-encoded JPEG frames with timestamps.
 */
export async function captureKeyframesFromHLS(streamUrl) {
  // We'll postMessage into the player.html iframe to request frames
  // The player should expose a method to capture frames at given times
  const iframe = document.querySelector(`iframe[src*="${encodeURIComponent(streamUrl)}"]`)
  if (!iframe) throw new Error('Player iframe not found for stream')

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Frame capture timeout')), 15000)

    const listener = ({ data }) => {
      if (data.type !== 'sentinel-keyframes') return
      clearTimeout(timeout)
      window.removeEventListener('message', listener)
      resolve(data.keyframes) // expected: [{ ts, base64 }, ...]
    }
    window.addEventListener('message', listener)

    iframe.contentWindow.postMessage({
      type: 'sentinel-capture-keyframes',
      payload: { times: [0, 5, 10] } // first, middle, last over 10s
    }, '*')
  })
}

/**
 * Call the combined VLM+RAG endpoint.
 */
export async function analyzeWithVLMAndRAG(camera) {
  const keyframes = await captureKeyframesFromHLS(camera.stream)
  const response = await fetch(`${API_BASE}/vlm+rag`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_id: camera.id,
      label: camera.label,
      stream_url: camera.stream,
      keyframes
    })
  })
  if (!response.ok) {
    throw new Error('VLM+RAG request failed')
  }
  return response.json()
}
