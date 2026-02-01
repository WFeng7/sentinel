const CACHE_NAME = 'sentinel-static-v6'

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) =>
        cache.addAll([
          '/',
          '/index.html',
          '/favicon/favicon.ico',
          '/favicon/favicon-16x16.png',
          '/favicon/favicon-32x32.png',
          '/favicon/apple-touch-icon.png',
          '/favicon/android-chrome-192x192.png',
          '/favicon/android-chrome-512x512.png',
          '/favicon/site.webmanifest'
        ])
      )
      .then(() => self.skipWaiting())
  )
})

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key)))
      )
      .then(() => self.clients.claim())
  )
})

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return

  const requestUrl = new URL(event.request.url)
  if (requestUrl.origin !== self.location.origin) return

  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached

      return fetch(event.request)
        .then((response) => {
          if (!response || response.type === 'opaque') return response
          const responseClone = response.clone()
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(event.request, responseClone))
            .catch(() => {})
          return response
        })
        .catch(() => cached || new Response('', { status: 504 }))
    })
  )
})
