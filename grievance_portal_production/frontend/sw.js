const CACHE_NAME = 'grievance-portal-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/file_grievance.html',
  '/track_grievance.html',
  '/login_new.html',
  '/css/main.css',
  '/css/stylesheet.css',
  '/js/main.js',
  '/js/api.js',
  '/images/logo.svg',
  '/manifest.json'
];

// Install - cache static assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

// Activate - clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))
      )
    )
  );
  self.clients.claim();
});

// Fetch - serve from cache, fallback to network
self.addEventListener('fetch', event => {
  // Skip API requests - always fetch from network
  if (event.request.url.includes('/api/') || 
      event.request.method !== 'GET') {
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cached => {
      return cached || fetch(event.request).then(response => {
        // Cache new successful responses
        if (response.status === 200) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      });
    }).catch(() => caches.match('/index.html'))
  );
});
