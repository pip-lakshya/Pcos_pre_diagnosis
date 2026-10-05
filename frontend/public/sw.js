const CACHE='narisaarthi-shell-v2';
const SHELL=['/','/index.html','/manifest.webmanifest','/narisaarthi-logo.png','/narisaarthi-icon-192.svg','/narisaarthi-icon-512.svg'];
self.addEventListener('install',event=>event.waitUntil((async()=>{
 const cache=await caches.open(CACHE);
 const page=await fetch('/index.html');
 await cache.put('/index.html',page.clone());
 const html=await page.text();
 const builtAssets=[...html.matchAll(/(?:src|href)="(\/assets\/[^\"]+)"/g)].map(match=>match[1]);
 await cache.addAll([...new Set([...SHELL,...builtAssets])]);
 await self.skipWaiting();
})()));
self.addEventListener('activate',event=>event.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(key=>key!==CACHE).map(key=>caches.delete(key)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',event=>{
 const request=event.request;
 if(request.method!=='GET'||new URL(request.url).origin!==self.location.origin)return;
 if(request.mode==='navigate'){
  event.respondWith(fetch(request).then(response=>{const copy=response.clone();void caches.open(CACHE).then(cache=>cache.put('/index.html',copy));return response}).catch(()=>caches.match('/index.html')));
  return;
 }
 if(request.destination==='script'||request.destination==='style'){
  event.respondWith(fetch(request).then(response=>{if(response.ok){const copy=response.clone();void caches.open(CACHE).then(cache=>cache.put(request,copy))}return response}).catch(()=>caches.match(request)));
  return;
 }
 if(request.destination==='image'||request.destination==='font'){
  event.respondWith(caches.match(request).then(cached=>cached||fetch(request).then(response=>{if(response.ok){const copy=response.clone();void caches.open(CACHE).then(cache=>cache.put(request,copy))}return response})));
 }
});
