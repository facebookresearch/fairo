/**
 * Welcome to your Workbox-powered service worker!
 *
 * You'll need to register this file in your web app and you should
 * disable HTTP caching for this file too.
 * See https://goo.gl/nhQhGp
 *
 * The rest of the code is auto-generated. Please don't update this file
 * directly; instead, make changes to your Workbox build configuration
 * and re-run your build process.
 * See https://goo.gl/2aRDsh
 */

importScripts("https://storage.googleapis.com/workbox-cdn/releases/4.3.1/workbox-sw.js");

importScripts(
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
  "/precache-manifest.ad95ae58e7dce145f02f6f39be3386aa.js"
=======
  "/precache-manifest.b75eceaeb13538603767c068fe3d10da.js"
>>>>>>> 016fff7... Checked in yarn build files
=======
  "/precache-manifest.a8549702307893d035e1435a3c191ecc.js"
>>>>>>> d8d0982... Checked in yarn build files
=======
  "/precache-manifest.d86fb27cb2ab11add239b7190cae9253.js"
>>>>>>> 6582f6f... Checked in yarn build files
);

self.addEventListener('message', (event) => {
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
});

workbox.core.clientsClaim();

/**
 * The workboxSW.precacheAndRoute() method efficiently caches and responds to
 * requests for URLs in the manifest.
 * See https://goo.gl/S9QRab
 */
self.__precacheManifest = [].concat(self.__precacheManifest || []);
workbox.precaching.precacheAndRoute(self.__precacheManifest, {});

workbox.routing.registerNavigationRoute(workbox.precaching.getCacheKeyForURL("/index.html"), {
  
  blacklist: [/^\/_/,/\/[^/?]+\.[^/]+$/],
});
