<p align="center">
  <img width="192" src="https://user-images.githubusercontent.com/12652154/71167221-945de480-2254-11ea-97ba-faadc933ed4f.png">
</p>

[![npm version](https://badge.fury.io/js/eigen.svg)](https://badge.fury.io/js/eigen)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://bertrandbev.github.io/eigen-js/#/)
[![Made with emscripten](https://img.shields.io/badge/Made%20width-emscripten-blue.svg)](https://github.com/emscripten-core/emscripten)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Eigen.js

Eigen.js is a port of the [Eigen](https://eigen.tuxfamily.org/) C++ linear algebra library

It uses a WebAssembly compiled subset of the [Eigen](https://eigen.tuxfamily.org/) library, and implements a garbage collection mechanism to manage memory

# Live demo & documentation

An interactive documentation is available at [eigen-js](https://bertrandbev.github.io/eigen-js/#/). Stress benchmarks can be found [here](https://bertrandbev.github.io/eigen-js/#/benchmark)

## Usage

Eigen.js can be installed via [npm](https://www.npmjs.com/package/eigen) or [yarn](https://yarnpkg.com/en/package/eigen)

```bash
npm install eigen
```

```bash
yarn add eigen
```

In a node (v14) application or in the browser (using [webpack](https://webpack.js.org/))

```js
// test.mjs
import eig from 'eigen';

(async () => {
  await eig.ready;
  const M = new eig.Matrix([[1, 2], [3, 4]]);
  M.print("M");
  M.inverse();
  M.print("Minv");
  eig.GC.flush();
})();
```

This minimal example can be found under ``./example``

## Allocation

The WebAssembly binary requires a manual memory management for objects allocated on the heap. Every time a matrix is created, it will be allocated on the heap and its memory won't be freed until its `delete()` method is invoked

```js
// test.mjs
import eig from 'eigen';

(async () => {
  await eig.ready;
  const M = new eig.Matrix([[1, 2], [3, 4]]); // Memory is allocated for M
  M.print("M");
  M.delete(); // Memory is freed here
  M.print("M"); // This will trigger an error
})();
```

It can be annoying to call `delete()` on every object, expecially for chained computations. Take for example `const I2 = eig.Matrix.identity(2, 2).matAdd(eig.Matrix.identity(2, 2))`. The identity matrix passed as an argument to `matAdd(...)` will be allocated but never freed, which will leak memory. To make things easier to manage, `eig.GC` keeps tracks of all the allocated objects on the heap and frees them all upon calling `eig.GC.flush()`.

There could be instances where one would want to keep some matrices in memory while freeing a bunch of temporary ones used for computations. The method `eig.GC.pushException(...matrices)` whitelists its arguments to prevent `eig.GC.flush()` from flushing them. `eig.GC.popException(...matrices)` cancels any previous whitelisting.

```js
// test.mjs
import eig from 'eigen';

(async () => {
  await eig.ready;
  const x = new eig.Matrix([[1, 2], [3, 4]]);
  eig.GC.pushException(x); // Whitelist x
  // Perform some computations
  const R = new eig.Matrix([[.1, 0], [.5, .1]]);
  x.matAddSelf(R.matMul(eig.Matrix.ones(2, 2)));
  // Free memory
  eig.GC.flush();
  x.print("x"); // x is still around!
})();
```

## Documentation

The documentation is available at [eigen.js](https://bertrandbev.github.io/eigen-js/#/)

## Build

Make sure [Emscripten](https://emscripten.org/docs/getting_started/Tutorial.html) is intalled & activated in your terminal session

```bash
source path/to/emsdk/emsdk_env.sh
emcc -v
```

Dowload the latest versions of [Eigen](https://gitlab.com/libeigen/eigen/-/releases/) and [OSQP](https://github.com/oxfordcontrol/osqp/) (optional, see below), and put then in the `lib` directory

```bash
lib/eigen
lib/osqp
```

Now compile osqp for a Webassembly target

```bash
cd lib/osqp
mkdir build; cd build
emcmake cmake ..
emmake make
```

Once done, eigen.js can be compile to a wasm binary

```bash
# From the root directory
mkdir build
emcc -I lib/eigen -I lib/osqp/include -Isrc lib/osqp/build/out/libosqp.a -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=0 -O3 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 --bind -o build/eigen_gen.js src/cpp/embind.cc 
```

If you are not interested in the OSQP functionality, you can build without installing it with
```
emcc -D NO_OSQP -I lib/eigen  -Isrc -s DISABLE_EXCEPTION_CATCHING=0 -s ASSERTIONS=0 -O3 -s ALLOW_MEMORY_GROWTH=1 -s MODULARIZE=1 --bind -o build/eigen_gen.js src/cpp/embind.cc
```

### Generate the documentation

The documentation is generated from classes descriptions using [documentation.js](https://documentation.js.org/)

```bash
documentation build src/classes/ -f json -o docs/doc.json
```