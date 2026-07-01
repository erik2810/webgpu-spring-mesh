import { defineConfig } from 'vite';

// `base: './'` keeps asset paths relative so the built demo can be dropped into a
// portfolio sub-path (e.g. erik2810.github.io/webgpu-spring-mesh/) without rewrites.
// `target: 'esnext'` is required for the top-level `await renderer.init()` in main.ts.
export default defineConfig({
  base: './',
  build: {
    target: 'esnext',
  },
  // three's `capabilities/WebGPU.js` uses top-level await, so both the dep
  // pre-bundler and the dev transform must target a TLA-capable environment.
  esbuild: {
    target: 'esnext',
  },
  optimizeDeps: {
    esbuildOptions: {
      target: 'esnext',
    },
  },
});
