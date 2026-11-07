import { copyFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { defineConfig, type Plugin } from "vite";

// Serve the ONNX Runtime WebAssembly files from this site instead of a CDN (supply chain, D-027):
// transformers.js would otherwise import them at run time from cdn.jsdelivr.net.
const ORT_FILES = ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"];
function localOrt(): Plugin {
  return {
    name: "local-onnxruntime",
    buildStart() {
      // the onnxruntime-web that transformers.js depends on (one copy, locked in package-lock.json)
      const ortDist = "node_modules/onnxruntime-web/dist";
      mkdirSync("public/ort", { recursive: true });
      for (const f of ORT_FILES) copyFileSync(join(ortDist, f), join("public/ort", f));
    },
  };
}

// base can be overridden for GitHub Pages: ATLAS_BASE=/sphota/ npm run build
export default defineConfig({
  base: process.env.ATLAS_BASE ?? "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  worker: { format: "es" },
  plugins: [localOrt()],
});
