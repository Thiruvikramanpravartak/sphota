import { defineConfig } from "vite";

// base can be overridden for GitHub Pages: ATLAS_BASE=/sphota/ npm run build
export default defineConfig({
  base: process.env.ATLAS_BASE ?? "./",
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  worker: { format: "es" },
});
