import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxy to the API in dev so the browser sees one origin and EventSource
    // works without CORS preflight games.
    proxy: {
      "/v1": { target: "http://localhost:8000", changeOrigin: true },
      "/healthz": "http://localhost:8000",
      "/readyz": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    rollupOptions: {
      output: {
        // recharts + d3 is most of the bundle and changes far less often than
        // our code, so it gets its own long-cached chunk.
        manualChunks: {
          charts: ["recharts"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
});
