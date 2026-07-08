import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the local FastAPI instance so the client
// code uses one origin-relative base URL in both dev and production.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
