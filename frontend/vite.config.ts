import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/upload": {
        target: "http://localhost:5000/",
        changeOrigin: true,
        secure: false,
      },
      "/video_feed": {
        target: "http://localhost:5000/",
        changeOrigin: true,
        secure: false,
      },
      "/original_video": {
        target: "http://localhost:5000/",
        changeOrigin: true,
        secure: false,
      },
    },
    cors: true,
  },
});
