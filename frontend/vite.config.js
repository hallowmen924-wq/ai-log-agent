import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const NGROK_DEV_HOST = 'wrath-studied-mushy.ngrok-free.dev';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      'buffer/': 'buffer',
    },
  },
  build: {
    chunkSizeWarningLimit: 1000,
    rollupOptions: {
      input: {
        main: 'index.html',
        bunny: 'bunny.html',
      },
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) {
            return undefined;
          }
          if (id.includes('plotly.js') || id.includes('react-plotly.js')) {
            return 'plotly-vendor';
          }
          if (id.includes('reactflow') || id.includes('@reactflow') || id.includes('d3-')) {
            return 'ontology-runtime';
          }
          if (id.includes('framer-motion') || id.includes('motion-dom') || id.includes('motion-utils')) {
            return 'motion-vendor';
          }
          return 'vendor';
        },
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    allowedHosts: [
      NGROK_DEV_HOST,
    ],
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:18000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: process.env.VITE_BACKEND_PROXY_TARGET || 'http://127.0.0.1:18000',
        ws: true,
        changeOrigin: true,
      },
    },
  },
});
