import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
    plugins: [react(), tailwindcss()],
    server: {
        port: 5173,
        proxy: {
            '/v1': {
                target: process.env.VITE_API_BASE_URL || 'http://localhost:8000',
                changeOrigin: true,
            },
            '/health': {
                target: process.env.VITE_API_BASE_URL || 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
})
