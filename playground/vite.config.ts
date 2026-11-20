import type { Plugin } from 'vite'
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import vue from '@vitejs/plugin-vue'
import unocss from 'unocss/vite'
import { defineConfig } from 'vite'

const reviewFile = fileURLToPath(new URL('../training/data/bench/written-review.jsonl', import.meta.url))

/**
 * Dev server endpoint that stores the review of the written test queries in the repository.
 * The review page posts the full list of verdicts after each change.
 */
function reviewEndpoint(): Plugin {
  return {
    name: 'review-endpoint',
    configureServer(server) {
      server.middlewares.use('/api/review', (request, response) => {
        let body = ''
        request.on('data', (chunk: string) => body += chunk)
        request.on('end', () => {
          const verdicts = JSON.parse(body) as unknown[]
          mkdirSync(dirname(reviewFile), { recursive: true })
          writeFileSync(reviewFile, verdicts.map(verdict => `${JSON.stringify(verdict)}\n`).join(''))
          response.end('ok')
        })
      })
    },
  }
}

export default defineConfig({
  plugins: [
    vue({
      template: {
        compilerOptions: {
          isCustomElement: tag => tag === 'iconify-icon',
        },
      },
    }),
    unocss(),
    reviewEndpoint(),
  ],
  resolve: {
    alias: {
      'gpu-icons-search/fallback': fileURLToPath(new URL('../src/fallback.ts', import.meta.url)),
      'gpu-icons-search': fileURLToPath(new URL('../src/index.ts', import.meta.url)),
    },
  },
  server: {
    fs: {
      allow: ['..'],
    },
  },
})
