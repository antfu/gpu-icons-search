import { existsSync } from 'node:fs'
import { defineConfig } from 'tsdown'
import { StaleGuardRecorder } from 'tsdown-stale-guard'

// The promote step of the training package writes these files
const assets = ['weights/model.bin', 'weights/aliases.json'].filter(file => existsSync(file))

export default defineConfig({
  entry: [
    'src/index.ts',
    'src/fallback.ts',
  ],
  dts: true,
  exports: true,
  publint: true,
  copy: assets.map(from => ({ from, to: 'dist' })),
  plugins: [
    StaleGuardRecorder(),
  ],
})
