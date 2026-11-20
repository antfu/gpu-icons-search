import { execFileSync } from 'node:child_process'
import { statSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { predictorMethod, runBenchmark } from './bench/run'
import { cacheDir, dataDir, trainingRoot, writeJSON } from './paths'
import { loadModel, referencePredictor } from './reference'

/** Model sizes to compare: rows of the feature table and length of a word vector */
const SIZES = [
  { rows: 4096, dim: 16 },
  { rows: 8192, dim: 24 },
  { rows: 16384, dim: 32 },
]

/** Shorter than a full run, so that the sweep finishes in reasonable time on a CPU */
const STEPS = 8000

export interface SweepEntry {
  rows: number
  dim: number
  steps: number
  /** Size of the weight file in bytes */
  bytes: number
  /** Share of found cases per test set, with the reference implementation */
  found: Record<string, number>
}

/**
 * Train the model at several sizes and score each one, to show size against score.
 * The result goes to `data/sweep.json`, which the playground shows.
 */
function main(): void {
  const entries: SweepEntry[] = []
  for (const { rows, dim } of SIZES) {
    const out = join(cacheDir, 'sweep', `r${rows}-d${dim}`)
    const args = [
      'run',
      '--locked',
      'python',
      'torch/train.py',
      '--rows',
      String(rows),
      '--dim',
      String(dim),
      '--steps',
      String(STEPS),
      '--qat-steps',
      String(STEPS / 4),
      '--out',
      out,
    ]
    execFileSync('uv', args, { cwd: trainingRoot, stdio: 'inherit' })
    const report = runBenchmark([
      predictorMethod('model', referencePredictor(loadModel(out))),
    ])
    const found = Object.fromEntries(Object.entries(report.model!).map(([set, score]) => [
      set,
      Number(score.found.toFixed(4)),
    ]))
    entries.push({
      rows,
      dim,
      steps: STEPS,
      bytes: statSync(join(out, 'model.bin')).size,
      found,
    })
    console.log(entries.at(-1))
  }
  writeJSON(join(dataDir, 'sweep.json'), entries)
}

try {
  main()
}
catch (error) {
  console.error(error)
  process.exitCode = 1
}
