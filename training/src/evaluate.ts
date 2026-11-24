import type { Prediction } from '../../src/types'
import type { BenchCase } from './bench/cases'
import type { Report } from './bench/run'
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { brotliCompressSync, gzipSync } from 'node:zlib'
import { CpuModel } from '../../src/cpu'
import { createPredictor } from '../../src/index'
import { GpuModel } from '../../src/runtime'
import { baselineMethods, loadCases, predictorMethod, printReport, runBenchmark, SCREEN } from './bench/run'
import { requestNodeDevice, useSoftwareVulkanIfNeeded } from './gpu'
import { compareWithFixtures, modelTop, TOLERANCE } from './parity'
import { cacheDir, writeJSON } from './paths'
import { loadModel } from './reference'

/** Name of the model in benchmark reports */
export const MODEL_METHOD = 'fzf+model'

export interface Evaluation {
  /** SHA-256 of the weight file that these numbers belong to */
  sha256: string
  /** The GPU adapter that ran the WGSL shaders */
  adapter: string
  /** Largest score difference to the PyTorch fixtures, per runtime. `reference` is the CPU runtime. */
  parity: { reference: number, webgpu: number, tolerance: number }
  /** Size of the weight file, raw and compressed */
  bytes: { raw: number, gzip: number, brotli: number }
  /** Median time of one warm `predict` call in milliseconds, on this adapter */
  latencyMs: number
  bench: Report
  /** The cases that the model did not find, with a flag if the baseline `fzf+alias` found them */
  misses: (BenchCase & { baselineFound: boolean })[]
}

function median(values: number[]): number {
  return [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!
}

/**
 * Evaluate a candidate on real WebGPU, through the public API of the package.
 * It writes `evaluation.json` next to the weight file. The promote step reads that file.
 * Usage: tsx src/evaluate.ts [candidate folder]
 */
async function main(): Promise<void> {
  useSoftwareVulkanIfNeeded()
  const dir = process.argv[2] ?? join(cacheDir, 'candidate')
  const file = readFileSync(join(dir, 'model.bin'))
  const weights = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)
  const { device, adapter } = await requestNodeDevice()

  const model = loadModel(dir)
  const gpu = new GpuModel(device, model)
  const parity = {
    reference: await compareWithFixtures(dir, modelTop(new CpuModel(model))),
    webgpu: await compareWithFixtures(dir, modelTop(gpu)),
    tolerance: TOLERANCE,
  }
  gpu.destroy()

  const predictor = await createPredictor({ weights, device })
  const cases = loadCases()
  const predictions = new Map<string, Prediction[]>()
  const times: number[] = []
  for (const query of new Set(cases.map(test => test.query))) {
    const started = performance.now()
    predictions.set(query, await predictor.predict(query))
    times.push(performance.now() - started)
  }
  predictor.dispose()

  const ranks = new Map<string, number[]>()
  const bench = runBenchmark([
    ...baselineMethods(),
    predictorMethod(MODEL_METHOD, query => predictions.get(query) ?? []),
  ], cases, ranks)
  const misses = cases.flatMap((test, index) => ranks.get(MODEL_METHOD)![index]! > SCREEN
    ? [
        { ...test, baselineFound: ranks.get('fzf+alias')![index]! <= SCREEN },
      ]
    : [])
  const evaluation: Evaluation = {
    sha256: createHash('sha256').update(file).digest('hex'),
    adapter,
    parity,
    bytes: { raw: file.byteLength, gzip: gzipSync(file).byteLength, brotli: brotliCompressSync(file).byteLength },
    latencyMs: Number(median(times).toFixed(2)),
    bench,
    misses,
  }
  writeJSON(join(dir, 'evaluation.json'), evaluation)

  printReport(bench)
  console.log(`\nadapter   ${adapter}`)
  console.log(`parity    cpu ${parity.reference.toExponential(1)}, webgpu ${parity.webgpu.toExponential(1)} (tolerance ${TOLERANCE})`)
  console.log(`size      ${evaluation.bytes.raw} bytes raw, ${evaluation.bytes.gzip} gzip, ${evaluation.bytes.brotli} brotli`)
  console.log(`latency   ${evaluation.latencyMs} ms median per warm call on this adapter`)
  process.exit(0)
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
