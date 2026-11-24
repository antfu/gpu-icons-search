import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { CpuModel } from '../../src/cpu'
import { GpuModel } from '../../src/runtime'
import { requestNodeDevice, useSoftwareVulkanIfNeeded } from './gpu'
import { cacheDir, readJSON } from './paths'
import { loadModel } from './reference'

/** Scores of two runtimes can differ by this much: float32 rounding and four stored decimals */
export const TOLERANCE = 2e-4

interface Fixture {
  /** The words of one query */
  words: string[]
  /** Unit vector of the query from PyTorch */
  vector: number[]
  /** Best words with their cosines from PyTorch */
  best: [word: string, score: number][]
}

type TopWords = (words: string[]) => { word: string, score: number }[] | Promise<{ word: string, score: number }[]>

/** Largest difference between the scores of a runtime and the PyTorch fixtures */
export async function compareWithFixtures(dir: string, top: TopWords): Promise<number> {
  let worst = 0
  for (const fixture of readJSON<Fixture[]>(join(dir, 'fixtures.json'))) {
    const scores = new Map((await top(fixture.words)).map(({ word, score }) => [word, score]))
    for (const [word, score] of fixture.best)
      worst = Math.max(worst, Math.abs(score - (scores.get(word) ?? Number.POSITIVE_INFINITY)))
  }
  return worst
}

/** The best words of a runtime for one bag of words */
export function modelTop(model: CpuModel | GpuModel): TopWords {
  return async words => model.run([
    {
      words,
      from: words.join(' '),
    },
  ])
}

/**
 * Compare the CPU runtime and the WGSL runtime with the PyTorch fixtures of a candidate.
 * Usage: tsx src/parity.ts [candidate folder]
 */
async function main(): Promise<void> {
  useSoftwareVulkanIfNeeded()
  const dir = process.argv[2] ?? join(cacheDir, 'candidate')
  const model = loadModel(dir)
  const { device, adapter } = await requestNodeDevice()
  const gpu = new GpuModel(device, model)
  const results = {
    reference: await compareWithFixtures(dir, modelTop(new CpuModel(model))),
    webgpu: await compareWithFixtures(dir, modelTop(gpu)),
  }
  gpu.destroy()
  console.log(`adapter: ${adapter}`)
  for (const [runtime, difference] of Object.entries(results)) {
    console.log(`${runtime.padEnd(10)} largest score difference ${difference.toExponential(2)} ${difference <= TOLERANCE
      ? 'ok'
      : 'FAILED'}`)
  }
  console.log(`weight file: ${readFileSync(join(dir, 'model.bin')).byteLength} bytes`)
  process.exit(Object.values(results).every(difference => difference <= TOLERANCE)
    ? 0
    : 1)
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error)
    process.exit(1)
  })
}
