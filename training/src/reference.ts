import type { ModelWeights } from '../../src/weights'
import type { Predictor } from './search'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { CpuModel } from '../../src/cpu'
import { decodePredictions, queryUnits } from '../../src/decode'
import { splitQuery } from '../../src/features'
import { parseWeights } from '../../src/weights'

/** Load a weight file that `torch/train.py` exported */
export function loadModel(dir: string): ModelWeights {
  const bytes = readFileSync(join(dir, 'model.bin'))
  return parseWeights(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
}

export interface PredictOptions {
  /** Most predictions per query */
  limit?: number
  /** Lowest cosine that counts as a prediction */
  minScore?: number
}

/**
 * Use the CPU runtime of the package as a synchronous predictor, for offline scripts.
 * It is the reference implementation: the WGSL runtime must give the same numbers.
 */
export function referencePredictor(model: ModelWeights, { limit = 8, minScore = 0.3 }: PredictOptions = {}): Predictor {
  const cpu = new CpuModel(model)
  return (query) => {
    const units = queryUnits(splitQuery(query))
    return decodePredictions(units, cpu.run(units), limit, minScore)
  }
}
