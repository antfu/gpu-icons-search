import type { ScoredWord } from '../../src/decode'
import type { ModelWeights } from '../../src/weights'
import type { Predictor } from './search'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { decodePredictions, queryUnits } from '../../src/decode'
import { splitQuery, wordFeatures } from '../../src/features'
import { parseWeights } from '../../src/weights'

/** Load a weight file that `torch/train.py` exported */
export function loadModel(dir: string): ModelWeights {
  const bytes = readFileSync(join(dir, 'model.bin'))
  return parseWeights(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength))
}

function dot(a: Float32Array, aOffset: number, b: Float32Array, bOffset: number, length: number): number {
  let sum = 0
  for (let i = 0; i < length; i++)
    sum += a[aOffset + i]! * b[bOffset + i]!
  return sum
}

/** The vector of one word: the mean of the rows of its features */
function wordVector(model: ModelWeights, word: string): Float32Array {
  const { dim, rows } = model
  const features = wordFeatures(word, rows)
  const vector = new Float32Array(dim)
  for (const row of features) {
    for (let i = 0; i < dim; i++)
      vector[i]! += model.features[row * dim + i]! / features.length
  }
  return vector
}

/**
 * Plain TypeScript version of the model, for tests and offline evaluation.
 * It returns the unit vector of a bag of words. The WGSL runtime must give the same numbers.
 */
export function encodeUnit(model: ModelWeights, words: string[]): Float32Array {
  const { dim, hidden } = model
  const pooled = new Float32Array(dim)
  let total = 0
  for (const word of words) {
    const vector = wordVector(model, word)
    const gate = 1 / (1 + Math.exp(-(dot(vector, 0, model.gateWeight, 0, dim) + model.gateBias[0]!)))
    total += gate
    for (let i = 0; i < dim; i++)
      pooled[i]! += gate * vector[i]!
  }
  for (let i = 0; i < dim; i++)
    pooled[i]! /= total + 1e-6
  const activation = new Float32Array(hidden)
  for (let h = 0; h < hidden; h++)
    activation[h] = Math.max(0, dot(pooled, 0, model.fc1Weight, h * dim, dim) + model.fc1Bias[h]!)
  const output = new Float32Array(dim)
  let norm = 0
  for (let i = 0; i < dim; i++) {
    output[i] = pooled[i]! + dot(activation, 0, model.fc2Weight, i * hidden, hidden) + model.fc2Bias[i]!
    norm += output[i]! ** 2
  }
  return output.map(value => value / Math.max(Math.sqrt(norm), 1e-12))
}

/** The best words of the word list for one unit vector, best first */
export function topWords(model: ModelWeights, vector: Float32Array, limit: number): { word: string, score: number }[] {
  const { dim, vocabulary } = model
  const best: { word: string, score: number }[] = []
  for (let index = 0; index < vocabulary.length; index++) {
    const score = dot(vector, 0, model.words, index * dim, dim)
    if (best.length < limit || score > best.at(-1)!.score) {
      best.push({ word: vocabulary[index]!, score })
      best.sort((a, b) => b.score - a.score)
      if (best.length > limit)
        best.pop()
    }
  }
  return best
}

export interface PredictOptions {
  /** Most predictions per query */
  limit?: number
  /** Lowest cosine that counts as a prediction */
  minScore?: number
}

/** Use the reference implementation as a predictor */
export function referencePredictor(model: ModelWeights, { limit = 8, minScore = 0.3 }: PredictOptions = {}): Predictor {
  return (query) => {
    const units = queryUnits(splitQuery(query))
    const scored: ScoredWord[] = units.flatMap((unit, index) =>
      topWords(model, encodeUnit(model, unit.words), limit + unit.words.length).map(top => ({ unit: index, ...top })))
    return decodePredictions(units, scored, limit, minScore)
  }
}
