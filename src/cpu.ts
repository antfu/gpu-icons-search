import type { QueryUnit, ScoredWord } from './decode'
import type { ModelWeights } from './weights'
import { wordFeatures } from './features'
import { TOP_WORDS } from './shader'

/**
 * The model on the CPU, in plain TypeScript.
 * It does the same steps as the three passes of the WGSL shader, and must give the same numbers.
 */
export class CpuModel {
  private readonly topIndex = new Uint32Array(TOP_WORDS)
  private readonly topScore = new Float64Array(TOP_WORDS)

  constructor(private readonly weights: ModelWeights) {}

  /** The vector of one word: the weighted mean of the rows of its features */
  private wordVector(word: string): Float32Array {
    const { dim, rows, features, featureWeight } = this.weights
    const vector = new Float32Array(dim)
    let total = 0
    for (const row of wordFeatures(word, rows)) {
      const weight = featureWeight[row]!
      total += weight
      for (let i = 0; i < dim; i++)
        vector[i]! += weight * features[row * dim + i]!
    }
    for (let i = 0; i < dim; i++)
      vector[i]! /= total
    return vector
  }

  /** The unit vector of a bag of words: a gated mean of the word vectors, then one residual layer */
  private encode(words: string[]): Float32Array {
    const { dim, hidden, gateWeight, gateBias, fc1Weight, fc1Bias, fc2Weight, fc2Bias } = this.weights
    const pooled = new Float32Array(dim)
    let total = 0
    for (const word of words) {
      const vector = this.wordVector(word)
      let gate = gateBias[0]!
      for (let i = 0; i < dim; i++)
        gate += vector[i]! * gateWeight[i]!
      gate = 1 / (1 + Math.exp(-gate))
      total += gate
      for (let i = 0; i < dim; i++)
        pooled[i]! += gate * vector[i]!
    }
    for (let i = 0; i < dim; i++)
      pooled[i]! /= total + 1e-6
    const activation = new Float32Array(hidden)
    for (let h = 0; h < hidden; h++) {
      let sum = fc1Bias[h]!
      for (let i = 0; i < dim; i++)
        sum += pooled[i]! * fc1Weight[h * dim + i]!
      activation[h] = Math.max(sum, 0)
    }
    const output = new Float32Array(dim)
    let norm = 0
    for (let i = 0; i < dim; i++) {
      let sum = pooled[i]! + fc2Bias[i]!
      for (let h = 0; h < hidden; h++)
        sum += activation[h]! * fc2Weight[i * hidden + h]!
      output[i] = sum
      norm += sum * sum
    }
    const length = Math.max(Math.sqrt(norm), 1e-12)
    for (let i = 0; i < dim; i++)
      output[i]! /= length
    return output
  }

  /**
   * Score the word list against one unit vector, and keep the words with the highest cosines.
   * It returns how many words it kept. They are in `topIndex` and `topScore`, best first.
   */
  private select(vector: Float32Array): number {
    const { dim, words, vocabulary } = this.weights
    const { topIndex, topScore } = this
    let kept = 0
    for (let word = 0; word < vocabulary.length; word++) {
      const start = word * dim
      let score = 0
      for (let i = 0; i < dim; i++)
        score += vector[i]! * words[start + i]!
      if (kept === TOP_WORDS && score <= topScore[kept - 1]!)
        continue
      let slot = kept < TOP_WORDS
        ? kept++
        : kept - 1
      // A word with the same score as an earlier word stays behind it, as on the GPU
      while (slot > 0 && topScore[slot - 1]! < score) {
        topScore[slot] = topScore[slot - 1]!
        topIndex[slot] = topIndex[slot - 1]!
        slot--
      }
      topScore[slot] = score
      topIndex[slot] = word
    }
    return kept
  }

  /** Score the word list for each unit, and return the best words of each unit */
  run(units: QueryUnit[]): ScoredWord[] {
    const scored: ScoredWord[] = []
    units.forEach((unit, index) => {
      const kept = this.select(this.encode(unit.words))
      for (let slot = 0; slot < kept; slot++)
        scored.push({ unit: index, word: this.weights.vocabulary[this.topIndex[slot]!]!, score: this.topScore[slot]! })
    })
    return scored
  }

  /** The CPU model holds no resources. This exists so that both models have the same methods. */
  destroy(): void {}
}
