import type { Prediction } from '../../src/types'
import type { Predictor } from './search'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { env, pipeline } from '@huggingface/transformers'
import { cacheDir } from './paths'

/** Small English embedding models with MIT or Apache licenses */
export const TEACHER_MODELS = ['Xenova/all-MiniLM-L6-v2', 'Xenova/bge-small-en-v1.5', 'Xenova/gte-small']

const BATCH = 256

env.cacheDir = join(cacheDir, 'huggingface')

/** Unit-length embeddings of a list of texts, one row per text */
export interface Embeddings {
  texts: string[]
  dimensions: number
  /** Row-major matrix, `texts.length` rows */
  values: Float32Array
}

function cacheFile(model: string, name: string): string {
  return join(cacheDir, 'teacher', model.replaceAll('/', '--'), name)
}

/**
 * Embed texts with a teacher model. A named result is kept in the cache folder.
 * Each text is embedded as it is, with mean pooling and unit length.
 */
export async function embed(model: string, texts: string[], name?: string): Promise<Embeddings> {
  const file = name
    ? cacheFile(model, `${name}.bin`)
    : undefined
  const listFile = name
    ? cacheFile(model, `${name}.json`)
    : undefined
  if (file && listFile && existsSync(file) && readFileSync(listFile, 'utf8') === JSON.stringify(texts)) {
    const values = new Float32Array(readFileSync(file).buffer)
    return { texts, dimensions: values.length / texts.length, values }
  }
  const extract = await pipeline('feature-extraction', model, { dtype: 'fp32' })
  let values = new Float32Array(0)
  let dimensions = 0
  for (let start = 0; start < texts.length; start += BATCH) {
    const output = await extract(texts.slice(start, start + BATCH), { pooling: 'mean', normalize: true })
    dimensions = output.dims[1]!
    if (values.length === 0)
      values = new Float32Array(texts.length * dimensions)
    values.set(output.data as Float32Array, start * dimensions)
  }
  if (file && listFile) {
    mkdirSync(cacheFile(model, ''), { recursive: true })
    writeFileSync(file, values)
    writeFileSync(listFile, JSON.stringify(texts))
  }
  return { texts, dimensions, values }
}

export interface Neighbor {
  index: number
  /** Cosine similarity */
  score: number
}

/** Find the rows of `targets` that are closest to row `row` of `queries` */
export function nearest(queries: Embeddings, row: number, targets: Embeddings, limit: number): Neighbor[] {
  const d = queries.dimensions
  const query = queries.values.subarray(row * d, (row + 1) * d)
  const best: Neighbor[] = []
  for (let index = 0; index < targets.texts.length; index++) {
    let score = 0
    const offset = index * d
    for (let i = 0; i < d; i++)
      score += query[i]! * targets.values[offset + i]!
    if (best.length < limit || score > best.at(-1)!.score) {
      best.push({ index, score })
      best.sort((a, b) => b.score - a.score)
      if (best.length > limit)
        best.pop()
    }
  }
  return best
}

/**
 * Use the teacher model itself as the predictor.
 * The queries must be known up front, because embedding is asynchronous.
 */
export async function teacherPredictor(model: string, vocabulary: string[], queries: string[], limit = 8): Promise<Predictor> {
  const words = await embed(model, vocabulary, 'vocabulary')
  const unique = [
    ...new Set(queries),
  ]
  const embedded = await embed(model, unique)
  const rows = new Map(unique.map((query, row) => [query, row]))
  return (query) => {
    const row = rows.get(query)
    if (row === undefined)
      return []
    return nearest(embedded, row, words, limit + 4)
      .filter(neighbor => vocabulary[neighbor.index] !== query)
      .slice(0, limit)
      .map((neighbor): Prediction => ({
        word: vocabulary[neighbor.index]!,
        score: neighbor.score,
        from: query,
        kind: 'related',
      }))
  }
}
