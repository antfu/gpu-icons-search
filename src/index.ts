import type { Prediction } from './types'
import type { ModelWeights } from './weights'
import { decodePredictions, queryUnits } from './decode'
import { splitQuery } from './features'
import { GpuModel } from './runtime'
import { parseWeights } from './weights'

export type { Prediction, PredictionKind } from './types'

export interface PredictorOptions {
  /**
   * The weight file, or a URL to fetch it from.
   * The default is the file that ships with the package.
   */
  weights?: ArrayBuffer | URL | string
  /** A device to run on. The default asks `navigator.gpu` for one. */
  device?: GPUDevice
}

export interface PredictOptions {
  /**
   * Most predictions to return.
   * @default 12
   */
  limit?: number
  /**
   * Lowest score that counts as a prediction, from 0 to 1.
   * @default 0.3
   */
  minScore?: number
  /** Abort a call whose result is no longer needed, for example after the next key press */
  signal?: AbortSignal
}

export interface Predictor {
  /** Predict words of the built-in word list that relate to the query, best first */
  predict: (query: string, options?: PredictOptions) => Promise<Prediction[]>
  /** The built-in word list. Every prediction is one of these words. */
  readonly vocabulary: readonly string[]
  /** Free the GPU buffers. The predictor cannot be used after this. */
  dispose: () => void
}

/** Tell if this environment has WebGPU. Without it, `createPredictor` rejects. */
export function isSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator
}

async function requestDevice(): Promise<GPUDevice> {
  if (!isSupported())
    throw new Error('WebGPU is not available in this environment')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter)
    throw new Error('WebGPU found no usable GPU adapter')
  return adapter.requestDevice()
}

async function loadWeights(source: PredictorOptions['weights']): Promise<ModelWeights> {
  if (source instanceof ArrayBuffer)
    return parseWeights(source)
  const response = await fetch(source ?? new URL('./model.bin', import.meta.url))
  if (!response.ok)
    throw new Error(`Download of the weight file failed with HTTP ${response.status}`)
  return parseWeights(await response.arrayBuffer())
}

/**
 * Load the model on the GPU.
 * This downloads the weight file, compiles the shaders and uploads the weights.
 */
export async function createPredictor(options: PredictorOptions = {}): Promise<Predictor> {
  const [device, weights] = await Promise.all([
    options.device ?? requestDevice(),
    loadWeights(options.weights),
  ])
  const model = new GpuModel(device, weights)
  // Calls share GPU buffers, so each call waits for the one before it
  let queue: Promise<unknown> = Promise.resolve()

  async function predict(query: string, { limit = 12, minScore = 0.3, signal }: PredictOptions = {}): Promise<Prediction[]> {
    const units = queryUnits(splitQuery(query))
    const turn = queue.then(async () => {
      signal?.throwIfAborted()
      return units.length > 0
        ? model.run(units)
        : []
    })
    queue = turn.catch(() => {})
    const scored = await turn
    signal?.throwIfAborted()
    return decodePredictions(units, scored, limit, minScore)
  }

  return {
    predict,
    vocabulary: weights.vocabulary,
    dispose: () => model.destroy(),
  }
}

let shared: Promise<Predictor> | undefined

/**
 * Predict related words with a shared predictor that loads on the first call.
 * It rejects if WebGPU is not available.
 */
export async function predict(query: string, options?: PredictOptions): Promise<Prediction[]> {
  shared ??= createPredictor()
  shared.catch(() => {
    shared = undefined
  })
  return (await shared).predict(query, options)
}
