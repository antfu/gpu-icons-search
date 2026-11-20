import type { QueryUnit, ScoredWord } from './decode'
import type { ModelWeights } from './weights'
import { wordFeatures } from './features'
import { createShader, INPUT_LAYOUT, MAX_UNITS, paramLayout, SCORE_GROUP, TOP_WORDS } from './shader'

/** Bytes of one `Best` entry in the result buffer: a word index and a score */
const BEST_BYTES = 8

/** Buffer usage flags of the WebGPU specification. Not every runtime defines the global `GPUBufferUsage`. */
const USAGE = {
  MAP_READ: 0x0001,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  STORAGE: 0x0080,
}

/** `MAP_MODE_READ` of the WebGPU specification */
const MAP_MODE_READ = 0x0001

/**
 * The model on one GPU device.
 * It uploads the weights once, and reuses its pipelines and buffers for every call.
 */
export class GpuModel {
  private readonly inputBuffer: GPUBuffer
  private readonly readBuffer: GPUBuffer
  private readonly bestBuffer: GPUBuffer
  private readonly buffers: GPUBuffer[]
  private readonly passes: { pipeline: GPUComputePipeline, group: GPUBindGroup }[]
  private readonly input = new Uint32Array(INPUT_LAYOUT.length)

  constructor(private readonly device: GPUDevice, private readonly weights: ModelWeights) {
    const { dim, hidden, vocabulary } = weights
    const shape = { dim, hidden, words: vocabulary.length }
    const layout = paramLayout(shape)
    const params = new Float32Array(layout.length)
    params.set(weights.gateWeight, layout.gateWeight)
    params.set(weights.gateBias, layout.gateBias)
    params.set(weights.fc1Weight, layout.fc1Weight)
    params.set(weights.fc1Bias, layout.fc1Bias)
    params.set(weights.fc2Weight, layout.fc2Weight)
    params.set(weights.fc2Bias, layout.fc2Bias)

    const storage = USAGE.STORAGE
    const features = this.upload(weights.features)
    const words = this.upload(weights.words)
    const paramBuffer = this.upload(params)
    this.inputBuffer = device.createBuffer({ size: this.input.byteLength, usage: storage | USAGE.COPY_DST })
    const units = device.createBuffer({ size: MAX_UNITS * dim * 4, usage: storage })
    const scores = device.createBuffer({ size: MAX_UNITS * vocabulary.length * 4, usage: storage })
    this.bestBuffer = device.createBuffer({ size: MAX_UNITS * TOP_WORDS * BEST_BYTES, usage: storage | USAGE.COPY_SRC })
    this.readBuffer = device.createBuffer({ size: this.bestBuffer.size, usage: USAGE.MAP_READ | USAGE.COPY_DST })
    this.buffers = [
      features,
      words,
      paramBuffer,
      this.inputBuffer,
      units,
      scores,
      this.bestBuffer,
      this.readBuffer,
    ]

    const module = device.createShaderModule({
      code: createShader(shape),
    })
    const input = this.inputBuffer
    this.passes = [
      this.createPass(module, 'encode', [
        [0, input],
        [1, features],
        [2, paramBuffer],
        [3, units],
      ]),
      this.createPass(module, 'score', [
        [0, input],
        [4, units],
        [5, words],
        [6, scores],
      ]),
      this.createPass(module, 'select', [
        [0, input],
        [7, scores],
        [8, this.bestBuffer],
      ]),
    ]
  }

  private upload(values: Float32Array): GPUBuffer {
    const buffer = this.device.createBuffer({ size: values.byteLength, usage: USAGE.STORAGE, mappedAtCreation: true })
    new Float32Array(buffer.getMappedRange()).set(values)
    buffer.unmap()
    return buffer
  }

  private createPass(module: GPUShaderModule, entryPoint: string, bindings: [number, GPUBuffer][]): GpuModel['passes'][number] {
    const pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint },
    })
    const group = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map(([binding, buffer]) => ({
        binding,
        resource: { buffer },
      })),
    })
    return { pipeline, group }
  }

  /** Write the units of one query into the input buffer, in the layout of `INPUT_LAYOUT` */
  private writeInput(units: QueryUnit[]): void {
    const { input } = this
    input[0] = units.length
    let word = 0
    let feature = 0
    units.forEach((unit, index) => {
      input[INPUT_LAYOUT.unitStart + index] = word
      for (const text of unit.words) {
        input[INPUT_LAYOUT.wordStart + word] = feature
        for (const row of wordFeatures(text, this.weights.rows))
          input[INPUT_LAYOUT.featureRows + feature++] = row
        word++
      }
    })
    input[INPUT_LAYOUT.unitStart + units.length] = word
    input[INPUT_LAYOUT.wordStart + word] = feature
    this.device.queue.writeBuffer(this.inputBuffer, 0, input)
  }

  /**
   * Score the word list for each unit, and return the best words of each unit.
   * Only one call can run at a time, because the calls share buffers.
   */
  async run(units: QueryUnit[]): Promise<ScoredWord[]> {
    this.writeInput(units)
    const words = this.weights.vocabulary.length
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    const counts = [
      [units.length, 1],
      [
        Math.ceil(words / SCORE_GROUP),
        units.length,
      ],
      [units.length, 1],
    ] as const
    this.passes.forEach(({ pipeline, group }, index) => {
      pass.setPipeline(pipeline)
      pass.setBindGroup(0, group)
      pass.dispatchWorkgroups(counts[index]![0], counts[index]![1])
    })
    pass.end()
    const bytes = units.length * TOP_WORDS * BEST_BYTES
    encoder.copyBufferToBuffer(this.bestBuffer, 0, this.readBuffer, 0, bytes)
    this.device.queue.submit([
      encoder.finish(),
    ])

    await this.readBuffer.mapAsync(MAP_MODE_READ, 0, bytes)
    const range = this.readBuffer.getMappedRange(0, bytes)
    const indexes = new Uint32Array(range.slice(0))
    const values = new Float32Array(indexes.buffer)
    this.readBuffer.unmap()

    const scored: ScoredWord[] = []
    for (let unit = 0; unit < units.length; unit++) {
      for (let slot = 0; slot < TOP_WORDS; slot++) {
        const at = (unit * TOP_WORDS + slot) * 2
        scored.push({ unit, word: this.weights.vocabulary[indexes[at]!]!, score: values[at + 1]! })
      }
    }
    return scored
  }

  destroy(): void {
    for (const buffer of this.buffers)
      buffer.destroy()
  }
}
