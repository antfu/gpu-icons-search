/**
 * The weight file has three parts:
 * a little-endian uint32 with the byte length of the header, the header as JSON, and the payload.
 * The payload starts at the next multiple of 4 bytes after the header.
 */

/** How the numbers of a tensor are stored */
type TensorType
  /** Little-endian float32 */
  = | 'f32'
  /** Signed 6-bit integers, four values in three bytes. The value is the integer times `scale`. */
    | 'q6'

interface TensorHeader {
  name: string
  shape: number[]
  type: TensorType
  /** Multiplier of a `q6` tensor */
  scale?: number
  /** Each row of the tensor gets unit length after loading */
  unit?: boolean
  /** Byte offset in the payload */
  offset: number
}

interface WeightsHeader {
  version: 1
  /** Rows of the feature table, a power of two */
  rows: number
  /** Length of a word vector */
  dim: number
  /** Width of the hidden layer */
  hidden: number
  tensors: TensorHeader[]
  /** Byte offset and length of the word list in the payload: UTF-8, one word per line */
  vocabulary: { offset: number, length: number }
}

export interface ModelWeights {
  rows: number
  dim: number
  hidden: number
  /** The built-in word list. Row `i` of the `words` tensor belongs to `vocabulary[i]`. */
  vocabulary: string[]
  /** `rows` x `dim` */
  features: Float32Array
  /** `vocabulary.length` x `dim`, unit rows */
  words: Float32Array
  gateWeight: Float32Array
  gateBias: Float32Array
  /** `hidden` x `dim` */
  fc1Weight: Float32Array
  fc1Bias: Float32Array
  /** `dim` x `hidden` */
  fc2Weight: Float32Array
  fc2Bias: Float32Array
}

const TENSOR_NAMES = [
  'features',
  'words',
  'gateWeight',
  'gateBias',
  'fc1Weight',
  'fc1Bias',
  'fc2Weight',
  'fc2Bias',
] as const

/** Unpack signed 6-bit integers. Four values share three bytes, lowest bits first. */
function unpack6(bytes: Uint8Array, count: number, scale: number): Float32Array {
  const values = new Float32Array(count)
  for (let i = 0, byte = 0; i < count; i += 4, byte += 3) {
    const group = bytes[byte]! | (bytes[byte + 1]! << 8) | (bytes[byte + 2]! << 16)
    for (let j = 0; j < 4 && i + j < count; j++)
      values[i + j] = (((group >> (6 * j)) & 63) - 32) * scale
  }
  return values
}

function normalizeRows(values: Float32Array, width: number): void {
  for (let start = 0; start < values.length; start += width) {
    let sum = 0
    for (let i = start; i < start + width; i++)
      sum += values[i]! ** 2
    const norm = Math.max(Math.sqrt(sum), 1e-12)
    for (let i = start; i < start + width; i++)
      values[i]! /= norm
  }
}

function readTensor(payload: Uint8Array, tensor: TensorHeader): Float32Array {
  const count = tensor.shape.reduce((a, b) => a * b, 1)
  let values: Float32Array
  if (tensor.type === 'q6') {
    values = unpack6(payload.subarray(tensor.offset), count, tensor.scale ?? 1)
  }
  else {
    // Copy, because the source may not be aligned to 4 bytes
    values = new Float32Array(payload.slice(tensor.offset, tensor.offset + count * 4).buffer)
  }
  if (tensor.unit)
    normalizeRows(values, tensor.shape.at(-1)!)
  return values
}

/** Parse a weight file into float arrays that are ready for upload to the GPU */
export function parseWeights(file: ArrayBuffer): ModelWeights {
  const view = new DataView(file)
  const headerLength = view.getUint32(0, true)
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(file, 4, headerLength))) as WeightsHeader
  if (header.version !== 1)
    throw new Error(`Unsupported weight file version ${String(header.version)}`)
  const payload = new Uint8Array(file, Math.ceil((4 + headerLength) / 4) * 4)
  const tensors = new Map(header.tensors.map(tensor => [
    tensor.name,
    readTensor(payload, tensor),
  ]))
  const missing = TENSOR_NAMES.filter(name => !tensors.has(name))
  if (missing.length > 0)
    throw new Error(`Weight file has no tensor ${missing.join(', ')}`)
  const { offset, length } = header.vocabulary
  const vocabulary = new TextDecoder().decode(payload.subarray(offset, offset + length)).split('\n')
  const [features, words, gateWeight, gateBias, fc1Weight, fc1Bias, fc2Weight, fc2Bias] = TENSOR_NAMES.map(name => tensors.get(name)!)
  return {
    rows: header.rows,
    dim: header.dim,
    hidden: header.hidden,
    vocabulary,
    features: features!,
    words: words!,
    gateWeight: gateWeight!,
    gateBias: gateBias!,
    fc1Weight: fc1Weight!,
    fc1Bias: fc1Bias!,
    fc2Weight: fc2Weight!,
    fc2Bias: fc2Bias!,
  }
}
