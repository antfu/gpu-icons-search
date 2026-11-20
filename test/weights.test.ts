import { describe, expect, it } from 'vitest'
import { parseWeights } from '../src/weights'

/** Pack integers from -32 to 31 the way the Python exporter does */
function pack6(values: number[]): number[] {
  const bytes: number[] = []
  for (let i = 0; i < values.length; i += 4) {
    const group = [0, 1, 2, 3].reduce((sum, j) => sum | (((values[i + j] ?? 0) + 32) << (6 * j)), 0)
    bytes.push(group & 255, (group >> 8) & 255, (group >> 16) & 255)
  }
  return bytes
}

function floats(values: number[]): number[] {
  return [
    ...new Uint8Array(new Float32Array(values).buffer),
  ]
}

/** Build a small weight file in memory, without the tensor named `skip` */
function buildFile(skip?: string): ArrayBuffer {
  const parts: { name: string, shape: number[], type: string, scale?: number, unit?: boolean, bytes: number[] }[] = [
    {
      name: 'features',
      shape: [2, 2],
      type: 'q6',
      scale: 0.5,
      bytes: pack6([-31, 0, 1, 31]),
    },
    {
      name: 'featureWeight',
      shape: [2],
      type: 'f32',
      bytes: floats([1, 0.5]),
    },
    {
      name: 'words',
      shape: [2, 2],
      type: 'q6',
      scale: 1,
      unit: true,
      bytes: pack6([3, 4, 0, -5]),
    },
    {
      name: 'gateWeight',
      shape: [2],
      type: 'f32',
      bytes: floats([1, 2]),
    },
    {
      name: 'gateBias',
      shape: [1],
      type: 'f32',
      bytes: floats([3]),
    },
    {
      name: 'fc1Weight',
      shape: [1, 2],
      type: 'f32',
      bytes: floats([4, 5]),
    },
    {
      name: 'fc1Bias',
      shape: [1],
      type: 'f32',
      bytes: floats([6]),
    },
    {
      name: 'fc2Weight',
      shape: [2, 1],
      type: 'f32',
      bytes: floats([7, 8]),
    },
    {
      name: 'fc2Bias',
      shape: [2],
      type: 'f32',
      bytes: floats([9, 10]),
    },
  ]
  const payload: number[] = []
  const tensors = parts.filter(part => part.name !== skip).map(({ bytes, ...tensor }) => {
    while (payload.length % 4 !== 0)
      payload.push(0)
    const offset = payload.length
    payload.push(...bytes)
    return { ...tensor, offset }
  })
  const words = [
    ...new TextEncoder().encode('home\nhouse'),
  ]
  const header = {
    version: 2,
    rows: 2,
    dim: 2,
    hidden: 1,
    tensors,
    vocabulary: { offset: payload.length, length: words.length },
  }
  payload.push(...words)
  const encoded = new TextEncoder().encode(JSON.stringify(header))
  const start = Math.ceil((4 + encoded.length) / 4) * 4
  const file = new Uint8Array(start + payload.length)
  new DataView(file.buffer).setUint32(0, encoded.length, true)
  file.set(encoded, 4)
  file.set(payload, start)
  return file.buffer
}

describe('parseWeights', () => {
  it('reads 6-bit and float tensors and the word list', () => {
    const weights = parseWeights(buildFile())
    expect(weights.vocabulary).toEqual(['home', 'house'])
    expect([...weights.features]).toEqual([-15.5, 0, 0.5, 15.5])
    expect([...weights.words].map(value => Number(value.toFixed(4)))).toEqual([0.6, 0.8, 0, -1])
    expect([...weights.fc2Weight]).toEqual([7, 8])
    expect(weights).toMatchObject({ rows: 2, dim: 2, hidden: 1 })
  })

  it('rejects a file with a missing tensor', () => {
    expect(() => parseWeights(buildFile('gateBias'))).toThrow('gateBias')
  })
})
