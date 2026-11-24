import type { ModelWeights } from '../src/weights'
import { readFileSync } from 'node:fs'
import { describe, expect, it } from 'vitest'
import { CpuModel } from '../src/cpu'
import { createPredictor } from '../src/index'

/**
 * A model that is small enough to compute by hand.
 * Every feature row is the same, so the vector of every word is `[3, 4]`.
 */
function tinyModel(): ModelWeights {
  return {
    rows: 4,
    dim: 2,
    hidden: 1,
    vocabulary: ['east', 'north', 'between'],
    features: new Float32Array([3, 4, 3, 4, 3, 4, 3, 4]),
    featureWeight: new Float32Array([1, 2, 3, 4]),
    words: new Float32Array([1, 0, 0, 1, 0.6, 0.8]),
    gateWeight: new Float32Array([0, 0]),
    gateBias: new Float32Array([0]),
    fc1Weight: new Float32Array([1, 0]),
    fc1Bias: new Float32Array([-1]),
    fc2Weight: new Float32Array([1, -1]),
    fc2Bias: new Float32Array([0, 1]),
  }
}

describe('cpu model', () => {
  it('computes the cosines of a small model', () => {
    const scored = new CpuModel(tinyModel()).run([
      {
        words: ['any'],
        from: 'any',
      },
      {
        words: ['any', 'other'],
        from: 'any other',
      },
    ])
    // The hidden unit is 3 - 1 = 2, so the output is [3 + 2, 4 - 2 + 1] = [5, 3]
    const length = Math.hypot(5, 3)
    const expected = [
      ['between', (5 * 0.6 + 3 * 0.8) / length],
      ['east', 5 / length],
      ['north', 3 / length],
    ] as const
    expect(scored.map(({ unit, word }) => [unit, word])).toEqual([0, 1].flatMap(unit => expected.map(([word]) => [unit, word])))
    scored.forEach(({ score }, index) => expect(score).toBeCloseTo(expected[index % 3]![1], 5))
  })
})

describe('cpu backend', () => {
  const file = readFileSync(new URL('../weights/model.bin', import.meta.url))
  const weights = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)

  it('predicts with the published weights and no GPU', async () => {
    const predictor = await createPredictor({ weights, backend: 'cpu' })
    expect(predictor.backend).toBe('cpu')
    const words = async (query: string): Promise<string[]> => (await predictor.predict(query)).map(prediction => prediction.word)
    expect(await words('movie')).toEqual(expect.arrayContaining(['cinema', 'film', 'video']))
    expect(await words('hom')).toEqual(expect.arrayContaining(['house', 'property', 'roof']))
    expect((await predictor.predict('calandar'))[0]).toMatchObject({ word: 'calendar', kind: 'correction' })
    expect(await predictor.predict('')).toEqual([])
    predictor.dispose()
  })

  it('falls back to the CPU where WebGPU is missing', async () => {
    expect((await createPredictor({ weights })).backend).toBe('cpu')
    await expect(createPredictor({ weights, backend: 'webgpu' })).rejects.toThrow('WebGPU')
  })
})
