import { describe, expect, it } from 'vitest'
import { MAX_WORD_FEATURES, splitQuery, wordFeatures } from '../src/features'

describe('splitQuery', () => {
  it('splits on everything that is not a letter or a digit', () => {
    expect(splitQuery('  Arrow-Up_right  2x!')).toEqual(['arrow', 'up', 'right', '2x'])
  })

  it('cuts long words and long queries', () => {
    expect(splitQuery('a'.repeat(40))).toEqual([
      'a'.repeat(24),
    ])
    expect(splitQuery('a b c d e f g h i j k l m n o')).toHaveLength(12)
  })
})

describe('wordFeatures', () => {
  it('returns stable rows inside the table', () => {
    expect(wordFeatures('home', 16384)).toMatchInlineSnapshot(`
      [
        6535,
        11704,
        15250,
        2655,
        3694,
        15790,
        12785,
        7683,
        3290,
        1956,
        2323,
        10908,
        14173,
      ]
    `)
    expect(wordFeatures('a'.repeat(24), 1024).length).toBe(MAX_WORD_FEATURES)
  })

  it('shares most features between a word and its prefix', () => {
    const full = new Set(wordFeatures('home', 1 << 20))
    const shared = wordFeatures('hom', 1 << 20).filter(row => full.has(row))
    expect(shared.length).toBe(6)
  })
})
