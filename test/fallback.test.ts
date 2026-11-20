import { describe, expect, it } from 'vitest'
import { predictFromTable } from '../src/fallback'

describe('predictFromTable', () => {
  const table = {
    home: ['house', 'building'],
    trash: ['delete', 'bin', 'home'],
  }

  it('looks up each typed word and ranks by position', () => {
    expect(predictFromTable(table, 'Home')).toEqual([
      {
        word: 'house',
        score: 1,
        from: 'home',
        kind: 'related',
      },
      {
        word: 'building',
        score: 1 - 1 / 3,
        from: 'home',
        kind: 'related',
      },
    ])
  })

  it('skips words that the user typed and respects the limit', () => {
    expect(predictFromTable(table, 'home trash', 3).map(prediction => prediction.word)).toEqual(['house', 'delete', 'building'])
  })

  it('returns nothing for unknown words', () => {
    expect(predictFromTable(table, 'hom')).toEqual([])
  })
})
