import { describe, expect, it } from 'vitest'
import { decodePredictions, queryUnits } from '../src/decode'

describe('queryUnits', () => {
  it('returns one unit for one word', () => {
    expect(queryUnits(['hom'])).toEqual([
      {
        words: ['hom'],
        from: 'hom',
      },
    ])
  })

  it('adds one unit per distinct word for a longer query', () => {
    expect(queryUnits(['user', 'settings', 'user'])).toEqual([
      {
        words: ['user', 'settings', 'user'],
        from: 'user settings user',
      },
      {
        words: ['user'],
        from: 'user',
      },
      {
        words: ['settings'],
        from: 'settings',
      },
    ])
  })

  it('returns nothing for an empty query', () => {
    expect(queryUnits([])).toEqual([])
  })
})

describe('decodePredictions', () => {
  it('labels completions, corrections and related words', () => {
    const units = queryUnits(['hoem'])
    const scored = [
      { unit: 0, word: 'home', score: 0.9 },
      { unit: 0, word: 'house', score: 0.7 },
      { unit: 0, word: 'hoemwork', score: 0.5 },
    ]
    expect(decodePredictions(units, scored, 8, 0.3)).toEqual([
      {
        word: 'home',
        score: 0.9,
        from: 'hoem',
        kind: 'correction',
      },
      {
        word: 'house',
        score: 0.7,
        from: 'hoem',
        kind: 'related',
      },
      {
        word: 'hoemwork',
        score: 0.5,
        from: 'hoem',
        kind: 'completion',
      },
    ])
  })

  it('drops typed words, weak words and duplicates', () => {
    const units = queryUnits(['user', 'settings'])
    const scored = [
      { unit: 0, word: 'settings', score: 0.9 },
      { unit: 0, word: 'profile', score: 0.6 },
      { unit: 1, word: 'account', score: 0.8 },
      { unit: 2, word: 'cog', score: 0.7 },
      { unit: 2, word: 'account', score: 0.4 },
      { unit: 1, word: 'weak', score: 0.1 },
    ]
    expect(decodePredictions(units, scored, 2, 0.3)).toEqual([
      {
        word: 'account',
        score: 0.8,
        from: 'user',
        kind: 'related',
      },
      {
        word: 'cog',
        score: 0.7,
        from: 'settings',
        kind: 'related',
      },
    ])
  })
})
