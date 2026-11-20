import type { Prediction, PredictionKind } from './types'

/** A part of the query that the model reads as one unit */
export interface QueryUnit {
  /** The words of the unit */
  words: string[]
  /** The typed text that predictions of this unit replace */
  from: string
}

/** A word of the word list that the model scored for a unit */
export interface ScoredWord {
  /** Index of the unit in the list from `queryUnits` */
  unit: number
  word: string
  /** Cosine between the unit and the word */
  score: number
}

/**
 * Split the words of a query into units.
 * The first unit is the whole query. A query of several words also gets one unit per word,
 * so that a prediction can replace one typed word and keep the others.
 */
export function queryUnits(words: string[]): QueryUnit[] {
  if (words.length === 0)
    return []
  const whole: QueryUnit = {
    words,
    from: words.join(' '),
  }
  return words.length === 1
    ? [whole]
    : [
        whole,
        ...[
          ...new Set(words),
        ].map(word => ({
          words: [word],
          from: word,
        })),
      ]
}

/** Tell if two words differ by at most `limit` single-letter edits or swaps of neighbors */
function isWithinEdits(a: string, b: string, limit: number): boolean {
  if (Math.abs(a.length - b.length) > limit)
    return false
  let previous2: number[] = []
  let previous = Array.from({ length: b.length + 1 }, (_, j) => j)
  for (let i = 1; i <= a.length; i++) {
    const current = [i]
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1]
        ? 0
        : 1
      let best = Math.min(previous[j]! + 1, current[j - 1]! + 1, previous[j - 1]! + cost)
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        best = Math.min(best, previous2[j - 2]! + 1)
      current.push(best)
    }
    previous2 = previous
    previous = current
  }
  return previous[b.length]! <= limit
}

function kindOf(from: string, word: string): PredictionKind {
  if (from.includes(' '))
    return 'related'
  if (word.startsWith(from))
    return 'completion'
  return isWithinEdits(from, word, from.length >= 8
    ? 2
    : 1)
    ? 'correction'
    : 'related'
}

/**
 * Turn scored words into predictions, best first.
 * A word that the user typed is not a prediction. A word that several units predict counts once.
 */
export function decodePredictions(units: QueryUnit[], scored: ScoredWord[], limit: number, minScore: number): Prediction[] {
  const typed = new Set(units.flatMap(unit => unit.words))
  const best = new Map<string, Prediction>()
  for (const { unit, word, score } of scored) {
    if (score < minScore || typed.has(word) || (best.get(word)?.score ?? -1) >= score)
      continue
    const from = units[unit]!.from
    best.set(word, {
      word,
      score: Math.min(1, score),
      from,
      kind: kindOf(from, word),
    })
  }
  return [
    ...best.values(),
  ].sort((a, b) => b.score - a.score).slice(0, limit)
}
