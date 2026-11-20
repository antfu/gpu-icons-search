import type { Prediction } from '../../src/types'
import type { Predictor } from './search'
import type { TaggedIcon } from './tags'
import { loadIconSets, splitWords } from './iconify'

export interface RelatedWord {
  word: string
  /** From 0 to 1 */
  score: number
}

/** For each word, the words that occur together with it, best first */
export type WordTable = Map<string, RelatedWord[]>

/** A pair of words must occur together this many times to count */
const MIN_PAIR_COUNT = 2

/**
 * Collect the groups of words that describe the same icon.
 * A group holds the words of an icon name and the words of its tags.
 * An Iconify alias and the icon it points to also form a group.
 */
export function collectWordGroups(tags: TaggedIcon[]): string[][] {
  const groups = tags.map(icon => [
    ...splitWords(icon.name),
    ...icon.tags.flatMap(splitWords),
  ])
  for (const set of loadIconSets()) {
    for (const [alias, parent] of Object.entries(set.aliases)) {
      groups.push([
        ...splitWords(alias),
        ...splitWords(parent),
      ])
    }
  }
  return groups.map(group => [
    ...new Set(group.filter(word => /[a-z]/.test(word))),
  ])
}

interface PairCounts {
  /** Number of groups that contain the word */
  words: Map<string, number>
  /** For each word, the number of groups that it shares with each word of the vocabulary */
  pairs: Map<string, Map<string, number>>
}

function countPairs(groups: string[][], vocabulary: Set<string>): PairCounts {
  const words = new Map<string, number>()
  const pairs = new Map<string, Map<string, number>>()
  for (const group of groups) {
    const targets = group.filter(word => vocabulary.has(word))
    for (const a of group) {
      words.set(a, (words.get(a) ?? 0) + 1)
      let row = pairs.get(a)
      if (!row)
        pairs.set(a, row = new Map())
      for (const b of targets)
        row.set(b, (row.get(b) ?? 0) + 1)
    }
  }
  return { words, pairs }
}

/**
 * Count which words occur together, and keep the strongest pairs.
 * The score of a pair is its count divided by the geometric mean of the two word counts.
 */
export function buildWordTable(groups: string[][], vocabulary: Set<string>, limit = 24): WordTable {
  const { words, pairs } = countPairs(groups, vocabulary)
  const table: WordTable = new Map()
  for (const [a, row] of pairs) {
    const related = [
      ...row.entries(),
    ]
      .filter(([b, count]) => a !== b && count >= MIN_PAIR_COUNT)
      .map(([b, count]) => ({ word: b, score: count / Math.sqrt(words.get(a)! * words.get(b)!) }))
      .sort((x, y) => y.score - x.score)
      .slice(0, limit)
    if (related.length > 0)
      table.set(a, related)
  }
  return table
}

/** Use a word table as a predictor: look up each typed word, with no model */
export function tablePredictor(table: WordTable, limit = 8): Predictor {
  return (query) => {
    const predictions: Prediction[] = []
    for (const from of new Set(splitWords(query))) {
      for (const { word, score } of table.get(from) ?? [])
        predictions.push({ word, score, from, kind: 'related' })
    }
    return predictions.sort((a, b) => b.score - a.score).slice(0, limit)
  }
}
