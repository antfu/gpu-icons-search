import type { Prediction } from './types'
import { splitQuery } from './features'

/** For each common query word, the words that the model predicts for it, best first */
export type AliasTable = Record<string, string[]>

/**
 * Download the static alias table that ships with the package.
 * It holds the predictions of the model for common words, for browsers without WebGPU.
 */
export async function loadAliasTable(url: URL | string = new URL('./aliases.json', import.meta.url)): Promise<AliasTable> {
  const response = await fetch(url)
  if (!response.ok)
    throw new Error(`Download of the alias table failed with HTTP ${response.status}`)
  return await response.json() as AliasTable
}

/**
 * Look up each typed word in the alias table.
 * This needs no GPU, but it knows only whole words: no typos, no half-typed words, no descriptions.
 */
export function predictFromTable(table: AliasTable, query: string, limit = 8): Prediction[] {
  const words = splitQuery(query)
  const predictions: Prediction[] = []
  for (const from of new Set(words)) {
    const related = (table[from] ?? []).filter(word => !words.includes(word))
    related.forEach((word, rank) => predictions.push({
      word,
      score: 1 - rank / (related.length + 1),
      from,
      kind: 'related',
    }))
  }
  return predictions.sort((a, b) => b.score - a.score).slice(0, limit)
}
