import type { Prediction } from '../../src/types'
import { extendedMatch, Fzf } from 'fzf'

/** A function that turns a query into predicted words, best first */
export type Predictor = (query: string) => Prediction[]

/** An icon name with the prediction that found it. A match of the typed text has no prediction. */
export interface SearchResult {
  icon: string
  via?: Prediction
}

/**
 * A correction or a completion stands for the word that the user meant to type.
 * Its matches belong with the matches of the typed text, before the matches of related words.
 */
export function isTypedWord(prediction: Prediction): boolean {
  return prediction.kind !== 'related'
}

/**
 * Text search over the icon names of one icon set.
 * It holds the reference integration of predictions with fzf,
 * and a replica of the current search logic of Icones.
 */
export class IconSearch {
  private readonly fuzzy: Fzf<string[]>
  private readonly extended: Fzf<string[]>
  private readonly cache = new Map<string, string[]>()

  constructor(readonly icons: string[]) {
    this.fuzzy = new Fzf(icons, { casing: 'case-insensitive', fuzzy: 'v1' })
    this.extended = new Fzf(icons, { casing: 'case-insensitive', match: extendedMatch })
  }

  private find(pattern: string, extended: boolean): string[] {
    const key = `${extended ? 'e' : 'f'}:${pattern}`
    let result = this.cache.get(key)
    if (!result) {
      result = (extended ? this.extended : this.fuzzy).find(pattern).map(entry => entry.item)
      this.cache.set(key, result)
    }
    return result
  }

  /**
   * Replica of `useSearch` in Icones (src/hooks/search.ts) for a single icon set.
   * It joins the query and its alias variants into one fzf OR pattern.
   */
  icones(query: string, aliasGroups: string[][] = []): string[] {
    const parts = query.trim().toLowerCase().split(' ').filter(Boolean)
    const candidates = new Set([
      parts.join(' '),
    ])
    parts.forEach((part, index) => {
      const group = aliasGroups.find(words => words.includes(part))
      for (const alias of group ?? []) {
        // Icones joins the rest of the query with commas here. The replica keeps that behavior.
        const rest = parts.slice(index + 1).join(',')
        candidates.add([
          ...parts.slice(0, index),
          alias,
          rest,
        ].join(' ').trim())
      }
    })
    const extended = /[ '^$!]/.test(query) || candidates.size > 1
    return this.find([...candidates].join(' | '), extended)
  }

  /**
   * The reference integration. The typed text comes first, with fuzzy matching as in Icones.
   * Then each predicted word follows as an exact match: corrections and completions first,
   * then related words, each group in score order.
   * Each result names the prediction that found it. A match of the typed text has none.
   */
  tieredWithReasons(query: string, predictions: Prediction[]): SearchResult[] {
    const parts = query.trim().toLowerCase().split(' ').filter(Boolean)
    const merged: SearchResult[] = this.icones(query).map(icon => ({ icon }))
    const seen = new Set(merged.map(result => result.icon))
    const ordered = [
      ...predictions.filter(isTypedWord),
      ...predictions.filter(prediction => !isTypedWord(prediction)),
    ]
    for (const prediction of ordered) {
      const index = parts.indexOf(prediction.from)
      const pattern = index < 0
        ? `'${prediction.word}`
        : parts.map((part, i) => i === index
            ? `'${prediction.word}`
            : part).join(' ')
      for (const icon of this.find(pattern, true)) {
        if (!seen.has(icon)) {
          seen.add(icon)
          merged.push({ icon, via: prediction })
        }
      }
    }
    return merged
  }

  /** The icon names of `tieredWithReasons`, in the same order */
  tiered(query: string, predictions: Prediction[]): string[] {
    return this.tieredWithReasons(query, predictions).map(result => result.icon)
  }
}
