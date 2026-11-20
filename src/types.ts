/**
 * How a predicted word relates to the typed word.
 *
 * - `completion`: the typed word is the start of the predicted word
 * - `correction`: the typed word is a misspelling of the predicted word
 * - `related`: the predicted word has a related meaning
 */
export type PredictionKind = 'completion' | 'correction' | 'related'

export interface Prediction {
  /** A word from the built-in word list */
  word: string
  /** From 0 to 1, higher is closer */
  score: number
  /** The typed word that produced this prediction, or the whole query */
  from: string
  kind: PredictionKind
}
