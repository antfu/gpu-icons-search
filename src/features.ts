/** Longest word that the model reads. Longer words are cut. */
const MAX_WORD_LENGTH = 24

/** Most words of a query that the model reads */
const MAX_QUERY_WORDS = 12

/** Lengths of the character pieces of a word */
const PIECE_LENGTHS = [2, 3, 4] as const

/** Most features that one word can have */
export const MAX_WORD_FEATURES = PIECE_LENGTHS.reduce((sum, n) => sum + MAX_WORD_LENGTH + 3 - n, 1)

/** Split a query into lowercase words of letters and digits */
export function splitQuery(query: string): string[] {
  return (query.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .slice(0, MAX_QUERY_WORDS)
    .map(word => word.slice(0, MAX_WORD_LENGTH))
}

/**
 * Hash a character piece to a 32-bit number.
 * FNV-1a over the piece length and the character codes, then the final mix of MurmurHash3.
 * The Python trainer has the same function, and a test compares both.
 */
function hashPiece(text: string, start: number, length: number, kind: number): number {
  let hash = Math.imul(0x811C9DC5 ^ kind, 0x01000193)
  for (let i = start; i < start + length; i++)
    hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193)
  hash ^= hash >>> 16
  hash = Math.imul(hash, 0x85EBCA6B)
  hash ^= hash >>> 13
  hash = Math.imul(hash, 0xC2B2AE35)
  return (hash ^ (hash >>> 16)) >>> 0
}

/**
 * Turn one word into rows of the feature table.
 * The features are the whole word and every piece of 2 to 4 characters of `<word>`.
 * Pieces let a half-typed or misspelled word share most features with the full word.
 *
 * @param word A word from `splitQuery`
 * @param rows Number of rows in the feature table, a power of two
 */
export function wordFeatures(word: string, rows: number): number[] {
  const text = `<${word}>`
  const features = [hashPiece(text, 0, text.length, 0) & (rows - 1)]
  for (const length of PIECE_LENGTHS) {
    for (let start = 0; start + length <= text.length; start++)
      features.push(hashPiece(text, start, length, length) & (rows - 1))
  }
  return features
}
