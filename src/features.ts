/** Longest word that the model reads. Longer words are cut. */
const MAX_WORD_LENGTH = 24

/** Most words of a query that the model reads */
export const MAX_QUERY_WORDS = 12

/** Feature kinds. Features of different kinds never share a hash. */
const KIND_WORD = 0
const KIND_PREFIX = 1
const KIND_PIECE = 3

/** A word needs this many letters before it gets features with one letter removed */
const MIN_DELETION_LENGTH = 3
const PIECE_LENGTH = 3

/** Most features that one word can have */
export const MAX_WORD_FEATURES = 3 * MAX_WORD_LENGTH

/** Split a query into lowercase words of letters and digits */
export function splitQuery(query: string): string[] {
  return (query.toLowerCase().match(/[a-z0-9]+/g) ?? [])
    .slice(0, MAX_QUERY_WORDS)
    .map(word => word.slice(0, MAX_WORD_LENGTH))
}

/**
 * Hash the characters from `start` to `end` of a text to a 32-bit number, without the character at `skip`.
 * FNV-1a over the character codes, then the final mix of MurmurHash3.
 * The Python trainer has the same function, and a test compares both.
 */
function hashText(text: string, start: number, end: number, kind: number, skip = -1): number {
  let hash = Math.imul(0x811C9DC5 ^ kind, 0x01000193)
  for (let i = start; i < end; i++) {
    if (i !== skip)
      hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193)
  }
  hash ^= hash >>> 16
  hash = Math.imul(hash, 0x85EBCA6B)
  hash ^= hash >>> 13
  hash = Math.imul(hash, 0xC2B2AE35)
  return (hash ^ (hash >>> 16)) >>> 0
}

/**
 * Turn one word into rows of the feature table. The features are:
 *
 * - the whole word, and the word with each single letter removed.
 *   A word and its misspelling share at least one of these.
 * - every prefix of two or more letters. A half-typed word shares these with the full word.
 * - every piece of three characters of `<word>`.
 *
 * @param word A word from `splitQuery`
 * @param rows Number of rows in the feature table, a power of two
 */
export function wordFeatures(word: string, rows: number): number[] {
  const mask = rows - 1
  const features = [hashText(word, 0, word.length, KIND_WORD) & mask]
  if (word.length >= MIN_DELETION_LENGTH) {
    for (let skip = 0; skip < word.length; skip++)
      features.push(hashText(word, 0, word.length, KIND_WORD, skip) & mask)
  }
  for (let end = 2; end <= word.length; end++)
    features.push(hashText(word, 0, end, KIND_PREFIX) & mask)
  const padded = `<${word}>`
  for (let start = 0; start + PIECE_LENGTH <= padded.length; start++)
    features.push(hashText(padded, start, start + PIECE_LENGTH, KIND_PIECE) & mask)
  return features
}
