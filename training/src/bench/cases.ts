export type BenchSet = 'alias' | 'transfer' | 'typing' | 'written'

export interface BenchCase {
  set: BenchSet
  /** Iconify prefix of the icon set that the query runs on */
  prefix: string
  query: string
  /** A result is correct if its name contains this word, or one of these words */
  word?: string | string[]
  /** A result is correct if it is this icon or a style variant of it */
  icon?: string
}

/** Tell if an icon name contains a word or a hyphenated fragment as a whole */
export function hasWord(name: string, word: string): boolean {
  return `-${name}-`.includes(`-${word}-`)
}

export function isCorrect(name: string, test: BenchCase): boolean {
  if (test.icon)
    return name === test.icon || name.startsWith(`${test.icon}-`)
  const words = typeof test.word === 'string'
    ? [test.word]
    : test.word ?? []
  return words.some(word => hasWord(name, word))
}

/** Small seeded random generator, so that the test sets are reproducible */
export function createRandom(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6D2B79F5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function sample<T>(items: T[], count: number, random: () => number): T[] {
  const pool = [...items]
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[pool[i], pool[j]] = [pool[j]!, pool[i]!]
  }
  return pool.slice(0, count)
}
