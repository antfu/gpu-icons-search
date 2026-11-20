import type { BenchCase } from './cases'
import { join } from 'node:path'
import { loadIconSets, splitWords } from '../iconify'
import { dataDir, readJSON, writeJSONL } from '../paths'
import { HELD_OUT_PREFIX, HELD_OUT_SOURCES, readTags } from '../tags'
import { createRandom, hasWord, sample } from './cases'

/** Icon sets that the `alias` test set runs on. They use different naming styles. */
export const BENCH_PREFIXES = [
  'mdi',
  'material-symbols',
  'lucide',
  'tabler',
  'ph',
  'bi',
  'carbon',
  'heroicons',
  'ri',
  'fa6-solid',
  'ion',
  'solar',
  'fluent',
  'iconoir',
]

const TRANSFER_CASES_PER_SOURCE = 1500
const TYPING_SOURCE_CASES = 1000
const LETTERS = 'abcdefghijklmnopqrstuvwxyz'

export function benchFile(set: string): string {
  return join(dataDir, 'bench', `${set}.jsonl`)
}

export function loadAliasGroups(): string[][] {
  return readJSON<{ groups: string[][] }>(join(dataDir, 'icones-alias.json')).groups
}

/** One case per icon set and ordered pair of words in a manual alias group */
function buildAliasCases(icons: Map<string, string[]>): BenchCase[] {
  const cases: BenchCase[] = []
  for (const prefix of BENCH_PREFIXES) {
    const names = icons.get(prefix)!
    for (const group of loadAliasGroups()) {
      const present = group.filter(word => names.some(name => hasWord(name, word)))
      for (const query of group) {
        for (const word of present.filter(word => word !== query))
          cases.push({ set: 'alias', prefix, query, word })
      }
    }
  }
  return cases
}

/** A tag is a fair query if the icon name contains none of its words */
function isHiddenTag(name: string, tag: string): boolean {
  const words = splitWords(tag)
  return /^[a-z0-9 -]+$/.test(tag)
    && words.some(word => word.length >= 3)
    && words.every(word => !name.includes(word))
}

/** Tags of icon sets that training never reads. The query is the tag, and the answer is its icon. */
function buildTransferCases(icons: Map<string, string[]>, random: () => number): BenchCase[] {
  return HELD_OUT_SOURCES.flatMap((source) => {
    const prefix = HELD_OUT_PREFIX[source]!
    const names = new Set(icons.get(prefix))
    const cases: BenchCase[] = readTags(source)
      .filter(icon => names.has(icon.name))
      .flatMap(icon => icon.tags
        .filter(tag => isHiddenTag(icon.name, tag))
        .map(tag => ({
          set: 'transfer' as const,
          prefix,
          query: tag,
          icon: icon.name,
        })))
    return sample(cases, TRANSFER_CASES_PER_SOURCE, random)
  })
}

function pick(text: string, random: () => number): number {
  return Math.floor(random() * text.length)
}

/** Apply one typing mistake to a word: swap, delete, replace, or insert one letter */
export function addTypo(word: string, random: () => number): string {
  const i = pick(word, random)
  const letter = LETTERS[pick(LETTERS, random)]!
  const variants = [
    i + 1 < word.length
      ? word.slice(0, i) + word[i + 1] + word[i] + word.slice(i + 2)
      : word,
    word.slice(0, i) + word.slice(i + 1),
    word.slice(0, i) + letter + word.slice(i + 1),
    word.slice(0, i) + letter + word.slice(i),
  ].filter(variant => variant !== word)
  return variants[pick(variants.join(''), random) % variants.length] ?? word
}

/** Replace the longest word of a query */
function changeLongestWord(query: string, change: (word: string) => string): string | undefined {
  const longest = splitWords(query).sort((a, b) => b.length - a.length)[0]
  if (!longest || longest.length < 4)
    return undefined
  return query.replace(longest, change(longest))
}

/** Half-typed and misspelled variants of queries from the other sets, with the same answers */
function buildTypingCases(sources: BenchCase[], random: () => number): BenchCase[] {
  const cases: BenchCase[] = []
  for (const source of sample(sources, TYPING_SOURCE_CASES, random)) {
    const prefix = changeLongestWord(source.query, word => word.slice(0, Math.max(3, Math.ceil(word.length * 0.6))))
    const typo = changeLongestWord(source.query, word => addTypo(word, random))
    for (const query of [prefix, typo]) {
      if (query && query !== source.query)
        cases.push({ ...source, set: 'typing', query })
    }
  }
  return cases
}

function main(): void {
  const icons = new Map(loadIconSets().map(set => [set.prefix, set.icons]))
  const random = createRandom(20261120)
  const alias = buildAliasCases(icons)
  const transfer = buildTransferCases(icons, random)
  const typing = buildTypingCases([
    ...sample(alias, TYPING_SOURCE_CASES, random),
    ...transfer,
  ], random)
  for (const [set, cases] of Object.entries({ alias, transfer, typing })) {
    writeJSONL(benchFile(set), cases)
    console.log(`${set.padEnd(9)} ${cases.length} cases`)
  }
}

if (import.meta.main)
  main()
