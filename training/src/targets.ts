import type { RelatedWord, WordTable } from './table'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { splitWords } from './iconify'
import { cacheDir, readJSON, writeJSON } from './paths'
import { buildWordTable, collectWordGroups } from './table'
import { readTrainingTags } from './tags'
import { embed, nearest } from './teacher'

/** The teacher model that the comparison in `bench/teachers.ts` selected */
export const TEACHER = 'Xenova/bge-small-en-v1.5'

/** Words that name a drawing style. They say nothing about the meaning of an icon. */
export const STYLE_WORDS = new Set([
  'outline',
  'outlined',
  'fill',
  'filled',
  'solid',
  'line',
  'linear',
  'bold',
  'thin',
  'light',
  'regular',
  'duotone',
  'twotone',
  'sharp',
  'round',
  'rounded',
  'broken',
  'bulk',
  'duocolor',
  'tone',
  'alt',
  'variant',
])

const TABLE_NEIGHBORS = 32
const TEACHER_NEIGHBORS = 12
const TARGETS_PER_QUERY = 32

export interface TargetOptions {
  /** Weight of the best teacher neighbor, relative to the best neighbor from the tags */
  teacherWeight: number
  /** Exponent on the relative tag score. A smaller value lifts weaker neighbors. */
  tagPower: number
  /** Weight of the neighbors of neighbors, relative to direct neighbors. 0 turns them off. */
  secondHopWeight: number
}

export const DEFAULT_TARGET_OPTIONS: TargetOptions = { teacherWeight: 0.25, tagPower: 0.5, secondHopWeight: 0 }

export interface TrainingQuery {
  /** One word, or a phrase of several words */
  text: string
  /** How often training samples this query, before normalization */
  weight: number
  /** Rows of the vocabulary with their target weights, best first */
  targets: [index: number, weight: number][]
}

export interface TrainingData {
  vocabulary: string[]
  queries: TrainingQuery[]
}

function addWeight(weights: Map<string, number>, word: string, weight: number): void {
  if (!STYLE_WORDS.has(word))
    weights.set(word, (weights.get(word) ?? 0) + weight)
}

/** Scale the scores of a neighbor list so that the best one is 1 */
function relative(neighbors: RelatedWord[]): RelatedWord[] {
  const best = neighbors[0]?.score ?? 1
  return neighbors.map(({ word, score }) => ({ word, score: score / best }))
}

/** For each word, its closest words of the vocabulary by teacher embedding. The result is kept in the cache folder. */
async function teacherTable(words: string[], vocabulary: string[]): Promise<WordTable> {
  const file = join(cacheDir, 'train', 'teacher-table.json')
  const key = createHash('sha256').update(JSON.stringify([
    TEACHER,
    TEACHER_NEIGHBORS,
    words,
    vocabulary,
  ])).digest('hex')
  if (existsSync(file)) {
    const cached = readJSON<{ key: string, table: [string, RelatedWord[]][] }>(file)
    if (cached.key === key)
      return new Map(cached.table)
  }
  const targets = await embed(TEACHER, vocabulary, 'vocabulary')
  const queries = await embed(TEACHER, words, 'queries')
  const table: WordTable = new Map()
  words.forEach((word, row) => {
    const neighbors = nearest(queries, row, targets, TEACHER_NEIGHBORS + 1)
      .filter(neighbor => vocabulary[neighbor.index] !== word)
      .slice(0, TEACHER_NEIGHBORS)
    // Similarities of single words sit in a narrow band. The rank spreads them out.
    table.set(word, neighbors.map((neighbor, rank) => ({ word: vocabulary[neighbor.index]!, score: 1 - rank / TEACHER_NEIGHBORS })))
  })
  writeJSON(file, {
    key,
    table: [...table],
  })
  return table
}

/** Count how often each phrase tag occurs with each word of an icon name */
function phraseTargets(vocabulary: Set<string>): Map<string, Map<string, number>> {
  const phrases = new Map<string, Map<string, number>>()
  for (const icon of readTrainingTags()) {
    const nameWords = splitWords(icon.name).filter(word => vocabulary.has(word))
    for (const tag of icon.tags) {
      const words = splitWords(tag)
      if (words.length < 2 || words.length > 5)
        continue
      const phrase = words.join(' ')
      let weights = phrases.get(phrase)
      if (!weights)
        phrases.set(phrase, weights = new Map())
      for (const word of nameWords)
        addWeight(weights, word, 1)
    }
  }
  return phrases
}

function toTargets(weights: Map<string, number>, index: Map<string, number>): TrainingQuery['targets'] {
  return [
    ...weights.entries(),
  ]
    .sort((a, b) => b[1] - a[1])
    .slice(0, TARGETS_PER_QUERY)
    .map(([word, weight]) => [
      index.get(word)!,
      Number(weight.toFixed(4)),
    ])
}

/**
 * Build the training targets of stage 1.
 * A single word points to itself, to the words that share icons with it, and to its teacher neighbors.
 * A phrase tag points to the name words of the icons that carry it.
 */
export async function buildTrainingData(options: TargetOptions = DEFAULT_TARGET_OPTIONS): Promise<TrainingData> {
  const vocabulary = readJSON<string[]>(join(cacheDir, 'vocabulary.json'))
  const inVocabulary = new Set(vocabulary)
  const index = new Map(vocabulary.map((word, i) => [word, i]))
  const groups = collectWordGroups(readTrainingTags())
  const groupCount = new Map<string, number>()
  for (const group of groups) {
    for (const word of group)
      groupCount.set(word, (groupCount.get(word) ?? 0) + 1)
  }
  const tagTable = buildWordTable(groups, inVocabulary, TABLE_NEIGHBORS)
  const words = [
    ...new Set([
      ...vocabulary,
      ...tagTable.keys(),
    ]),
  ]
  const teacher = await teacherTable(words, vocabulary)

  /** Direct neighbors of a word, from the tags and from the teacher */
  const direct = new Map(words.map((word) => {
    const weights = new Map<string, number>()
    for (const { word: related, score } of relative(tagTable.get(word) ?? []))
      addWeight(weights, related, score ** options.tagPower)
    for (const { word: related, score } of teacher.get(word) ?? [])
      addWeight(weights, related, options.teacherWeight * score)
    return [word, weights]
  }))

  const queries: TrainingQuery[] = words.map((word) => {
    const weights = new Map(direct.get(word))
    if (options.secondHopWeight > 0) {
      for (const [middle, first] of direct.get(word)!) {
        for (const [related, second] of direct.get(middle) ?? []) {
          if (related !== word)
            addWeight(weights, related, options.secondHopWeight * first * second)
        }
      }
    }
    if (inVocabulary.has(word))
      weights.set(word, Math.max(1, ...weights.values()))
    return {
      text: word,
      weight: Math.sqrt(1 + (groupCount.get(word) ?? 0)),
      targets: toTargets(weights, index),
    }
  })
  for (const [phrase, weights] of phraseTargets(inVocabulary)) {
    const best = Math.max(...weights.values())
    const scaled = new Map([...weights].map(([word, weight]) => [word, weight / best]))
    if (scaled.size > 0) {
      queries.push({
        text: phrase,
        weight: Math.sqrt(best),
        targets: toTargets(scaled, index),
      })
    }
  }
  return {
    vocabulary,
    queries: queries.filter(query => query.targets.length > 0),
  }
}

async function main(): Promise<void> {
  const data = await buildTrainingData()
  writeJSON(join(cacheDir, 'train', 'targets.json'), data)
  const phrases = data.queries.filter(query => query.text.includes(' ')).length
  console.log(`${data.queries.length - phrases} words and ${phrases} phrases point to ${data.vocabulary.length} target words`)
}

if (import.meta.main) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
