import type { Predictor } from '../search'
import type { BenchCase, BenchSet } from './cases'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import { loadIconSets } from '../iconify'
import { cacheDir, readJSON, readJSONL, writeJSON } from '../paths'
import { IconSearch } from '../search'
import { buildWordTable, collectWordGroups, tablePredictor } from '../table'
import { readTrainingTags } from '../tags'
import { benchFile, loadAliasGroups } from './build'
import { isCorrect } from './cases'

/** A result counts as found if it is in the first results that fit on one screen */
export const SCREEN = 50

/** Rank that marks a case with no correct result */
const NOT_FOUND = Number.POSITIVE_INFINITY

const BENCH_SETS: BenchSet[] = [
  'alias',
  'transfer',
  'typing',
  'written',
]

export interface Method {
  name: string
  search: (icons: IconSearch, query: string) => string[]
}

export interface Score {
  cases: number
  /**
   * Share of cases with a correct icon on the first screen.
   * The matches of the typed text and the other matches count as two lists,
   * because the reference integration shows the matches of the typed text first.
   */
  found: number
  /** Mean of 1 divided by the rank of the first correct icon, in the better of the two lists */
  mrr: number
}

export type Report = Record<string, Partial<Record<BenchSet | 'overall', Score>>>

export function loadCases(): BenchCase[] {
  return BENCH_SETS.filter(set => existsSync(benchFile(set))).flatMap(set => readJSONL<BenchCase>(benchFile(set)))
}

/** Create the baseline methods that need no model */
export function baselineMethods(): Method[] {
  const aliasGroups = loadAliasGroups()
  const vocabulary = new Set(readJSON<string[]>(join(cacheDir, 'vocabulary.json')))
  const table = tablePredictor(buildWordTable(collectWordGroups(readTrainingTags()), vocabulary))
  return [
    {
      name: 'fzf',
      search: (icons, query) => icons.icones(query),
    },
    {
      name: 'fzf+alias',
      search: (icons, query) => icons.icones(query, aliasGroups),
    },
    predictorMethod('fzf+table', table),
  ]
}

/** Wrap a predictor with the reference fzf integration */
export function predictorMethod(name: string, predict: Predictor): Method {
  return {
    name,
    search: (icons, query) => icons.tiered(query, predict(query)),
  }
}

function scoreCases(cases: BenchCase[], ranks: number[]): Score {
  const found = ranks.filter(rank => rank <= SCREEN).length
  const mrr = ranks.reduce((sum, rank) => sum + 1 / rank, 0)
  return { cases: cases.length, found: found / cases.length, mrr: mrr / cases.length }
}

function firstCorrect(icons: string[], test: BenchCase): number {
  const index = icons.findIndex(name => isCorrect(name, test))
  return index < 0
    ? NOT_FOUND
    : index + 1
}

/**
 * Run each method on each case, and score each test set.
 * If `ranks` is given, it receives the rank of the first correct icon of each case, per method.
 */
export function runBenchmark(methods: Method[], cases: BenchCase[] = loadCases(), ranks?: Map<string, number[]>): Report {
  const searches = new Map<string, IconSearch>()
  for (const set of loadIconSets()) {
    if (cases.some(test => test.prefix === set.prefix))
      searches.set(set.prefix, new IconSearch(set.icons))
  }
  const report: Report = {}
  for (const method of methods) {
    const results = new Map<string, { all: string[], related: string[] }>()
    const found = cases.map((test) => {
      const key = `${test.prefix}\n${test.query}`
      let lists = results.get(key)
      if (!lists) {
        const search = searches.get(test.prefix)!
        const typed = new Set(search.icones(test.query))
        const all = method.search(search, test.query)
        results.set(key, lists = {
          all,
          related: all.filter(name => !typed.has(name)),
        })
      }
      return Math.min(firstCorrect(lists.all, test), firstCorrect(lists.related, test))
    })
    ranks?.set(method.name, found)
    const scores: Report[string] = {}
    for (const set of BENCH_SETS) {
      const index = cases.flatMap((test, i) => test.set === set
        ? [i]
        : [])
      if (index.length > 0)
        scores[set] = scoreCases(index.map(i => cases[i]!), index.map(i => found[i]!))
    }
    const parts = Object.values(scores)
    scores.overall = {
      cases: cases.length,
      found: parts.reduce((sum, score) => sum + score.found, 0) / parts.length,
      mrr: parts.reduce((sum, score) => sum + score.mrr, 0) / parts.length,
    }
    report[method.name] = scores
  }
  return report
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function printReport(report: Report): void {
  const sets = [...BENCH_SETS, 'overall' as const].filter(set => Object.values(report).some(scores => scores[set]))
  console.log(`| Method | ${sets.map(set => `${set} found | ${set} MRR`).join(' | ')} |`)
  console.log(`| --- | ${sets.map(() => '---: | ---:').join(' | ')} |`)
  for (const [name, scores] of Object.entries(report))
    console.log(`| \`${name}\` | ${sets.map(set => `${percent(scores[set]!.found)} | ${scores[set]!.mrr.toFixed(3)}`).join(' | ')} |`)
}

function main(): void {
  const started = performance.now()
  const report = runBenchmark(baselineMethods())
  printReport(report)
  writeJSON(join(cacheDir, 'bench', 'baselines.json'), report)
  console.log(`\n${loadCases().length} cases in ${((performance.now() - started) / 1000).toFixed(1)} s`)
}

if (import.meta.main)
  main()
