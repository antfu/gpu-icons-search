import type { Backend, Prediction, Predictor } from 'gpu-icons-search'
import type { AliasTable } from 'gpu-icons-search/fallback'
import type { BenchCase } from '../../training/src/bench/cases'
import type { Evaluation } from '../../training/src/evaluate'
import type { SweepEntry } from '../../training/src/sweep'
import { createPredictor } from 'gpu-icons-search'
import { loadAliasTable, predictFromTable } from 'gpu-icons-search/fallback'
import { IconSearch } from '../../training/src/search'

export interface IconSetInfo {
  prefix: string
  name: string
  total: number
}

/** How the playground gets predictions: from the model on WebGPU or on the CPU, from the alias table, or not at all */
type PredictMode = Backend | 'table' | 'none'

async function fetchJSON<T>(path: string): Promise<T | undefined> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${path}`)
  // The dev server answers a missing file with the HTML page
  if (!response.ok || !response.headers.get('content-type')?.includes('json'))
    return undefined
  return await response.json() as T
}

export async function loadIconSets(): Promise<IconSetInfo[]> {
  return await fetchJSON<IconSetInfo[]>('sets.json') ?? []
}

const searches = new Map<string, Promise<IconSearch>>()

/** Load the icon names of one icon set, and build its text search */
export function loadSearch(prefix: string): Promise<IconSearch> {
  let search = searches.get(prefix)
  if (!search) {
    search = fetchJSON<string[]>(`icons/${prefix}.json`).then(icons => new IconSearch(icons ?? []))
    searches.set(prefix, search)
  }
  return search
}

export function loadEvaluation(): Promise<Evaluation | undefined> {
  return fetchJSON<Evaluation>('evaluation.json')
}

export function loadSweep(): Promise<SweepEntry[]> {
  return fetchJSON<SweepEntry[]>('sweep.json').then(entries => entries ?? [])
}

export function loadWrittenCases(): Promise<BenchCase[]> {
  return fetchJSON<BenchCase[]>('written.json').then(cases => cases ?? [])
}

export interface PredictSource {
  mode: PredictMode
  /** Why the model is not in use, if it is not */
  note?: string
  predict: (query: string, limit: number, signal?: AbortSignal) => Promise<Prediction[]>
}

async function tableSource(note: string): Promise<PredictSource> {
  let table: AliasTable | undefined
  try {
    table = await loadAliasTable(`${import.meta.env.BASE_URL}data/aliases.json`)
  }
  catch {
    return {
      mode: 'none',
      note: `${note} No alias table is available either.`,
      predict: async () => [],
    }
  }
  const aliases = table
  return {
    mode: 'table',
    note,
    predict: async (query, limit) => predictFromTable(aliases, query, limit),
  }
}

/** Load the model. If it does not load, fall back to the static alias table. */
export async function loadPredictSource(): Promise<PredictSource> {
  let predictor: Predictor
  try {
    predictor = await createPredictor({ weights: `${import.meta.env.BASE_URL}data/model.bin` })
  }
  catch (error) {
    return tableSource(`The model did not load: ${String(error)}`)
  }
  return {
    mode: predictor.backend,
    predict: (query, limit, signal) => predictor.predict(query, { limit, signal }),
  }
}
