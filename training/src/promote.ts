import type { AliasTable } from '../../src/fallback'
import type { BenchSet } from './bench/cases'
import type { Evaluation } from './evaluate'
import { createHash } from 'node:crypto'
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import { MODEL_METHOD } from './evaluate'
import { cacheDir, readJSON, trainingRoot, writeJSON } from './paths'
import { loadModel, referencePredictor } from './reference'

/** The download limit: the compressed weight file must stay below this */
const MAX_COMPRESSED_BYTES = 1024 * 1024

/** Words in the static alias table, most frequent first */
const ALIAS_TABLE_WORDS = 4000
const ALIASES_PER_WORD = 6

/** The baseline that a candidate must beat */
const BASELINE = 'fzf+alias'

/**
 * The `alias` test set is the alias list that `fzf+alias` itself uses, so that baseline holds the answers.
 * The maintainer decided on 2026-11-20 to compare this set with the tag table instead.
 */
const ALIAS_BASELINE = 'fzf+table'

/** Folder of the published weights, at the root of the repository */
const weightsDir = join(trainingRoot, '..', 'weights')
const qualityFile = join(trainingRoot, 'quality.json')

/** Scores of the last promoted model. A new candidate must not score lower. */
type Floors = Partial<Record<BenchSet | 'overall', number>>

interface Gate {
  rule: string
  pass: boolean
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

function checkGates(evaluation: Evaluation, floors: Floors): Gate[] {
  const model = evaluation.bench[MODEL_METHOD]!
  const gates: Gate[] = []
  for (const [set, score] of Object.entries(model) as [BenchSet | 'overall', { found: number }][]) {
    const name = set === 'alias'
      ? ALIAS_BASELINE
      : BASELINE
    const target = evaluation.bench[name]![set]!.found
    gates.push(set === 'overall'
      ? { rule: `overall: ${percent(score.found)} is above ${name} at ${percent(target)}`, pass: score.found > target }
      : { rule: `${set}: ${percent(score.found)} is not below ${name} at ${percent(target)}`, pass: score.found >= target })
    const floor = floors[set]
    if (floor !== undefined)
      gates.push({ rule: `${set}: ${percent(score.found)} is not below the last promoted model at ${percent(floor)}`, pass: score.found >= floor })
  }
  const { reference, webgpu, tolerance } = evaluation.parity
  gates.push({ rule: `WebGPU output matches the PyTorch fixtures (${webgpu.toExponential(1)} within ${tolerance})`, pass: webgpu <= tolerance })
  gates.push({ rule: `reference output matches the PyTorch fixtures (${reference.toExponential(1)} within ${tolerance})`, pass: reference <= tolerance })
  gates.push({ rule: `compressed size ${evaluation.bytes.brotli} bytes is within ${MAX_COMPRESSED_BYTES}`, pass: evaluation.bytes.brotli <= MAX_COMPRESSED_BYTES })
  return gates
}

/** Run the model on its most frequent words, for browsers that cannot run it themselves */
function buildAliasTable(dir: string): AliasTable {
  const model = loadModel(dir)
  const predict = referencePredictor(model, { limit: ALIASES_PER_WORD + 4 })
  const table: AliasTable = {}
  for (const word of model.vocabulary.slice(0, ALIAS_TABLE_WORDS)) {
    const related = predict(word).filter(prediction => prediction.kind === 'related').slice(0, ALIASES_PER_WORD)
    if (related.length > 0)
      table[word] = related.map(prediction => prediction.word)
  }
  return table
}

/**
 * Copy a candidate into the published package, if it passes every rule of the ship gate.
 * Run `evaluate` on the candidate first.
 * Usage: tsx src/promote.ts [candidate folder]
 */
function main(): void {
  const dir = process.argv[2] ?? join(cacheDir, 'candidate')
  const evaluationFile = join(dir, 'evaluation.json')
  if (!existsSync(evaluationFile))
    throw new Error(`No evaluation.json in ${dir}. Run the evaluate script first.`)
  const evaluation = readJSON<Evaluation>(evaluationFile)
  const sha256 = createHash('sha256').update(readFileSync(join(dir, 'model.bin'))).digest('hex')
  if (sha256 !== evaluation.sha256)
    throw new Error('The weight file changed after the evaluation. Run the evaluate script again.')

  const gates = checkGates(evaluation, readJSON<Floors>(qualityFile))
  for (const gate of gates)
    console.log(`${gate.pass ? 'pass' : 'FAIL'}  ${gate.rule}`)
  if (gates.some(gate => !gate.pass)) {
    console.error('\nThe candidate is not promoted.')
    process.exitCode = 1
    return
  }

  mkdirSync(weightsDir, { recursive: true })
  copyFileSync(join(dir, 'model.bin'), join(weightsDir, 'model.bin'))
  writeJSON(join(weightsDir, 'aliases.json'), buildAliasTable(dir))
  const training = existsSync(join(dir, 'report.json'))
    ? readJSON<unknown>(join(dir, 'report.json'))
    : undefined
  // The list of missed queries is large, and only the playground needs it
  const scores = { ...evaluation, misses: undefined }
  writeJSON(join(weightsDir, 'report.json'), {
    sha256,
    promoted: new Date().toISOString(),
    evaluation: scores,
    training,
  })
  const model = evaluation.bench[MODEL_METHOD]!
  writeJSON(qualityFile, Object.fromEntries(Object.entries(model).map(([set, score]) => [set, score.found])))
  console.log(`\nPromoted ${sha256.slice(0, 12)} to weights/`)
}

main()
