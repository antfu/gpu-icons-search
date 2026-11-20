import { join } from 'node:path'
import process from 'node:process'
import { cacheDir, readJSON, writeJSON } from '../paths'
import { TEACHER_MODELS, teacherPredictor } from '../teacher'
import { loadCases, predictorMethod, printReport, runBenchmark } from './run'

/** Compare the teacher models as direct predictors, to select one */
async function main(): Promise<void> {
  const vocabulary = readJSON<string[]>(join(cacheDir, 'vocabulary.json'))
  const cases = loadCases()
  const queries = cases.map(test => test.query)
  const methods = []
  for (const model of TEACHER_MODELS)
    methods.push(predictorMethod(`fzf+teacher ${model}`, await teacherPredictor(model, vocabulary, queries)))
  const report = runBenchmark(methods, cases)
  printReport(report)
  writeJSON(join(cacheDir, 'bench', 'teachers.json'), report)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
