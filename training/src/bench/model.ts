import { join } from 'node:path'
import process from 'node:process'
import { cacheDir, writeJSON } from '../paths'
import { loadModel, referencePredictor } from '../reference'
import { baselineMethods, predictorMethod, printReport, runBenchmark } from './run'

/**
 * Score a candidate with the reference implementation, next to the baselines.
 * Usage: tsx src/bench/model.ts [candidate folder] [limit,limit,...]
 */
function main(): void {
  const dir = process.argv[2] ?? join(cacheDir, 'candidate')
  const limits = (process.argv[3] ?? '8').split(',').map(Number)
  const model = loadModel(dir)
  const methods = [
    ...baselineMethods().filter(method => method.name !== 'fzf'),
    ...limits.map(limit => predictorMethod(`fzf+model top${limit}`, referencePredictor(model, { limit }))),
  ]
  const report = runBenchmark(methods)
  printReport(report)
  writeJSON(join(dir, 'bench.json'), report)
}

main()
