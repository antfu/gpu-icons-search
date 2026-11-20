import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

/** Root of the training workspace package */
const trainingRoot = fileURLToPath(new URL('..', import.meta.url))

/** Git-ignored folder for everything that scripts can regenerate */
export const cacheDir = join(trainingRoot, '.cache')

/** Committed data that scripts cannot regenerate */
export const dataDir = join(trainingRoot, 'data')

/** Install folder of a dependency of the training package */
export function packageDir(name: string): string {
  return join(trainingRoot, 'node_modules', name)
}

export function readJSON<T>(file: string): T {
  return JSON.parse(readFileSync(file, 'utf8')) as T
}

export function writeJSON(file: string, data: unknown): void {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, `${JSON.stringify(data)}\n`)
}

export function writeJSONL(file: string, rows: unknown[]): void {
  mkdirSync(dirname(file), { recursive: true })
  writeFileSync(file, rows.map(row => `${JSON.stringify(row)}\n`).join(''))
}
