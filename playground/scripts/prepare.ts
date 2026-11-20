import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

/**
 * Write the static data of the playground into `public/data`:
 * icon names per icon set, the model weights, the benchmark report and the written test queries.
 * The model comes from `weights/` if a model is promoted, or from the last candidate.
 */

const root = fileURLToPath(new URL('../..', import.meta.url))
const out = fileURLToPath(new URL('../public/data', import.meta.url))
const iconify = fileURLToPath(new URL('../node_modules/@iconify/json', import.meta.url))

/** The icon sets of the benchmark. Keep in sync with `training/src/bench/build.ts`. */
const PREFIXES = [
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

function readJSON<T>(file: string): T {
  return JSON.parse(readFileSync(file, 'utf8')) as T
}

function firstExisting(...files: string[]): string | undefined {
  return files.find(file => existsSync(file))
}

mkdirSync(join(out, 'icons'), { recursive: true })

const collections = readJSON<Record<string, { name: string }>>(join(iconify, 'collections.json'))
const sets = PREFIXES.map((prefix) => {
  const { icons } = readJSON<{ icons: Record<string, { hidden?: boolean }> }>(join(iconify, 'json', `${prefix}.json`))
  const names = Object.keys(icons).filter(name => !icons[name]!.hidden)
  writeFileSync(join(out, 'icons', `${prefix}.json`), JSON.stringify(names))
  return { prefix, name: collections[prefix]!.name, total: names.length }
})
writeFileSync(join(out, 'sets.json'), JSON.stringify(sets))

const candidate = join(root, 'training/.cache/candidate')
const copies: [target: string, ...sources: string[]][] = [
  [
    'model.bin',
    join(root, 'weights/model.bin'),
    join(candidate, 'model.bin'),
  ],
  [
    'aliases.json',
    join(root, 'weights/aliases.json'),
  ],
  [
    'evaluation.json',
    join(candidate, 'evaluation.json'),
  ],
  [
    'report.json',
    join(root, 'weights/report.json'),
  ],
  [
    'sweep.json',
    join(root, 'training/data/sweep.json'),
  ],
]
for (const [target, ...sources] of copies) {
  const source = firstExisting(...sources)
  if (source)
    copyFileSync(source, join(out, target))
  console.log(`${target.padEnd(16)} ${source
    ? `from ${source.replace(root, '')}`
    : 'missing'}`)
}

const written = join(root, 'training/data/bench/written.jsonl')
const cases = existsSync(written)
  ? readFileSync(written, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line) as unknown)
  : []
writeFileSync(join(out, 'written.json'), JSON.stringify(cases))
console.log(`${sets.length} icon sets, ${cases.length} written test queries`)
