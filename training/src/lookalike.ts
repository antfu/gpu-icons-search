import { join } from 'node:path'
import process from 'node:process'
import { Resvg } from '@resvg/resvg-js'
import { createRandom } from './bench/cases'
import { loadCollections, splitWords } from './iconify'
import { cacheDir, packageDir, readJSON, writeJSONL } from './paths'
import { STYLE_WORDS } from './targets'

/** Side of the rendered bitmap, and of the signature after the crop to the drawing */
const RENDER = 32
const SIDE = 16
const DIMENSIONS = SIDE * SIDE

/** Hash tables and bits per table of the search for similar signatures */
const TABLES = 6
const BITS = 14

/** Two icons look alike if the cosine of their signatures is at least this */
const MIN_SIMILARITY = 0.9

/** Icon set categories of general interface icons. Other sets hold logos, flags, emoji and maps. */
const UI_CATEGORIES = new Set([
  'UI 24px',
  'UI 16px / 32px',
  'UI Other / Mixed Grid',
  'Material',
])

/** Larger icon sets are skipped, because rendering takes about 5 ms per icon on a CPU */
const MAX_SET_ICONS = 8000

/** Buckets with more icons than this are skipped: they hold shapes such as plain circles */
const MAX_BUCKET = 400

/**
 * Stage 3 of the training signals: icons from different icon sets that look alike.
 *
 * The ablation of 2026-11-20 added the name words of each pair to the tag table.
 * The scores went down at every similarity threshold, so training does not read these pairs.
 * `MODEL_CARD.md` has the numbers.
 */
export interface LookalikePair {
  /** Two icon ids with their prefix, for example `mdi:dots-vertical` */
  a: string
  b: string
  similarity: number
}

interface RawIconSet {
  width?: number
  height?: number
  icons: Record<string, { body: string, width?: number, height?: number, hidden?: boolean }>
}

interface Bitmap {
  /** Side of the square bitmap in pixels */
  size: number
  /** Darkness of each pixel, from 0 for the background to 1 */
  ink: Float32Array
}

/** Draw an icon in black on white. Icons that fail to render or are not square give no bitmap. */
function renderIcon(body: string, width: number, height: number): Bitmap | undefined {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" color="#000">${body}</svg>`
  try {
    const image = new Resvg(svg, {
      fitTo: { mode: 'width', value: RENDER },
      background: '#fff',
    }).render()
    if (image.height !== image.width)
      return undefined
    const pixels = image.pixels
    const ink = Float32Array.from({ length: image.width ** 2 }, (_, index) => 1 - pixels[index * 4]! / 255)
    return { size: image.width, ink }
  }
  catch {
    return undefined
  }
}

/** The smallest square around the drawing, as its left and top edge and its side */
function drawingSquare({ size, ink }: Bitmap): { left: number, top: number, span: number } | undefined {
  let [left, top, right, bottom] = [size, size, -1, -1]
  for (let index = 0; index < ink.length; index++) {
    if (ink[index]! <= 0.2)
      continue
    const x = index % size
    const y = Math.floor(index / size)
    left = Math.min(left, x)
    right = Math.max(right, x)
    top = Math.min(top, y)
    bottom = Math.max(bottom, y)
  }
  if (right < left)
    return undefined
  const span = Math.max(right - left, bottom - top) + 1
  return { left: left + (right - left + 1 - span) / 2, top: top + (bottom - top + 1 - span) / 2, span }
}

/**
 * Render an icon and reduce it to a small signature.
 * The signature is the drawing cropped to its bounding box, scaled to 16 x 16, with zero mean and unit length.
 * The crop removes the padding, which differs between icon sets.
 */
function signature(body: string, width: number, height: number): Float32Array | undefined {
  const bitmap = renderIcon(body, width, height)
  const square = bitmap && drawingSquare(bitmap)
  if (!bitmap || !square)
    return undefined
  const { size, ink } = bitmap
  const result = Float32Array.from({ length: DIMENSIONS }, (_, index) => {
    // Sample the center of each cell of the square around the drawing
    const x = Math.floor(square.left + (index % SIDE + 0.5) * square.span / SIDE)
    const y = Math.floor(square.top + (Math.floor(index / SIDE) + 0.5) * square.span / SIDE)
    return x >= 0 && x < size && y >= 0 && y < size
      ? ink[y * size + x]!
      : 0
  })
  const mean = result.reduce((sum, value) => sum + value, 0) / DIMENSIONS
  const norm = Math.sqrt(result.reduce((sum, value) => sum + (value - mean) ** 2, 0))
  return norm < 1e-3
    ? undefined
    : result.map(value => (value - mean) / norm)
}

/** Words of an icon name that say what the icon shows */
function meaningWords(name: string): string[] {
  return splitWords(name).filter(word => /[a-z]/.test(word) && !STYLE_WORDS.has(word))
}

function cosine(a: Float32Array, b: Float32Array): number {
  let sum = 0
  for (let i = 0; i < DIMENSIONS; i++)
    sum += a[i]! * b[i]!
  return sum
}

/** Render every icon with a short name from the single-color interface icon sets of moderate size */
function renderIcons(): { ids: string[], signatures: Float32Array[] } {
  const root = packageDir('@iconify/json')
  // `--sets=mdi,lucide` limits the run to a few icon sets, for a quick test
  const only = process.argv.find(argument => argument.startsWith('--sets='))?.slice(7).split(',')
  const ids: string[] = []
  const signatures: Float32Array[] = []
  for (const [prefix, info] of Object.entries(loadCollections())) {
    if (info.hidden || info.palette || !UI_CATEGORIES.has(info.category ?? '') || (info.total ?? 0) > MAX_SET_ICONS)
      continue
    if (only && !only.includes(prefix))
      continue
    const set = readJSON<RawIconSet>(join(root, 'json', `${prefix}.json`))
    for (const [name, icon] of Object.entries(set.icons)) {
      const words = meaningWords(name)
      if (icon.hidden || words.length === 0 || words.length > 3)
        continue
      const result = signature(icon.body, icon.width ?? set.width ?? 16, icon.height ?? set.height ?? 16)
      if (result) {
        ids.push(`${prefix}:${name}`)
        signatures.push(result)
      }
    }
    console.log(`${prefix.padEnd(24)} ${ids.length} icons so far`)
  }
  return { ids, signatures }
}

/**
 * Find pairs of icons from different icon sets that look alike but have different names.
 * Random hyperplanes sort the signatures into buckets, and only icons of one bucket are compared.
 */
function findPairs(ids: string[], signatures: Float32Array[]): LookalikePair[] {
  const random = createRandom(11)
  const pairs = new Map<string, LookalikePair>()
  for (let table = 0; table < TABLES; table++) {
    const planes = Array.from({ length: BITS }, () => Float32Array.from({ length: DIMENSIONS }, () => random() - 0.5))
    const buckets = new Map<number, number[]>()
    signatures.forEach((vector, index) => {
      const code = planes.reduce((bits, plane, bit) => cosine(vector, plane) > 0
        ? bits | (1 << bit)
        : bits, 0)
      const bucket = buckets.get(code)
      if (bucket)
        bucket.push(index)
      else
        buckets.set(code, [index])
    })
    for (const bucket of buckets.values()) {
      if (bucket.length > MAX_BUCKET)
        continue
      for (let i = 0; i < bucket.length; i++) {
        for (let j = i + 1; j < bucket.length; j++)
          addPair(pairs, ids, signatures, bucket[i]!, bucket[j]!)
      }
    }
  }
  return [
    ...pairs.values(),
  ]
}

function addPair(pairs: Map<string, LookalikePair>, ids: string[], signatures: Float32Array[], i: number, j: number): void {
  const [a, b] = [ids[i]!, ids[j]!]
  const key = `${a} ${b}`
  if (pairs.has(key) || a.split(':')[0] === b.split(':')[0])
    return
  const similarity = cosine(signatures[i]!, signatures[j]!)
  if (similarity < MIN_SIMILARITY)
    return
  const wordsA = meaningWords(a.split(':')[1]!)
  const wordsB = new Set(meaningWords(b.split(':')[1]!))
  // The pair teaches something only if the names differ
  if (wordsA.length === wordsB.size && wordsA.every(word => wordsB.has(word)))
    return
  pairs.set(key, {
    a,
    b,
    similarity: Number(similarity.toFixed(3)),
  })
}

function lookalikeFile(): string {
  return join(cacheDir, 'lookalike', 'pairs.jsonl')
}

function main(): void {
  const started = performance.now()
  const { ids, signatures } = renderIcons()
  console.log(`rendered ${ids.length} icons in ${((performance.now() - started) / 1000).toFixed(0)} s`)
  const pairs = findPairs(ids, signatures).sort((x, y) => y.similarity - x.similarity)
  writeJSONL(lookalikeFile(), pairs)
  console.log(`${pairs.length} look-alike pairs with different names in ${((performance.now() - started) / 1000).toFixed(0)} s`)
  for (const pair of pairs.filter((_, index) => index % Math.ceil(pairs.length / 25) === 0))
    console.log(`${pair.similarity}  ${pair.a}  ${pair.b}`)
}

if (import.meta.main) {
  try {
    main()
  }
  catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}
