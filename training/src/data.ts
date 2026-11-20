import { join } from 'node:path'
import process from 'node:process'
import { iconifyVersion, loadIconSets, splitWords } from './iconify'
import { cacheDir, writeJSON, writeJSONL } from './paths'
import { fetchPinned } from './pinned'
import { tagLoaders } from './tags'

/** Words that at least this many icon sets use go into the built-in word list */
const MIN_SETS = 3

export interface WordStat {
  word: string
  /** Number of icon names that contain the word */
  icons: number
  /** Number of icon sets that contain the word */
  sets: number
}

function countWords(): WordStat[] {
  const stats = new Map<string, { icons: number, sets: Set<string> }>()
  for (const set of loadIconSets()) {
    for (const name of set.icons) {
      for (const word of new Set(splitWords(name))) {
        let stat = stats.get(word)
        if (!stat) {
          stats.set(word, stat = {
            icons: 0,
            sets: new Set(),
          })
        }
        stat.icons++
        stat.sets.add(set.prefix)
      }
    }
  }
  return [
    ...stats.entries(),
  ]
    .map(([word, stat]) => ({ word, icons: stat.icons, sets: stat.sets.size }))
    .sort((a, b) => b.icons - a.icons || a.word.localeCompare(b.word))
}

async function main(): Promise<void> {
  const words = countWords()
  const vocabulary = words.filter(stat => stat.sets >= MIN_SETS && /[a-z]/.test(stat.word) && stat.word.length > 1)
  writeJSON(join(cacheDir, 'words.json'), words)
  writeJSON(join(cacheDir, 'vocabulary.json'), vocabulary.map(stat => stat.word))
  console.log(`@iconify/json ${iconifyVersion()}: ${words.length} distinct words, ${vocabulary.length} in the word list`)

  const pinned = await fetchPinned()
  for (const [source, load] of Object.entries(tagLoaders(pinned))) {
    const icons = (await load()).filter(icon => icon.tags.length > 0)
    writeJSONL(join(cacheDir, 'tags', `${source}.jsonl`), icons)
    console.log(`${source.padEnd(17)} ${String(icons.length).padStart(6)} icons with tags, ${icons.reduce((sum, icon) => sum + icon.tags.length, 0)} tags`)
  }
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
