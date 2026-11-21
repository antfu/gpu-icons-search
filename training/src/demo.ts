import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { cacheDir, packageDir, readJSON, trainingRoot } from './paths'
import { loadModel, referencePredictor } from './reference'
import { IconSearch } from './search'

/** Icon set of the demo. Its icons are plain line drawings with short names. */
const PREFIX = 'lucide'

/** Queries of the demo, each with the kind of help that it shows */
const QUERIES: [query: string, label: string][] = [
  ['calandar', 'typo'],
  ['movie', 'other word'],
  ['dark mode', 'phrase'],
  ['profile', 'other word'],
]

const WORDS_SHOWN = 5
const ICONS_SHOWN = 7

interface RawIconSet {
  icons: Record<string, { body: string, hidden?: boolean }>
}

const STYLE = `
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px 32px; width: 1200px; background: #fff; color: #171717; font: 15px/1.4 ui-sans-serif, system-ui, sans-serif; }
  .row { display: grid; grid-template-columns: 230px 250px 1fr; gap: 28px; align-items: center; padding: 18px 0; border-top: 1px solid #e5e5e5; }
  .row:first-of-type { border-top: 0; }
  .head { color: #737373; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; padding: 0 0 8px; }
  .query { font: 600 20px ui-monospace, monospace; padding: 10px 14px; border: 1.5px solid #0d9488; border-radius: 10px; }
  .note { margin-top: 6px; font-size: 12px; color: #737373; }
  .words { display: flex; flex-wrap: wrap; gap: 6px; }
  .word { font: 13px ui-monospace, monospace; padding: 3px 9px; border-radius: 999px; background: #f0fdfa; border: 1px solid #99f6e4; color: #115e59; }
  .word b { font-weight: 400; color: #5eaaa1; margin-left: 4px; }
  .icons { display: grid; grid-template-columns: repeat(${ICONS_SHOWN}, 1fr); gap: 6px; }
  .icon { text-align: center; font-size: 10px; color: #737373; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .icon svg { display: block; width: 30px; height: 30px; margin: 0 auto 5px; color: #171717; }
`

function escape(text: string): string {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;')
}

/**
 * Write a static page that shows what the promoted model predicts for a few queries,
 * and the icons that the reference integration then finds.
 * A screenshot of this page is the demo image of the README.
 */
function main(): void {
  const set = readJSON<RawIconSet>(join(packageDir('@iconify/json'), 'json', `${PREFIX}.json`))
  const names = Object.keys(set.icons).filter(name => !set.icons[name]!.hidden)
  const search = new IconSearch(names)
  const predict = referencePredictor(loadModel(join(trainingRoot, '..', 'weights')))

  const rows = QUERIES.map(([query, label]) => {
    const predictions = predict(query)
    const results = search.tieredWithReasons(query, predictions).slice(0, ICONS_SHOWN)
    const words = predictions.slice(0, WORDS_SHOWN)
      .map(prediction => `<span class="word">${escape(prediction.word)}<b>${prediction.score.toFixed(2)}</b></span>`)
    const icons = results
      .map(result => `<div class="icon"><svg viewBox="0 0 24 24">${set.icons[result.icon]!.body}</svg>${escape(result.icon)}</div>`)
    return `<div class="row">
      <div><div class="query">${escape(query)}</div><div class="note">${label} · fzf alone finds ${search.icones(query).length} icons</div></div>
      <div class="words">${words.join('')}</div>
      <div class="icons">${icons.join('')}</div>
    </div>`
  })

  const page = `<!doctype html><meta charset="utf-8"><style>${STYLE}</style>
    <div class="row head"><div>You type</div><div>The model predicts</div><div>Your matcher finds, in Lucide</div></div>
    ${rows.join('\n')}`
  const dir = join(cacheDir, 'demo')
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'index.html'), page)
  console.log(`Wrote ${join(dir, 'index.html')}. Take a screenshot of it for .github/assets/demo.png.`)
}

main()
