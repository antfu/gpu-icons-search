# gpu-icons-search

[![npm version][npm-version-src]][npm-version-href]
[![npm downloads][npm-downloads-src]][npm-downloads-href]
[![bundle][bundle-src]][bundle-href]
[![JSDocs][jsdocs-src]][jsdocs-href]
[![License][license-src]][license-href]

A small model that runs on WebGPU in the browser. It reads an icon search query and predicts the words that icon sets use for the same thing. For example, `movie` gives `cinema`, `film` and `video`, and the misspelled `calandar` gives `calendar`.

![Four queries, the words that the model predicts for them, and the icons that a text matcher then finds](./.github/assets/demo.png)

The model handles English words, short phrases, half-typed words and typos. It only predicts words. Your own text matcher, for example fzf, still finds the icons. [Icones](https://github.com/antfu-collective/icones) is the first intended user: the predictions replace its manual alias list.

On the benchmark of this repository, the search finds a correct icon on the first screen for 53.6% of the queries. The current Icones search finds one for 27.5%.

> [!WARNING]
> This project is experimental and not published to npm yet. See [MODEL_CARD.md](./MODEL_CARD.md) for the scores and the limits of the current model.

## Usage

```ts
import { predict } from 'gpu-icons-search'

const predictions = await predict('hom')
// [
//   { word: 'house', score: 0.45, from: 'hom', kind: 'related' },
//   { word: 'property', score: 0.39, from: 'hom', kind: 'related' },
//   { word: 'roof', score: 0.39, from: 'hom', kind: 'related' },
//   ...
// ]
```

The result has up to 12 words, best first. A word that contains the typed text, such as `home` here, is left out, because a text matcher finds its icons already.

`predict` loads the model on the first call. It rejects if WebGPU is not available. Every predicted word comes from a built-in list of words that occur in Iconify icon names.

Create your own predictor to control the weight file, the GPU device and the lifetime:

```ts
import { createPredictor, isSupported } from 'gpu-icons-search'

if (isSupported()) {
  const predictor = await createPredictor({ weights: '/assets/model.bin' })
  const controller = new AbortController()
  console.log(await predictor.predict('shoping cart', { limit: 8, signal: controller.signal }))
  predictor.dispose()
}
```

Abort a call when the user presses the next key. The call then rejects and its result is dropped.

## Use the predictions with fzf

Run your matcher on the typed text first. Then run it on each predicted word as an exact match, in score order. This keeps the matches of the typed text on top.

```ts
import { extendedMatch, Fzf } from 'fzf'
import { predict } from 'gpu-icons-search'

const fzf = new Fzf(iconNames, { casing: 'case-insensitive', match: extendedMatch })

async function search(query: string): Promise<string[]> {
  const results = new Set(fzf.find(query).map(entry => entry.item))
  for (const prediction of await predict(query)) {
    // The quote makes fzf match the word exactly
    for (const entry of fzf.find(`'${prediction.word}`))
      results.add(entry.item)
  }
  return [...results]
}
```

`training/src/search.ts` holds the full reference integration, which also replaces a single word of a longer query.

## Without WebGPU

The package has no CPU version of the model. For browsers without WebGPU, it ships a static table with the predictions of the model for common words. The table knows whole words only: no typos, no half-typed words, no descriptions.

```ts
import { isSupported, predict } from 'gpu-icons-search'
import { loadAliasTable, predictFromTable } from 'gpu-icons-search/fallback'

async function predictAnywhere(query: string) {
  if (isSupported())
    return predict(query)
  return predictFromTable(await loadAliasTable(), query)
}
```

## How it works

1. The CPU splits the query into words and hashes each word to rows of a feature table.
2. A WGSL shader turns the rows into one vector per word and one for the whole query.
3. A second shader compares each vector with the vector of every word in the word list.
4. A third shader keeps the best words, and the CPU reads them back.

The weights use 6 bits each. The weight file with the word list is below 1 MB.

## Development

The repository has three parts:

- The root is the published package: the WebGPU runtime in `src/`.
- `training/` holds the data pipeline, the benchmark and the PyTorch training loop.
- `playground/` compares the model with the current Icones search.

Install Node.js 22 or later, pnpm, and [uv](https://docs.astral.sh/uv/) for the Python part. On Linux without a GPU, also install Playwright Chromium: the scripts use its software Vulkan driver for WebGPU.

```sh
pnpm install
pnpm data       # read Iconify and the tag sources, build the training targets
pnpm bench      # print the benchmark table of the baselines
pnpm train      # fit a candidate with PyTorch (uses the Apple GPU on a Mac)
pnpm evaluate   # run the candidate on real WebGPU and score it
pnpm promote    # copy the candidate into weights/ if it passes the ship gate
pnpm play       # start the playground
```

`pnpm -C training run demo` writes the page behind the image at the top of this file. The image is a screenshot of that page.

[PLAN.md](./PLAN.md) records the design decisions. [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) lists the training data and its licenses.

## Prior art

Three projects showed that a small trained model with a hand-written WebGPU runtime can replace hand-written rules. This project follows their approach:

- [gpu-lexer](https://github.com/vercel-labs/gpu-lexer) by Shu Ding: a syntax highlighter for every language.
- [gpu-time](https://github.com/arikchakma/gpu-time) by Arik Chakma: a parser for date and time expressions.
- [gpu-cron](https://github.com/manuschillerdev/gpu-cron) by Manu Schiller: a parser for recurring schedules.

These ideas come from them:

- A model of about 1 MB or less, with WGSL shaders written for that one model and no runtime dependencies.
- Hashed features of each word in place of a word list on the input side.
- Weights of 6 bits each, with training that rounds the weights in its last steps.
- A candidate that must match the training framework on real WebGPU and pass a benchmark before it replaces the shipped weights.

The task differs. Those models give each word of the input a label. This model finds the closest words in a word list.

The project also builds on:

- [Icones](https://github.com/antfu-collective/icones): its manual alias list is the problem that this model solves, and a test set of the benchmark.
- [Iconify](https://iconify.design/): the icon names and the word list come from its icon sets.
- [fzf for JavaScript](https://github.com/ajitid/fzf-for-js): the text matcher of Icones and of the reference integration.

The code, the training data pipeline and the weights of this project are original.

## Sponsors

<p align="center">
  <a href="https://cdn.jsdelivr.net/gh/antfu/static/sponsors.svg">
    <img src="https://cdn.jsdelivr.net/gh/antfu/static/sponsors.svg" alt="Sponsors"/>
  </a>
</p>

## License

[MIT](./LICENSE) License © [Anthony Fu](https://github.com/antfu)

<!-- Badges -->

[npm-version-src]: https://img.shields.io/npm/v/gpu-icons-search?style=flat&colorA=080f12&colorB=1fa669
[npm-version-href]: https://npmx.dev/package/gpu-icons-search
[npm-downloads-src]: https://img.shields.io/npm/dm/gpu-icons-search?style=flat&colorA=080f12&colorB=1fa669
[npm-downloads-href]: https://npmx.dev/package/gpu-icons-search
[bundle-src]: https://img.shields.io/bundlephobia/minzip/gpu-icons-search?style=flat&colorA=080f12&colorB=1fa669&label=minzip
[bundle-href]: https://bundlephobia.com/result?p=gpu-icons-search
[license-src]: https://img.shields.io/github/license/antfu/gpu-icons-search.svg?style=flat&colorA=080f12&colorB=1fa669
[license-href]: https://github.com/antfu/gpu-icons-search/blob/main/LICENSE
[jsdocs-src]: https://img.shields.io/badge/jsdocs-reference-080f12?style=flat&colorA=080f12&colorB=1fa669
[jsdocs-href]: https://www.jsdocs.io/package/gpu-icons-search
