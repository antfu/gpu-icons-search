# gpu-icons-search

[![npm version][npm-version-src]][npm-version-href]
[![npm downloads][npm-downloads-src]][npm-downloads-href]
[![bundle][bundle-src]][bundle-href]
[![JSDocs][jsdocs-src]][jsdocs-href]
[![License][license-src]][license-href]

A small model that runs on WebGPU and predicts related icon words for a search query. For example, `hom` gives `home` and `house`. It handles English keywords, short descriptions, half-typed words and typos.

The model only predicts words. Your own text matcher, for example fzf, still finds the icons. [Icones](https://github.com/antfu-collective/icones) is the first intended user: the predictions replace its manual alias list.

> [!WARNING]
> This project is experimental. No model passes the ship gate yet, so the package has no published weights. See [MODEL_CARD.md](./MODEL_CARD.md) for the current scores.

## Usage

```ts
import { predict } from 'gpu-icons-search'

const predictions = await predict('hom')
// [
//   { word: 'home', score: 0.43, from: 'hom', kind: 'completion' },
//   { word: 'house', score: 0.42, from: 'hom', kind: 'related' },
// ]
```

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

[PLAN.md](./PLAN.md) records the design decisions. [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) lists the training data and its licenses.

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
