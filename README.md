# gpu-icons-search

[![npm version][npm-version-src]][npm-version-href]
[![npm downloads][npm-downloads-src]][npm-downloads-href]
[![bundle][bundle-src]][bundle-href]
[![JSDocs][jsdocs-src]][jsdocs-href]
[![License][license-src]][license-href]

A tiny WebGPU model that predicts related icon words for a search query, for example `hom` → `home`, `house`, `building`. See [PLAN.md](./PLAN.md).

## Note for Developers

This starter recommands using [npm Trusted Publisher](https://github.com/e18e/ecosystem-issues/issues/201), where the release is done on CI to ensure the security of the packages.

To do so, you need to run `pnpm publish` manually for the very first time to create the package on npm, and then go to `https://www.npmjs.com/package/<your-package-name>/access` to set the connection to your GitHub repo.

Then for the future releases, you can run `pnpm run release` to do the release and the GitHub Actions will take care of the release process.

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
