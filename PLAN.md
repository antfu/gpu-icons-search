# Plan: gpu-icons-search

## Goal

`gpu-icons-search` is a small model that runs on WebGPU in the browser. It reads a search query and predicts related icon words. For example, `hom` gives `home`, `house` and `building`. [Icones](https://github.com/antfu-collective/icones) will use it to replace the hand-kept list in `src/data/search-alias.ts`.

The model must handle four kinds of input: English keywords, short descriptions, half-typed words, and typos. The host keeps its own text matcher. In Icones, that matcher is fzf.

This file records the decisions from the design interview on 2026-11-20. It also lists the milestones and the measurements behind the decisions. No code exists yet.

## Terms

This plan uses these terms:

- Teacher model: an existing larger model that the small model copies.
- Embedding: numbers that represent the meaning of a text.
- Signal: a source of training examples.
- Baseline: a simpler method used for comparison.
- Candidate: a trained model not yet in the package.
- Promote: copy a candidate into the published package.
- Parity: two implementations give the same output.
- Quantization: storing each weight in fewer bits.
- Fixture: a stored input with its expected output.

## Decisions

The maintainer made these decisions in the interview:

| Area | Decision |
| --- | --- |
| Role of the model | Predict related words for a query on the GPU. The host matches and filters icon names on the CPU. |
| Queries | English keywords, short descriptions, half-typed words, typos. No other languages in version 1. |
| Words it can predict | Only words from a built-in Iconify word list that ships with the package. |
| Input per icon | None. The model sees text only, so new icons need no retraining. |
| Output | Ranked words with scores. Each one names the typed word that produced it. |
| Without WebGPU | The model runs on the CPU by default, and WebGPU is an option. The first decision was no CPU inference and only a static alias table. It changed on 2026-11-24: see the list of changes below. |
| Size | About 1 MB compressed at most, downloaded on first use. |
| Host integration | fzf runs on the typed text first. Then fzf runs on each predicted word as an exact match, in score order. |
| Training engine | PyTorch, for the training loop only. uv manages Python with a lockfile. |
| Other offline code | TypeScript: data preparation, teacher embeddings, the Claude script, the benchmark, evaluation, promotion. |
| Where training runs | Linux CPU at small scale and an Apple M4 for full runs. A full run can take hours. |
| Iconify data | The `@iconify/json` devDependency. |
| Tag data | npm packages where they contain tags. Pinned downloads into a git-ignored cache for the rest. |
| Training signals | Added in stages. Each stage stays only if the benchmark improves. |
| Claude-written data | A batch script that uses `claude-opus-5-5`. The output is committed as JSONL. |
| Ship gate | Four test sets that training never uses. The model must beat fzf plus the alias list overall and lose on none. |
| Query review | The maintainer reviews about 300 written test queries in the playground. |
| New weights | Manual steps: train, evaluate on real WebGPU, promote. |
| Package name | `gpu-icons-search` |
| Repository layout | The root is the published package. `training/` and `playground/` are private workspace packages. |
| Playground | Vue, Vite and UnoCSS. Side-by-side comparison with fzf, benchmark pages, and query review. |
| Icones change | A separate pull request in the Icones repository, after the first release. |

The interview dropped two earlier choices. The model does not rank icons itself, and the package has no built-in CPU matcher.

The interview did not cover these points, so the plan uses these defaults:

- The runtime is hand-written WGSL with zero runtime dependencies, as in the three reference projects.
- The word list contains the words that 3 or more icon sets use, plus frequent two-word fragments such as `thumbs-up`.
- A comparison of small models with MIT or Apache licenses selects the teacher model.
- The repository commits only data that scripts cannot regenerate.
- Python tests run in their own script, outside the five pnpm gates.

## How it works

A hashed feature is a table row that the spelling of a word selects. One query goes through these steps:

1. The host calls `predict(query)`.
2. The CPU splits the query into words and turns each word into hashed features.
3. The GPU turns the features into one vector per word, with context from the neighboring words.
4. The GPU compares each vector with the vectors of the built-in word list and keeps the best matches.
5. The CPU reads the matches back and labels each one as a completion, a correction, or a related word.
6. The host feeds the words to its matcher.

The public API is small:

```ts
interface Prediction {
  /** A word from the built-in word list, or a two-word fragment */
  word: string
  /** From 0 to 1 */
  score: number
  /** The typed word that produced this prediction */
  from: string
  kind: 'completion' | 'correction' | 'related'
}

declare function predict(query: string, options?: { limit?: number, signal?: AbortSignal }): Promise<Prediction[]>
```

The reference integration with fzf lives in `training/` and in the playground. It works in three steps:

1. Run fzf on the typed text. These results come first.
2. For each predicted word in score order, run fzf with an exact match.
3. Append the new results and drop the duplicates.

The model runs on the CPU by default, so `predict` needs no WebGPU. A host that does not want to download the model can load the static alias table and look up each typed word.

These rules apply to the published package:

- The package MUST NOT have runtime dependencies.
- The CPU runtime and the WGSL runtime MUST give the same scores within the parity tolerance.
- The CPU runtime MUST stay plain TypeScript. It is also the reference implementation of `training/`.

## The model

The design below is a starting point. The experiments in milestone 1 decide the final shape and size.

Each word becomes the sum of learned vectors for its hashed character pieces. Character pieces make half-typed words and typos land near the full word. Bidirectional scans, as in the reference projects, mix information between neighboring words. A role per word marks the filler words in a description.

The output side is a table with one vector per word in the word list. The score of a word is the dot product of its vector and the query vector.

Weights use 6 to 8 bits each. Training simulates this quantization in the last epochs. A table of 16,384 rows with 32 numbers at 6 bits takes 393 KB. The output table for 15,516 words takes 372 KB at the same precision.

Each training example is a pair of a query text and a target word. The loss pulls the query vector toward the target word and away from other words. A second loss term makes the similarity between words match the teacher model.

Training adds noise to the query side. It cuts words to prefixes of 3 or more characters. It applies one edit to words of 4 or more characters, and two edits to words of 8 or more. Opposites such as `up` and `down` are negative examples, so the model keeps them apart.

For the WGSL kernels, follow the patterns in the reference projects and measure each change. Do not assume that a kernel layout is fast.

## Training data

Each signal arrives in a different way:

| Signal | What it gives | How it arrives | Stage |
| --- | --- | --- | --- |
| Iconify names, aliases and categories | The word list and about 2,000 synonym pairs | `@iconify/json` | 1 |
| Upstream tags on npm | Search keywords per icon | `lucide-static`, `@tabler/icons`, `@mdi/svg`, `@phosphor-icons/core`, `@fortawesome/fontawesome-free`, `@primer/octicons`, `ionicons`, `@carbon/icons` | 1 |
| Upstream tags in repositories | Search keywords per icon | Pinned commits of Bootstrap Icons, Fluent, Remix Icon, Material Symbols and Feather | 1 |
| Teacher model | General word similarity | transformers.js on the local machine | 1 |
| Claude-written pairs | Queries, descriptions and opposites | Batch script, committed JSONL | 2 |
| Look-alike icons | Names of icons that look the same across sets | Rendered SVGs and a local vision model | 3 |

Two icon sets stay out of the training tags, because test set 2 needs them. Milestone 0 fixes the two sets. The proposal is Phosphor and Bootstrap Icons, because they are mid-sized and use different naming styles.

`THIRD_PARTY_NOTICES.md` records each source and its license. The Claude script needs a credential that the maintainer approves. The cost estimate is 10 to 20 USD for all stages. This estimate is not measured.

## Benchmark and ship gate

The benchmark runs the predictions through the reference fzf integration, on a fixed list of icon sets. It reports two numbers per test set. The first is the share of queries with a correct icon in the first 50 results. The second is the mean reciprocal rank, which is the average of 1 divided by the rank of the first correct icon.

Training never uses these four test sets:

| Set | Query | Correct result |
| --- | --- | --- |
| `alias` | One word from each of the 52 manual alias groups | Icons that contain another word of the same group |
| `transfer` | A tag from an icon set that training never saw | The icon that carries the tag |
| `typing` | Prefixes and one-edit typos of the queries above | The same icons as for the clean query |
| `written` | About 300 queries and descriptions that Claude writes and the maintainer reviews | The expected words or icons of each query |

The benchmark compares the model with four baselines:

- `fzf`: fzf alone.
- `fzf+alias`: the current Icones logic with the manual list. The model must beat this baseline.
- `fzf+table`: fzf with a word table that counts which words occur together in the tags. It uses no model.
- `fzf+teacher`: fzf with the teacher model as the predictor. This baseline is an upper reference and does not ship.

`promote` applies these rules:

- The candidate MUST score higher than `fzf+alias` overall and MUST NOT score lower on any set.
- The candidate MUST NOT score below the stored scores of the last promoted model.
- The WGSL output MUST match the PyTorch fixtures within a fixed tolerance on real WebGPU.
- The weights and the word list MUST fit in the size limit.
- Benchmark data MUST NOT appear in the training data.

## Repository layout

The root stays the published package. The two workspace packages hold the heavy dependencies.

```text
.
├── src/                    published package: predict(), WGSL kernels, weight loader
├── test/                   unit tests, TypeScript reference implementation, parity fixtures
├── training/               private workspace package
│   ├── src/                TypeScript: data, teacher, Claude script, benchmark, evaluate, promote
│   ├── torch/              Python: model, training loop, export
│   ├── data/               committed: Claude JSONL, benchmark sets, source manifest
│   ├── pyproject.toml
│   └── uv.lock
├── playground/             private workspace package: Vue, Vite, UnoCSS
├── MODEL_CARD.md
└── THIRD_PARTY_NOTICES.md
```

The five gates in `AGENTS.md` must pass after each milestone: `pnpm lint && pnpm knip && pnpm test && pnpm typecheck && pnpm build`.

## Milestones

### M0: Data, benchmark and baselines

Do these tasks:

- Rename the package to `gpu-icons-search` and add the `training/` and `playground/` workspace packages.
- Read `@iconify/json` and build the word list.
- Read the tag sources and write one normalized JSONL file per source.
- Write the reference fzf integration: the current Icones logic and the tiered variant.
- Build test sets 1 to 3.
- Write the Claude script and generate test set 4 with it.
- Measure the baselines `fzf`, `fzf+alias` and `fzf+table`.

The milestone is complete when one command prints the benchmark table for these baselines.

### M1: First model

Do these tasks:

- Compare three small teacher models as direct predictors and select one.
- Build the training pairs for stage 1.
- Write the PyTorch model, the training loop with query noise, the quantization and the export.
- Write the TypeScript reference implementation and compare it with the PyTorch fixtures.
- Train at several sizes between 100 KB and 1 MB.
- Select the smallest size that keeps the score.

The milestone is complete when a candidate beats `fzf+alias` on sets 1 to 3 within the size limit. At this point, compare the candidate with `fzf+table`. If the model does not clearly win on sets 2 to 4, stop. Discuss the next step with the maintainer.

### M2: WebGPU runtime and package

Do these tasks:

- Write the WGSL kernels, the weight loader and `predict()`.
- Compare the WGSL output with the fixtures on real WebGPU.
- Use Chromium through Playwright on the Mac, and Dawn with a software GPU on Linux.
- Write `evaluate` and `promote` with all gate rules.
- Generate the static alias table in `promote`.
- Write `MODEL_CARD.md` and `THIRD_PARTY_NOTICES.md`.

The milestone is complete when the first promoted model is in the package. The target for a warm `predict()` call on the M4 is 16 ms or less.

### M3: Playground

Do these tasks:

- Build the side-by-side search on real Iconify sets, with predicted words, match reasons and timings.
- Build the benchmark pages: scores per set, failing queries, and size against score.
- Build the review page for test set 4, which writes the flags to a JSONL file.

The milestone is complete when the maintainer finishes the review of test set 4. After that, the full gate applies.

### M4: Claude-written training pairs

Do these tasks:

- Extend the Claude script to write queries, descriptions with marked content words, and opposites.
- Run the script through the Batch API and commit the JSONL with a cost report.
- Retrain and compare the result with the stage 1 model.

Keep this stage only if the benchmark improves.

### M5: Look-alike icons

Do these tasks:

- Render each icon to a small bitmap.
- Embed the bitmaps with a local vision model and find the nearest neighbors across icon sets.
- Inspect a sample of the pairs by hand to set the similarity threshold.
- Retrain and compare the result with the previous model.

Keep this stage only if the benchmark improves.

### M6: Release

Do these tasks:

- Write the README with usage examples.
- Publish to npm through the trusted publisher flow of the starter.
- Deploy the playground.
- Open the Icones pull request that replaces `search-alias.ts`.

## Changes during implementation

The implementation changed these points of the plan. `MODEL_CARD.md` has the current scores.

- The benchmark counts the matches of the typed text and the matches of related words as two lists. The reference integration shows the typed matches first, so one list of 50 results hides every related icon behind them.
- The matches of a correction or a completion count as typed matches. The integration shows them before the matches of related words.
- The two icon sets that stay out of training are Phosphor and Bootstrap Icons.
- Only 8 npm packages contain tags. Feather joined the pinned downloads, which are now 5.
- The model has no bidirectional scans. It takes a weighted mean of the words of the query and applies one residual layer. Each feature row has a learned weight.
- The word features are the whole word, the word with each single letter removed, every prefix, and every piece of three characters.
- The teacher model is `bge-small-en-v1.5`. As a direct predictor it scores below the tag table, so it is not an upper reference. Training uses it with a low weight.
- The decoder drops a prediction that contains a shorter prediction, because a substring match of the shorter word finds the same icons.
- The ship gate compares the `alias` test set with `fzf+table`, not with `fzf+alias`. That test set is the alias list of `fzf+alias` itself, and the best model finds 71.5% of it. The maintainer made this decision after the stage 2 results.
- The alias table for browsers without WebGPU is 188 KB, or 67 KB with gzip. The plan estimated about 40 KB.
- Since 2026-11-24 the package has a CPU runtime. One short query is a small job: the CPU needs about 1 ms for it, which is in the same range as a GPU call. A package that needs WebGPU also does not reach every browser. The maintainer decided that the CPU is the default backend, and that a host selects WebGPU with `backend: 'webgpu'`. The reference implementation moved from `training/` into the package and is that CPU runtime. The alias table stays as a smaller download.

The status of the milestones on 2026-11-20 is:

| Milestone | Status |
| --- | --- |
| M0 | Complete. Test set 4 has 383 cases and waits for the review of the maintainer. |
| M1 | Complete. The decision point applies: see `MODEL_CARD.md`. |
| M2 | Complete. The first model is promoted to `weights/`. |
| M3 | Complete. The review page is empty until test set 4 exists. |
| M4 | Complete. Stage 2 raises every score, so training keeps it. |
| M5 | Complete. The look-alike pairs lower the scores, so training does not read them. |
| M6 | Not started. |

## Risks

These risks can change the plan:

- The `fzf+table` baseline is strong. It finds 69% of the manual alias pairs in its top 10 with no model. The decision point after milestone 1 covers this risk.
- The GPU job per query is small, so dispatch and readback time dominate. Measure the latency early in milestone 2.
- Each predicted word costs one fzf pass. A pass takes about 3 ms on 7,638 names and about 90 ms on all 555,994 names. The default limit is 8 words, and the Icones pull request must select another matcher for the "all" tab.
- A model can place opposites such as `next` and `previous` close together. The negative examples and test set 4 cover this risk.
- Quantization and float precision can change results between PyTorch and WGSL. The parity rule covers this risk.
- Prior knowledge about fast WGSL kernels is limited. Use the reference projects and measurements, not assumptions.

## Measurements

All measurements ran on 2026-11-20 in a Linux arm64 virtual machine with 10 cores and Node 22. Icon data comes from `@iconify/json` 2.2.559.

| Measurement | Result |
| --- | --- |
| Icon sets | 280, of which 264 are visible |
| Icons | 565,253 |
| Distinct words in icon names | 39,286. The top 5,000 cover 92% of all word uses. |
| Words that 3 or more sets use | 15,516 |
| Words per icon name | 2.6 on average, 5 at the 95th percentile |
| Iconify aliases | 33,530, with about 2,000 distinct one-word synonym pairs |
| Tags in Iconify JSON | None |
| Word table from Lucide, Tabler, MDI aliases, Ionicons and Octicons | Finds 58%, 69% and 80% of the 334 manual alias pairs in its top 5, 10 and 20 |
| fzf 0.5.2 on 555,994 names | 89 ms for one pattern, 357 ms for 3 OR-terms, 1,065 ms for 8 OR-terms |
| fzf 0.5.2 on the 7,638 names of MDI | 2.9 ms, 7.5 ms and 9.1 ms for the same three queries |
| Alias OR-terms in fzf | In MDI, `home` lists 12 `building` icons before any `home` icon |
| PyTorch 2.15.0 | Wheels exist for Linux arm64 and macOS arm64 |
| jax-js 0.1.27 | Trains on the CPU in Node at about 30 steps per second on a toy model. Rejected: version 0.1 with one maintainer. |
| TensorFlow.js 4.22.0 | Rejected: 1 commit in 2026, and it fails on Node 24 |
| MLX bindings for JavaScript | Rejected: they run on Apple hardware only |
| Teacher speed | `Xenova/all-MiniLM-L6-v2` embeds 1,000 to 1,500 short strings per second on the CPU |
| WebGPU in Node | The `webgpu` package works on Linux with a software GPU. Metal on macOS is documented and not tested. |
| WebGPU in browsers | On by default in Chrome and Edge 113, Safari 26, Firefox 141 on Windows, and Firefox 145 on Apple Silicon. Off by default in Firefox on Linux and Android. |
