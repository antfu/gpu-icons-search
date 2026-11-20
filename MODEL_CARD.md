# Model card

## Status

No model is promoted. The best candidate of 2026-11-20 fails one rule of the ship gate, so the package has no weights in `weights/`. The candidate is not in the repository. The commands in the last section rebuild it in about 25 minutes on a 10-core CPU.

## Architecture

The model has 1,033,378 parameters. The two large tables use 6 bits per weight.

1. The CPU splits the query into words. It hashes each word to rows of a feature table with 16,384 rows. The features of a word are the whole word, the word with each single letter removed, every prefix of two or more letters, and every piece of three characters.
2. The vector of a word is the weighted mean of its rows, with 32 numbers per row. Each row has a learned weight.
3. A learned gate gives each word of the query a weight. The weighted mean of the word vectors goes through one residual layer with 64 hidden units.
4. The score of a target word is the cosine between that result and the row of the word in a second table. The second table has one row for each of the 15,265 words of the built-in word list.

The model runs once for the whole query, and once for each word of a longer query. The runtime is three WGSL compute passes with no dependencies: `encode`, `score` and `select`. The package has no CPU version of the model.

## Training

Training reads only stage 1 signals:

- Words that occur together in the tags of an icon. The tags come from 11 icon sets, listed in `THIRD_PARTY_NOTICES.md`.
- Alias names in Iconify icon sets.
- The 12 closest words by `bge-small-en-v1.5` embedding, with a weight of 0.25.

A single word points to itself and to its related words. A tag of several words points to the name words of its icons. During training, 25% of the single words are cut to a prefix and 25% get one or two typing mistakes. For such a word, the intended word gets half of the target weight. 15% of the examples get filler words such as "icon for" around them.

The run used PyTorch 2.15.0 on a CPU: 24,000 steps with a batch of 1,024, AdamW, and 6-bit rounding in the last 5,000 steps. It took 20 minutes.

## Scores

`evaluate` ran the candidate through the public API on WebGPU. The adapter was SwiftShader, a software GPU. The numbers are the share of queries with a correct icon on the first screen of 50 results.

| Method | alias | transfer | typing | overall |
| --- | ---: | ---: | ---: | ---: |
| `fzf` | 8.1% | 1.7% | 4.5% | 4.8% |
| `fzf+alias`, the current Icones search | 93.2% | 8.6% | 4.6% | 35.5% |
| `fzf+table`, tag counts with no model | 69.2% | 33.6% | 5.3% | 36.0% |
| `fzf+model`, this candidate | 61.4% | 31.5% | 23.0% | 38.7% |

The test sets have 2,643, 3,000 and 1,876 cases. Test set 4, the written queries, does not exist yet.

The ship gate gives this result:

| Rule | Result |
| --- | --- |
| Overall score above `fzf+alias` | Pass |
| `alias` not below `fzf+alias` | Fail: 61.4% against 93.2% |
| `transfer` not below `fzf+alias` | Pass |
| `typing` not below `fzf+alias` | Pass |
| WebGPU output matches the PyTorch fixtures | Pass: the largest difference is 0.00005 |
| Compressed size within 1 MB | Pass: 841,021 bytes with Brotli |

The `alias` test set comes from the manual alias list of Icones. The `fzf+alias` baseline uses that same list, so it holds the answers of this test set.

Other measurements of the candidate:

- For 600 common words, the intended word is in the top 8 predictions for 89% of the prefixes and for 34% of the misspellings.
- A warm `predict` call takes 0.97 ms on the software GPU. No measurement on a real GPU exists.
- The weight file is 960,672 bytes. The word list takes about 100,000 of them.

`training/data/sweep.json` compares three model sizes at 8,000 steps. The scores rise with the size up to the 1 MB limit.

## Decision point

The plan asks for a discussion if the model does not clearly win against `fzf+table` on test sets 2 to 4. The model wins on `typing`. It does not win on `transfer`, and it scores lower on `alias`. The tags and the teacher model limit both methods: the training targets themselves reach about 69% on `alias` and 36% on `transfer`.

Three options can raise the scores. Each one needs a decision of the maintainer:

- Run the Claude script for stage 2. It adds the synonyms and descriptions that the tags do not contain.
- Interleave the results of the predicted words. With one block per predicted word, the first word can fill the whole first screen. In an offline measurement on an earlier candidate, interleaving lifts `alias` from 64% to 70%.
- Return more predictions. With 12 words instead of 8, `alias` gains about 5 points and `transfer` gains about 4.

## Limits

- Descriptions work only for simple cases. For "icon for removing an item", the candidate predicts `remove` and `delete`, but also `drone` and `vacuum`.
- No test checks that opposites stay apart, for example `next` and `previous`.
- The model knows English only.
- A word outside the built-in word list is never predicted.
- The tests ran on a software GPU in Node and in headless Chromium. Nobody tested a real GPU, Safari or Firefox.

## Reproduce

```sh
pnpm install
pnpm data
pnpm -C training run train --steps 24000 --qat-steps 5000
pnpm evaluate
pnpm promote
```
