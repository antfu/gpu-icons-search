import type { BenchCase } from './bench/cases'
import { existsSync } from 'node:fs'
import { join } from 'node:path'
import process from 'node:process'
import Anthropic from '@anthropic-ai/sdk'
import { zodOutputFormat } from '@anthropic-ai/sdk/helpers/zod'
import { z } from 'zod'
import { BENCH_PREFIXES, benchFile } from './bench/build'
import { createRandom, sample } from './bench/cases'
import { loadIconSets, splitWords } from './iconify'
import { cacheDir, dataDir, readJSON, writeJSON, writeJSONL } from './paths'

const MODEL = 'claude-opus-5-5'

/** Batch prices of the model in USD per million tokens: half of the standard prices */
const PRICE = { input: 2, output: 10 }

/** Icons per request. A larger group needs fewer requests, but a longer answer. */
const GROUP = 25

/** Icons that the written test set asks about */
const WRITTEN_ICONS = 150

/** Words that the stage 2 training pairs cover, most frequent first */
const PAIR_WORDS = 6000

const WrittenSchema = z.object({
  icons: z.array(z.object({
    id: z.string().describe('The icon id from the request, unchanged'),
    keywords: z.array(z.string()).describe('Two search queries of one or two words'),
    description: z.string().describe('One short description of what the user wants'),
  })),
})

const PairsSchema = z.object({
  words: z.array(z.object({
    word: z.string().describe('The word from the request, unchanged'),
    queries: z.array(z.string()).describe('Up to eight search queries of one or two words that mean the same in an icon search'),
    descriptions: z.array(z.string()).describe('Up to three short descriptions that a user could type'),
    opposites: z.array(z.string()).describe('Up to three words that look related but must not match, for example opposites'),
  })),
})

const WRITTEN_PROMPT = `You help to test an icon search. For each icon below, write what a user would type to find it.

Rules:
- Do not use any word of the icon name, and no word that contains one of them.
- "keywords": two different queries of one or two words, as a user types them in a search box.
- "description": one short phrase of four to eight words that describes the need, for example "icon for removing an item".
- Use plain English. Think of what the icon shows and what apps use it for.
- Skip nothing. Return every icon id exactly as given.`

const PAIRS_PROMPT = `You help to train an icon search. Each word below occurs in the names of icons in many icon sets.
For each word, write what users type when they look for an icon that this word names.

Rules:
- "queries": synonyms and related search terms of one or two words. Include what the icon is used for in apps, for example "notification" for "bell".
- "descriptions": short phrases of four to eight words, for example "icon for removing an item" for "trash".
- "opposites": words that look related but name a different icon, for example "down" for "up" or "unlock" for "lock".
- Do not repeat the word itself. Leave a list empty if the word is a name, a brand, or has no clear meaning.`

type Job = 'written' | 'pairs'

interface BatchRequest {
  custom_id: string
  params: Anthropic.Messages.MessageCreateParamsNonStreaming
}

interface JobState {
  batch: string
  /** For the written set: the icon set and name of each icon id */
  icons?: Record<string, { prefix: string, name: string }>
}

function chunk<T>(items: T[], size: number): T[][] {
  return Array.from({
    length: Math.ceil(items.length / size),
  }, (_, index) => items.slice(index * size, (index + 1) * size))
}

function request(id: string, system: string, content: string, format: ReturnType<typeof zodOutputFormat>): BatchRequest {
  return {
    custom_id: id,
    params: {
      model: MODEL,
      max_tokens: 16000,
      thinking: { type: 'adaptive' },
      output_config: { effort: 'medium', format },
      system,
      messages: [
        { role: 'user', content },
      ],
    },
  }
}

/** Pick icons with at least two words in the name, spread over the benchmark icon sets */
function pickWrittenIcons(): Record<string, { prefix: string, name: string }> {
  const random = createRandom(4)
  const picked: Record<string, { prefix: string, name: string }> = {}
  const sets = loadIconSets().filter(set => BENCH_PREFIXES.includes(set.prefix))
  for (const set of sets) {
    const names = set.icons.filter(name => /^[a-z]+(?:-[a-z]+)?$/.test(name))
    for (const name of sample(names, Math.ceil(WRITTEN_ICONS / sets.length), random))
      picked[`${set.prefix}:${name}`] = { prefix: set.prefix, name }
  }
  return picked
}

function buildRequests(job: Job): { requests: BatchRequest[], icons?: JobState['icons'] } {
  if (job === 'written') {
    const icons = pickWrittenIcons()
    const requests = chunk(Object.keys(icons), GROUP).map((ids, index) =>
      request(`written-${index}`, WRITTEN_PROMPT, ids.join('\n'), zodOutputFormat(WrittenSchema)))
    return { requests, icons }
  }
  const words = readJSON<string[]>(join(cacheDir, 'vocabulary.json')).slice(0, PAIR_WORDS)
  const requests = chunk(words, GROUP * 2).map((group, index) =>
    request(`pairs-${index}`, PAIRS_PROMPT, group.join('\n'), zodOutputFormat(PairsSchema)))
  return { requests }
}

function stateFile(job: Job): string {
  return join(cacheDir, 'claude', `${job}.json`)
}

async function submit(client: Anthropic, job: Job): Promise<void> {
  const { requests, icons } = buildRequests(job)
  const batch = await client.messages.batches.create({ requests })
  writeJSON(stateFile(job), { batch: batch.id, icons } satisfies JobState)
  console.log(`Submitted batch ${batch.id} with ${requests.length} requests. Run "collect ${job}" when it ends.`)
}

/** Turn the answers about icons into test cases. A case is dropped if it uses a word of the icon name. */
function writtenCases(answers: z.infer<typeof WrittenSchema>[], icons: NonNullable<JobState['icons']>): BenchCase[] {
  const cases: BenchCase[] = []
  for (const answer of answers.flatMap(result => result.icons)) {
    const icon = icons[answer.id]
    if (!icon)
      continue
    for (const query of [...answer.keywords, answer.description]) {
      if (splitWords(query).every(word => !icon.name.includes(word))) {
        cases.push({
          set: 'written',
          prefix: icon.prefix,
          query: query.toLowerCase(),
          icon: icon.name,
        })
      }
    }
  }
  return cases
}

async function collect(client: Anthropic, job: Job): Promise<void> {
  const state = readJSON<JobState>(stateFile(job))
  const batch = await client.messages.batches.retrieve(state.batch)
  if (batch.processing_status !== 'ended') {
    console.log(`Batch ${state.batch} is ${batch.processing_status}: ${batch.request_counts.processing} requests still run.`)
    return
  }
  const answers: unknown[] = []
  const usage = { input: 0, output: 0 }
  let failed = 0
  for await (const result of await client.messages.batches.results(state.batch)) {
    const message = result.result.type === 'succeeded'
      ? result.result.message
      : undefined
    const text = message?.content.find(block => block.type === 'text')?.text
    if (!message || message.stop_reason !== 'end_turn' || !text) {
      failed++
      continue
    }
    usage.input += message.usage.input_tokens
    usage.output += message.usage.output_tokens
    answers.push(JSON.parse(text))
  }
  const cost = (usage.input * PRICE.input + usage.output * PRICE.output) / 1e6
  if (job === 'written') {
    const cases = writtenCases(answers.map(answer => WrittenSchema.parse(answer)), state.icons ?? {})
    writeJSONL(benchFile('written'), cases)
    console.log(`Wrote ${cases.length} written test cases. Review them in the playground before they count.`)
  }
  else {
    const words = answers.flatMap(answer => PairsSchema.parse(answer).words)
    writeJSONL(join(dataDir, 'claude', 'pairs.jsonl'), words)
    console.log(`Wrote training pairs for ${words.length} words.`)
  }
  writeJSON(join(dataDir, 'claude', `${job}-report.json`), {
    model: MODEL,
    batch: state.batch,
    failed,
    usage,
    costUSD: Number(cost.toFixed(2)),
  })
  console.log(`${failed} requests failed. Tokens: ${usage.input} in, ${usage.output} out. Cost: ${cost.toFixed(2)} USD.`)
}

/**
 * Write test queries and training pairs with Claude, through the Message Batches API.
 *
 * Usage: tsx src/claude.ts <plan|submit|collect> <written|pairs>
 *
 * `plan` calls no API. `submit` and `collect` spend money on the account of the credential in the environment.
 */
async function main(): Promise<void> {
  const [action, job] = process.argv.slice(2)
  if (!['plan', 'submit', 'collect'].includes(action ?? '') || (job !== 'written' && job !== 'pairs'))
    throw new Error('Usage: tsx src/claude.ts <plan|submit|collect> <written|pairs>')
  if (action === 'plan') {
    const { requests } = buildRequests(job)
    const characters = requests.reduce((sum, item) => sum + JSON.stringify(item.params.messages).length + String(item.params.system).length, 0)
    console.log(`${requests.length} requests to ${MODEL}, about ${Math.round(characters / 4)} input tokens. No API call was made.`)
    return
  }
  if (action === 'collect' && !existsSync(stateFile(job)))
    throw new Error(`No submitted batch for "${job}". Run "submit ${job}" first.`)
  const client = new Anthropic()
  await (action === 'submit'
    ? submit(client, job)
    : collect(client, job))
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
