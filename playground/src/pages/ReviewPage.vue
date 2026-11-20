<script setup lang="ts">
import type { BenchCase } from '../../../training/src/bench/cases'
import type { PredictSource } from '../data'
import { onMounted, ref, watch } from 'vue'
import { isCorrect } from '../../../training/src/bench/cases'
import { loadSearch, loadWrittenCases } from '../data'

type Verdict = 'keep' | 'wrong'

const props = defineProps<{ source?: PredictSource }>()
const emit = defineEmits<{ open: [target: { prefix: string, query: string }] }>()

const STORAGE_KEY = 'gpu-icons-search:review'

const cases = ref<BenchCase[]>([])
const verdicts = ref<Record<string, Verdict>>(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}') as Record<string, Verdict>)
/** Rank of the expected icon in the new search, per case. 0 means not found. */
const ranks = ref<Record<string, number>>({})
const saved = ref('')

function key(test: BenchCase): string {
  return `${test.prefix}:${test.icon}:${test.query}`
}

onMounted(async () => {
  cases.value = await loadWrittenCases()
})

watch([
  cases,
  () => props.source,
], async () => {
  const source = props.source
  if (!source)
    return
  for (const test of cases.value) {
    const search = await loadSearch(test.prefix)
    const results = search.tiered(test.query, await source.predict(test.query, 8))
    ranks.value[key(test)] = results.findIndex(name => isCorrect(name, test)) + 1
  }
})

/** Store the verdicts in the browser, and in the repository when the dev server runs */
async function setVerdict(test: BenchCase, verdict: Verdict): Promise<void> {
  verdicts.value[key(test)] = verdict
  localStorage.setItem(STORAGE_KEY, JSON.stringify(verdicts.value))
  const rows = cases.value.filter(item => verdicts.value[key(item)]).map(item => ({ ...item, verdict: verdicts.value[key(item)] }))
  try {
    const response = await fetch('/api/review', {
      method: 'POST',
      body: JSON.stringify(rows),
    })
    saved.value = response.ok
      ? `Saved ${rows.length} verdicts to training/data/bench/written-review.jsonl`
      : 'The dev server did not save the verdicts'
  }
  catch {
    saved.value = 'Verdicts are stored in this browser only'
  }
}
</script>

<template>
  <p v-if="cases.length === 0" class="panel p-4 text-sm">
    No written test queries yet. Run <code>pnpm -C training claude submit written</code>, then <code>collect written</code>, and restart the playground.
  </p>
  <section v-else class="panel p-4">
    <h2 class="mb-1 text-sm font-semibold">
      Written test queries
    </h2>
    <p class="mb-3 text-xs muted">
      Mark a query as wrong if a user would not type it for this icon. {{ Object.keys(verdicts).length }} of {{ cases.length }} reviewed. {{ saved }}
    </p>
    <div
      v-for="test in cases"
      :key="key(test)"
      class="flex items-center gap-3 border-t border-neutral-200 py-2 dark:border-neutral-800"
    >
      <iconify-icon :icon="`${test.prefix}:${test.icon}`" width="28" height="28" />
      <div class="min-w-0 flex-1">
        <button class="font-mono hover:underline" @click="emit('open', test)">
          {{ test.query }}
        </button>
        <div class="text-xs muted">
          {{ test.prefix }}:{{ test.icon }}
          <template v-if="ranks[key(test)] !== undefined">
            , {{ ranks[key(test)] ? `found at rank ${ranks[key(test)]}` : 'not found' }}
          </template>
        </div>
      </div>
      <button class="tab" :class="verdicts[key(test)] === 'keep' ? 'tab-active' : ''" @click="setVerdict(test, 'keep')">
        Keep
      </button>
      <button class="tab" :class="verdicts[key(test)] === 'wrong' ? 'tab-active' : ''" @click="setVerdict(test, 'wrong')">
        Wrong
      </button>
    </div>
  </section>
</template>
