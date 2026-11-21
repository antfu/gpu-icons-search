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
    <div class="flex flex-wrap items-baseline gap-x-3 gap-y-1">
      <h2 class="section-title">
        Written test queries
      </h2>
      <span class="badge">{{ Object.keys(verdicts).length }} / {{ cases.length }} reviewed</span>
    </div>
    <div class="mt-2 h-1 overflow-hidden rounded-full bg-neutral-100 dark:bg-neutral-800">
      <div class="h-full bg-teal-500 transition-all" :style="{ width: `${Object.keys(verdicts).length / cases.length * 100}%` }" />
    </div>
    <p class="mb-2 mt-2 text-xs muted">
      Mark a query as wrong if a user would not type it for this icon. {{ saved }}
    </p>
    <div
      v-for="test in cases"
      :key="key(test)"
      class="flex items-center gap-3 border-t border-base py-2.5"
    >
      <div class="h-10 w-10 flex shrink-0 items-center justify-center rounded-lg bg-neutral-100 dark:bg-neutral-800">
        <iconify-icon :icon="`${test.prefix}:${test.icon}`" width="24" height="24" />
      </div>
      <div class="min-w-0 flex-1">
        <button class="text-left font-mono hover:underline" @click="emit('open', test)">
          {{ test.query }}
        </button>
        <div class="truncate text-xs muted">
          {{ test.prefix }}:{{ test.icon }}
          <template v-if="ranks[key(test)] !== undefined">
            · <span :class="ranks[key(test)] ? 'accent' : ''">{{ ranks[key(test)] ? `rank ${ranks[key(test)]}` : 'not found' }}</span>
          </template>
        </div>
      </div>
      <div class="segmented shrink-0">
        <button
          class="segment px-2.5"
          :class="verdicts[key(test)] === 'keep' ? 'segment-active !text-teal-700 dark:!text-teal-400' : ''"
          title="Keep"
          @click="setVerdict(test, 'keep')"
        >
          <iconify-icon icon="ph:check-bold" class="sm:hidden" width="16" height="16" />
          <span class="hidden sm:inline">Keep</span>
        </button>
        <button
          class="segment px-2.5"
          :class="verdicts[key(test)] === 'wrong' ? 'segment-active !text-red-600 dark:!text-red-400' : ''"
          title="Wrong"
          @click="setVerdict(test, 'wrong')"
        >
          <iconify-icon icon="ph:x-bold" class="sm:hidden" width="16" height="16" />
          <span class="hidden sm:inline">Wrong</span>
        </button>
      </div>
    </div>
  </section>
</template>
