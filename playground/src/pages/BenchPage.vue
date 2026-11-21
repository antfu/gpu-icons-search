<script setup lang="ts">
import type { BenchSet } from '../../../training/src/bench/cases'
import type { Evaluation } from '../../../training/src/evaluate'
import type { SweepEntry } from '../../../training/src/sweep'
import { computed, onMounted, ref } from 'vue'
import { loadEvaluation, loadSweep } from '../data'

const emit = defineEmits<{ open: [target: { prefix: string, query: string }] }>()

const SETS: (BenchSet | 'overall')[] = [
  'alias',
  'transfer',
  'typing',
  'written',
  'overall',
]
const SHOWN_MISSES = 200

const evaluation = ref<Evaluation>()
const sweep = ref<SweepEntry[]>([])
const loaded = ref(false)
const missSet = ref<BenchSet>('alias')
const onlyRegressions = ref(false)

onMounted(async () => {
  evaluation.value = await loadEvaluation()
  sweep.value = await loadSweep()
  loaded.value = true
})

const sets = computed(() => SETS.filter(set => Object.values(evaluation.value?.bench ?? {}).some(scores => scores[set])))

const misses = computed(() => (evaluation.value?.misses ?? [])
  .filter(miss => miss.set === missSet.value && (!onlyRegressions.value || miss.baselineFound)))

function percent(value: number | undefined): string {
  return value === undefined
    ? ''
    : `${(value * 100).toFixed(1)}%`
}

/** Best score of a column, to mark the leading method */
function best(set: BenchSet | 'overall'): number {
  return Math.max(...Object.values(evaluation.value?.bench ?? {}).map(scores => scores[set]?.found ?? 0))
}

function expected(miss: Evaluation['misses'][number]): string {
  return miss.icon ?? [miss.word].flat().join(', ')
}
</script>

<template>
  <p v-if="loaded && !evaluation" class="panel p-4 text-sm">
    No evaluation found. Run <code>pnpm -C training evaluate</code>, then restart the playground.
  </p>
  <template v-if="evaluation">
    <section class="panel p-4">
      <h2 class="mb-1 section-title">
        Scores per test set
      </h2>
      <p class="mb-3 text-xs muted">
        Share of queries with a correct icon on the first screen. Matches of the typed text and other matches count as two lists.
      </p>
      <div class="scroll-x -mx-4 px-4">
        <table class="w-full min-w-md whitespace-nowrap text-sm tabular-nums">
          <thead>
            <tr class="text-left text-xs muted">
              <th class="sticky left-0 bg-white py-1.5 pr-3 font-normal dark:bg-neutral-900">
                Method
              </th>
              <th v-for="set in sets" :key="set" class="py-1.5 pl-3 text-right font-normal">
                {{ set }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(scores, method) in evaluation.bench" :key="method" class="border-t border-neutral-200 dark:border-neutral-800">
              <td class="sticky left-0 bg-white py-1.5 pr-3 font-mono dark:bg-neutral-900">
                {{ method }}
              </td>
              <td
                v-for="set in sets"
                :key="set"
                class="py-1.5 pl-3 text-right font-mono"
                :class="scores[set]?.found === best(set) ? 'font-bold accent' : ''"
              >
                {{ percent(scores[set]?.found) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <dl class="grid mt-4 gap-x-4 gap-y-0.5 border-t border-base pt-3 text-xs sm:grid-cols-[max-content_1fr] sm:gap-y-1.5">
        <dt class="muted">
          GPU adapter
        </dt>
        <dd class="mb-2 sm:mb-0">
          {{ evaluation.adapter }}
        </dd>
        <dt class="muted">
          Parity with PyTorch
        </dt>
        <dd class="mb-2 sm:mb-0">
          WebGPU {{ evaluation.parity.webgpu.toExponential(1) }}, reference {{ evaluation.parity.reference.toExponential(1) }}, tolerance {{ evaluation.parity.tolerance }}
        </dd>
        <dt class="muted">
          Weight file
        </dt>
        <dd class="mb-2 sm:mb-0">
          {{ evaluation.bytes.raw }} bytes, {{ evaluation.bytes.brotli }} with Brotli
        </dd>
        <dt class="muted">
          Warm call
        </dt>
        <dd class="mb-2 sm:mb-0">
          {{ evaluation.latencyMs }} ms median on that adapter
        </dd>
      </dl>
    </section>

    <section v-if="sweep.length > 0" class="panel mt-4 p-4">
      <h2 class="mb-1 section-title">
        Size against score
      </h2>
      <p class="mb-3 text-xs muted">
        The same training at {{ sweep[0]!.steps }} steps for each size, scored with the reference implementation.
      </p>
      <div class="scroll-x -mx-4 px-4">
        <table class="w-full min-w-md whitespace-nowrap text-sm tabular-nums">
          <thead>
            <tr class="text-left text-xs muted">
              <th class="py-1.5 pr-3 font-normal">
                Feature rows
              </th>
              <th class="py-1.5 pr-3 font-normal">
                Vector length
              </th>
              <th class="py-1.5 pl-3 text-right font-normal">
                Weight file
              </th>
              <th v-for="set in Object.keys(sweep[0]!.found)" :key="set" class="py-1.5 pl-3 text-right font-normal">
                {{ set }}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="entry in sweep" :key="entry.bytes" class="border-t border-neutral-200 font-mono dark:border-neutral-800">
              <td class="py-1.5 pr-3">
                {{ entry.rows }}
              </td>
              <td class="py-1.5 pr-3">
                {{ entry.dim }}
              </td>
              <td class="py-1.5 pl-3 text-right">
                {{ Math.round(entry.bytes / 1024) }} KB
              </td>
              <td v-for="(value, set) in entry.found" :key="set" class="py-1.5 pl-3 text-right">
                {{ percent(value) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel mt-4 p-4">
      <div class="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
        <h2 class="section-title">
          Queries that the model misses
        </h2>
        <label class="flex items-center gap-1.5 text-xs muted sm:order-last sm:ml-auto">
          <input v-model="onlyRegressions" type="checkbox" class="accent-teal-600">
          Only where fzf+alias finds it
        </label>
        <div class="scroll-x -mx-4 w-[calc(100%+2rem)] flex gap-1.5 px-4 sm:mx-0 sm:w-auto sm:px-0">
          <button
            v-for="set in sets.filter(set => set !== 'overall')"
            :key="set"
            class="tab"
            :class="missSet === set ? 'tab-active' : ''"
            @click="missSet = set as BenchSet"
          >
            {{ set }}
          </button>
        </div>
      </div>
      <p class="mb-2 text-xs muted">
        {{ misses.length }} misses. The first {{ Math.min(misses.length, SHOWN_MISSES) }} are shown. Click a row to open it in the search page.
      </p>
      <ul class="-mx-2 text-sm">
        <li v-for="(miss, index) in misses.slice(0, SHOWN_MISSES)" :key="index" class="border-t border-base">
          <button
            class="grid w-full gap-x-4 rounded-md px-2 py-2 text-left sm:grid-cols-[8rem_minmax(0,1fr)_minmax(0,1fr)] hover:bg-neutral-100 dark:hover:bg-neutral-800"
            @click="emit('open', miss)"
          >
            <span class="font-mono sm:order-2">{{ miss.query }}</span>
            <span class="text-xs font-mono muted sm:order-1 sm:text-sm">{{ miss.prefix }}</span>
            <span class="truncate text-xs muted sm:order-3 sm:text-sm">expected {{ expected(miss) }}</span>
          </button>
        </li>
      </ul>
    </section>
  </template>
</template>
