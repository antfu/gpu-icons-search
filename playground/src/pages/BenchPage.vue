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
      <h2 class="mb-1 text-sm font-semibold">
        Scores per test set
      </h2>
      <p class="mb-3 text-xs muted">
        Share of queries with a correct icon on the first screen. Matches of the typed text and other matches count as two lists.
      </p>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left muted">
            <th class="py-1 font-normal">
              Method
            </th>
            <th v-for="set in sets" :key="set" class="py-1 text-right font-normal">
              {{ set }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(scores, method) in evaluation.bench" :key="method" class="border-t border-neutral-200 dark:border-neutral-800">
            <td class="py-1 font-mono">
              {{ method }}
            </td>
            <td
              v-for="set in sets"
              :key="set"
              class="py-1 text-right font-mono"
              :class="scores[set]?.found === best(set) ? 'font-bold text-teal-700 dark:text-teal-400' : ''"
            >
              {{ percent(scores[set]?.found) }}
            </td>
          </tr>
        </tbody>
      </table>
      <dl class="grid grid-cols-[max-content_1fr] mt-4 gap-x-4 gap-y-1 text-xs">
        <dt class="muted">
          GPU adapter
        </dt>
        <dd>{{ evaluation.adapter }}</dd>
        <dt class="muted">
          Parity with PyTorch
        </dt>
        <dd>WebGPU {{ evaluation.parity.webgpu.toExponential(1) }}, reference {{ evaluation.parity.reference.toExponential(1) }}, tolerance {{ evaluation.parity.tolerance }}</dd>
        <dt class="muted">
          Weight file
        </dt>
        <dd>{{ evaluation.bytes.raw }} bytes, {{ evaluation.bytes.brotli }} with Brotli</dd>
        <dt class="muted">
          Warm call
        </dt>
        <dd>{{ evaluation.latencyMs }} ms median on that adapter</dd>
      </dl>
    </section>

    <section v-if="sweep.length > 0" class="panel mt-4 p-4">
      <h2 class="mb-1 text-sm font-semibold">
        Size against score
      </h2>
      <p class="mb-3 text-xs muted">
        The same training at {{ sweep[0]!.steps }} steps for each size, scored with the reference implementation.
      </p>
      <table class="w-full text-sm">
        <thead>
          <tr class="text-left muted">
            <th class="py-1 font-normal">
              Feature rows
            </th>
            <th class="py-1 font-normal">
              Vector length
            </th>
            <th class="py-1 text-right font-normal">
              Weight file
            </th>
            <th v-for="set in Object.keys(sweep[0]!.found)" :key="set" class="py-1 text-right font-normal">
              {{ set }}
            </th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="entry in sweep" :key="entry.bytes" class="border-t border-neutral-200 font-mono dark:border-neutral-800">
            <td class="py-1">
              {{ entry.rows }}
            </td>
            <td class="py-1">
              {{ entry.dim }}
            </td>
            <td class="py-1 text-right">
              {{ Math.round(entry.bytes / 1024) }} KB
            </td>
            <td v-for="(value, set) in entry.found" :key="set" class="py-1 text-right">
              {{ percent(value) }}
            </td>
          </tr>
        </tbody>
      </table>
    </section>

    <section class="panel mt-4 p-4">
      <div class="mb-3 flex flex-wrap items-center gap-2">
        <h2 class="text-sm font-semibold">
          Queries that the model misses
        </h2>
        <button
          v-for="set in sets.filter(set => set !== 'overall')"
          :key="set"
          class="tab"
          :class="missSet === set ? 'tab-active' : ''"
          @click="missSet = set as BenchSet"
        >
          {{ set }}
        </button>
        <label class="ml-auto flex items-center gap-1 text-xs muted">
          <input v-model="onlyRegressions" type="checkbox">
          Only where fzf+alias finds it
        </label>
      </div>
      <p class="mb-2 text-xs muted">
        {{ misses.length }} misses. The first {{ Math.min(misses.length, SHOWN_MISSES) }} are shown. Click a row to open it in the search page.
      </p>
      <table class="w-full text-sm">
        <tbody>
          <tr
            v-for="(miss, index) in misses.slice(0, SHOWN_MISSES)"
            :key="index"
            class="cursor-pointer border-t border-neutral-200 hover:bg-neutral-100 dark:border-neutral-800 dark:hover:bg-neutral-800"
            @click="emit('open', miss)"
          >
            <td class="w-32 py-1 font-mono muted">
              {{ miss.prefix }}
            </td>
            <td class="py-1 font-mono">
              {{ miss.query }}
            </td>
            <td class="py-1 muted">
              expected {{ expected(miss) }}
            </td>
          </tr>
        </tbody>
      </table>
    </section>
  </template>
</template>
