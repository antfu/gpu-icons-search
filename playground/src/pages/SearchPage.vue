<script setup lang="ts">
import type { Prediction } from 'gpu-icons-search'
import type { IconSearch } from '../../../training/src/search'
import type { IconSetInfo, PredictSource } from '../data'
import { computed, ref, shallowRef, watch } from 'vue'
import aliasData from '../../../training/data/icones-alias.json'
import IconGrid from '../components/IconGrid.vue'
import { loadSearch } from '../data'

const props = defineProps<{
  sets: IconSetInfo[]
  source?: PredictSource
}>()

const prefix = defineModel<string>('prefix', { required: true })
const query = defineModel<string>('query', { required: true })

/** Results drawn per column */
const SHOWN = 72

const limit = ref(8)
const search = shallowRef<IconSearch>()
const today = shallowRef<{ icon: string }[]>([])
const tiered = shallowRef<{ icon: string, via?: Prediction }[]>([])
const predictions = shallowRef<Prediction[]>([])
const times = ref({ today: 0, predict: 0, match: 0 })
let controller: AbortController | undefined

const typedCount = computed(() => tiered.value.filter(result => !result.via).length)

watch(prefix, async (value) => {
  search.value = undefined
  search.value = await loadSearch(value)
}, { immediate: true })

watch([
  query,
  search,
  limit,
  () => props.source,
], async () => {
  const icons = search.value
  const text = query.value
  controller?.abort()
  if (!icons || !text.trim()) {
    today.value = []
    tiered.value = []
    predictions.value = []
    return
  }
  let started = performance.now()
  today.value = icons.icones(text, aliasData.groups).map(icon => ({ icon }))
  times.value.today = performance.now() - started

  controller = new AbortController()
  started = performance.now()
  let predicted: Prediction[] = []
  try {
    predicted = await props.source?.predict(text, limit.value, controller.signal) ?? []
  }
  catch {
    // A newer key press aborted this call
    return
  }
  times.value.predict = performance.now() - started

  started = performance.now()
  predictions.value = predicted
  tiered.value = icons.tieredWithReasons(text, predicted)
  times.value.match = performance.now() - started
}, { immediate: true })

function ms(value: number): string {
  return `${value.toFixed(1)} ms`
}
</script>

<template>
  <div class="flex flex-wrap items-center gap-2">
    <select v-model="prefix" class="field">
      <option v-for="set in sets" :key="set.prefix" :value="set.prefix">
        {{ set.name }} ({{ set.total }})
      </option>
    </select>
    <input
      v-model="query"
      class="field min-w-60 flex-1"
      placeholder="Type a word, a typo, or a short description"
      autofocus
      spellcheck="false"
    >
    <label class="flex items-center gap-2 text-sm muted">
      Predictions
      <input v-model.number="limit" type="range" min="1" max="16">
      <span class="w-5 font-mono">{{ limit }}</span>
    </label>
  </div>

  <div class="mt-3 min-h-8 flex flex-wrap items-center gap-1.5">
    <span class="text-xs muted">Predicted words:</span>
    <span v-if="predictions.length === 0" class="text-xs muted">none</span>
    <span
      v-for="prediction in predictions"
      :key="prediction.word"
      class="chip"
      :title="`${prediction.kind}, from the typed text ${prediction.from}`"
    >
      {{ prediction.word }}
      <span class="muted">{{ prediction.score.toFixed(2) }}</span>
      <span v-if="prediction.kind !== 'related'" class="text-teal-700 dark:text-teal-400">{{ prediction.kind }}</span>
    </span>
  </div>

  <div class="grid mt-3 gap-4 lg:grid-cols-2">
    <section class="panel p-3">
      <h2 class="text-sm font-semibold">
        Icones today
      </h2>
      <p class="mb-2 text-xs muted">
        fzf with the manual alias list. {{ today.length }} results in {{ ms(times.today) }}.
      </p>
      <IconGrid :prefix="prefix" :results="today" :limit="SHOWN" />
    </section>
    <section class="panel p-3">
      <h2 class="text-sm font-semibold">
        gpu-icons-search
      </h2>
      <p class="mb-2 text-xs muted">
        fzf on the typed text, then each predicted word. {{ typedCount }} typed and {{ tiered.length - typedCount }} predicted results.
        Prediction {{ ms(times.predict) }}, matching {{ ms(times.match) }}.
      </p>
      <IconGrid :prefix="prefix" :results="tiered.slice(0, Math.min(typedCount, SHOWN / 2))" :limit="SHOWN" />
      <p v-if="tiered.length > typedCount" class="my-2 border-t border-neutral-200 pt-2 text-xs muted dark:border-neutral-800">
        Found by predicted words
      </p>
      <IconGrid :prefix="prefix" :results="tiered.slice(typedCount)" :limit="SHOWN" />
    </section>
  </div>
</template>
