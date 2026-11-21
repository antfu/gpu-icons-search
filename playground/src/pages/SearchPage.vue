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

const EXAMPLES = [
  'hapy',
  'bin',
  'settings gear',
  'shopping',
  'cloud upload',
  'heart rate',
]

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

/** Result column shown on narrow screens, where the two columns do not fit side by side */
const column = ref<'model' | 'today'>('model')

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
  <div class="grid grid-cols-[1fr_auto] gap-2 sm:grid-cols-[minmax(0,14rem)_1fr_auto]">
    <label class="relative col-span-2 sm:col-span-1 sm:col-start-2 sm:row-start-1">
      <span class="sr-only">Query</span>
      <iconify-icon icon="ph:magnifying-glass" class="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 muted" width="18" height="18" />
      <input
        v-model="query"
        type="search"
        class="field w-full pl-10"
        placeholder="A word, a typo, or a short description"
        autofocus
        spellcheck="false"
        autocapitalize="off"
        autocomplete="off"
        enterkeyhint="search"
      >
    </label>
    <label class="min-w-0 sm:col-start-1 sm:row-start-1">
      <span class="sr-only">Icon set</span>
      <select v-model="prefix" class="field w-full">
        <option v-for="set in sets" :key="set.prefix" :value="set.prefix">
          {{ set.name }} ({{ set.total }})
        </option>
      </select>
    </label>
    <label class="field flex items-center gap-2 muted" title="How many words the model predicts">
      <span class="text-xs">Words</span>
      <input v-model.number="limit" type="range" min="1" max="16" class="w-20 accent-teal-600 sm:w-24">
      <span class="w-5 text-right text-sm font-mono text-neutral-900 dark:text-neutral-100">{{ limit }}</span>
    </label>
  </div>

  <div class="mt-3 flex items-center gap-2">
    <span class="shrink-0 text-xs muted">Predicted</span>
    <div class="scroll-x flex flex-1 gap-1.5 py-0.5 sm:flex-wrap">
      <span v-if="predictions.length === 0" class="text-xs muted">none yet</span>
      <span
        v-for="prediction in predictions"
        :key="prediction.word"
        class="chip"
        :class="prediction.kind !== 'related' ? 'border-teal-500/40 bg-teal-500/5' : ''"
        :title="`${prediction.kind}, from the typed text ${prediction.from}`"
      >
        {{ prediction.word }}
        <span class="muted tabular-nums">{{ prediction.score.toFixed(2) }}</span>
        <span v-if="prediction.kind !== 'related'" class="accent">{{ prediction.kind }}</span>
      </span>
    </div>
  </div>

  <section v-if="!query.trim()" class="panel mt-4 px-4 py-10 text-center">
    <iconify-icon icon="ph:magnifying-glass" class="muted" width="32" height="32" />
    <p class="mt-2 text-sm">
      Compare the model with the current Icones search.
    </p>
    <p class="mt-1 text-xs muted">
      Try a typo, a synonym or a description:
    </p>
    <div class="mt-3 flex flex-wrap justify-center gap-1.5">
      <button v-for="example in EXAMPLES" :key="example" class="chip cursor-pointer hover:border-teal-500/60" @click="query = example">
        {{ example }}
      </button>
    </div>
  </section>

  <template v-else>
    <div class="segmented mt-4 w-full lg:hidden">
      <button class="segment flex-1" :class="column === 'model' ? 'segment-active' : ''" @click="column = 'model'">
        gpu-icons-search <span class="badge ml-1">{{ tiered.length }}</span>
      </button>
      <button class="segment flex-1" :class="column === 'today' ? 'segment-active' : ''" @click="column = 'today'">
        Icones today <span class="badge ml-1">{{ today.length }}</span>
      </button>
    </div>

    <div class="grid mt-3 gap-4 lg:mt-4 lg:grid-cols-2">
      <section class="panel p-3 sm:p-4 lg:order-last lg:block" :class="column === 'model' ? '' : 'hidden'">
        <h2 class="flex items-center gap-2 section-title">
          gpu-icons-search
          <span class="badge">{{ typedCount }} typed · {{ tiered.length - typedCount }} predicted</span>
        </h2>
        <p class="mb-3 mt-1 text-xs muted">
          fzf on the typed text, then each predicted word. Prediction {{ ms(times.predict) }}, matching {{ ms(times.match) }}.
        </p>
        <IconGrid :prefix="prefix" :results="tiered.slice(0, Math.min(typedCount, SHOWN / 2))" :limit="SHOWN" />
        <p v-if="tiered.length > typedCount" class="my-3 flex items-center gap-2 text-xs font-medium accent">
          <iconify-icon icon="ph:sparkle" width="14" height="14" />
          Found by predicted words
          <span class="h-px flex-1 bg-teal-500/20" />
        </p>
        <IconGrid :prefix="prefix" :results="tiered.slice(typedCount)" :limit="SHOWN" />
        <p v-if="query.trim() && tiered.length === 0" class="py-8 text-center text-sm muted">
          No results
        </p>
      </section>
      <section class="panel p-3 sm:p-4 lg:block" :class="column === 'today' ? '' : 'hidden'">
        <h2 class="flex items-center gap-2 section-title">
          Icones today
          <span class="badge">{{ today.length }} results</span>
        </h2>
        <p class="mb-3 mt-1 text-xs muted">
          fzf with the manual alias list, in {{ ms(times.today) }}.
        </p>
        <IconGrid :prefix="prefix" :results="today" :limit="SHOWN" />
        <p v-if="query.trim() && today.length === 0" class="py-8 text-center text-sm muted">
          No results
        </p>
      </section>
    </div>
  </template>
</template>
