<script setup lang="ts">
import type { IconSetInfo, PredictSource } from './data'
import { computed, onMounted, ref, shallowRef } from 'vue'
import { loadIconSets, loadPredictSource } from './data'
import BenchPage from './pages/BenchPage.vue'
import ReviewPage from './pages/ReviewPage.vue'
import SearchPage from './pages/SearchPage.vue'

type Page = 'search' | 'benchmark' | 'review'

const pages: { id: Page, label: string }[] = [
  { id: 'search', label: 'Search' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'review', label: 'Review' },
]

const page = ref<Page>('search')
const sets = ref<IconSetInfo[]>([])
const source = shallowRef<PredictSource>()
const prefix = ref('mdi')
const query = ref('')

onMounted(async () => {
  sets.value = await loadIconSets()
  source.value = await loadPredictSource()
})

const status = computed(() => {
  if (!source.value)
    return 'Loading the model'
  return {
    webgpu: 'Model runs on WebGPU',
    cpu: 'Model runs on the CPU',
    table: 'Static alias table',
    none: 'No predictions',
  }[source.value.mode]
})

const statusColor = computed(() => {
  if (!source.value)
    return 'bg-neutral-400 animate-pulse'
  return source.value.mode === 'webgpu' || source.value.mode === 'cpu'
    ? 'bg-teal-500'
    : 'bg-amber-500'
})

/** Open a query from another page in the search page */
function openSearch(target: { prefix: string, query: string }): void {
  prefix.value = target.prefix
  query.value = target.query
  page.value = 'search'
  window.scrollTo({ top: 0 })
}
</script>

<template>
  <div class="min-h-screen font-sans">
    <header class="sticky top-0 z-10 border-b border-base bg-neutral-50/85 backdrop-blur dark:bg-neutral-950/85">
      <div class="mx-auto max-w-7xl flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
        <h1 class="flex items-center gap-2 text-base font-semibold font-mono">
          <iconify-icon icon="ph:magnifying-glass-bold" class="accent" width="18" height="18" />
          gpu-icons-search
        </h1>
        <span class="ml-auto flex items-center gap-1.5 text-xs muted sm:order-last">
          <span class="h-2 w-2 shrink-0 rounded-full" :class="statusColor" />
          {{ status }}
        </span>
        <nav class="segmented w-full sm:w-auto">
          <button
            v-for="item in pages"
            :key="item.id"
            class="segment flex-1 sm:flex-none"
            :class="page === item.id ? 'segment-active' : ''"
            @click="page = item.id"
          >
            {{ item.label }}
          </button>
        </nav>
      </div>
    </header>

    <main class="mx-auto max-w-7xl px-4 pb-10 pt-4">
      <p v-if="source?.note" class="mb-4 border border-amber-500/30 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-800 dark:text-amber-300">
        {{ source.note }} {{ source.mode === 'table' ? 'The static alias table is in use.' : '' }}
      </p>
      <SearchPage
        v-if="page === 'search'"
        v-model:prefix="prefix"
        v-model:query="query"
        :sets="sets"
        :source="source"
      />
      <BenchPage v-else-if="page === 'benchmark'" @open="openSearch" />
      <ReviewPage v-else :source="source" @open="openSearch" />
    </main>
  </div>
</template>
