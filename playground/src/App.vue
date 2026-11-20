<script setup lang="ts">
import type { IconSetInfo, PredictSource } from './data'
import { onMounted, ref, shallowRef } from 'vue'
import { loadIconSets, loadPredictSource } from './data'
import BenchPage from './pages/BenchPage.vue'
import ReviewPage from './pages/ReviewPage.vue'
import SearchPage from './pages/SearchPage.vue'

type Page = 'search' | 'benchmark' | 'review'

const pages: { id: Page, label: string }[] = [
  { id: 'search', label: 'Search' },
  { id: 'benchmark', label: 'Benchmark' },
  { id: 'review', label: 'Query review' },
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

/** Open a query from another page in the search page */
function openSearch(target: { prefix: string, query: string }): void {
  prefix.value = target.prefix
  query.value = target.query
  page.value = 'search'
}
</script>

<template>
  <div class="mx-auto max-w-7xl px-4 py-4 font-sans">
    <header class="mb-4 flex flex-wrap items-center gap-3">
      <h1 class="text-lg font-semibold font-mono">
        gpu-icons-search
      </h1>
      <nav class="flex gap-1">
        <button
          v-for="item in pages"
          :key="item.id"
          class="tab"
          :class="page === item.id ? 'tab-active' : ''"
          @click="page = item.id"
        >
          {{ item.label }}
        </button>
      </nav>
      <span class="ml-auto text-xs muted">
        <template v-if="!source">Loading the model</template>
        <template v-else-if="source.mode === 'webgpu'">Model runs on WebGPU</template>
        <template v-else>{{ source.note }} {{ source.mode === 'table' ? 'The static alias table is in use.' : '' }}</template>
      </span>
    </header>

    <SearchPage
      v-if="page === 'search'"
      v-model:prefix="prefix"
      v-model:query="query"
      :sets="sets"
      :source="source"
    />
    <BenchPage v-else-if="page === 'benchmark'" @open="openSearch" />
    <ReviewPage v-else :source="source" @open="openSearch" />
  </div>
</template>
