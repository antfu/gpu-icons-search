<script setup lang="ts">
import type { Prediction } from 'gpu-icons-search'

defineProps<{
  prefix: string
  /** Icon names with the prediction that found each one, if any */
  results: { icon: string, via?: Prediction }[]
  /** How many results to draw */
  limit: number
}>()
</script>

<template>
  <div class="grid grid-cols-[repeat(auto-fill,minmax(4.5rem,1fr))] gap-1 sm:grid-cols-[repeat(auto-fill,minmax(5.5rem,1fr))]">
    <div
      v-for="result in results.slice(0, limit)"
      :key="result.icon"
      class="flex flex-col items-center gap-1.5 rounded-lg px-1 py-2.5 text-center"
      :class="result.via ? 'bg-teal-500/10' : 'hover:bg-neutral-100 dark:hover:bg-neutral-800/60'"
      :title="result.via ? `${result.icon} (found by ${result.via.word})` : result.icon"
    >
      <iconify-icon :icon="`${prefix}:${result.icon}`" width="26" height="26" />
      <span class="line-clamp-2 w-full break-words text-[10px] leading-tight muted">{{ result.icon }}</span>
      <span v-if="result.via" class="text-[10px] font-mono accent">{{ result.via.word }}</span>
    </div>
  </div>
</template>
