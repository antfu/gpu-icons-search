import { defineConfig, presetWind3 } from 'unocss'

export default defineConfig({
  presets: [
    presetWind3({ dark: 'media' }),
  ],
  shortcuts: {
    'border-base': 'border-neutral-200 dark:border-neutral-800',
    'panel': 'border border-base rounded-xl bg-white dark:bg-neutral-900',
    'muted': 'text-neutral-500 dark:text-neutral-400',
    'accent': 'text-teal-700 dark:text-teal-400',
    'chip': 'inline-flex shrink-0 items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono border border-base bg-white dark:bg-neutral-900',
    'badge': 'inline-flex items-center px-1.5 py-0.5 rounded-md text-[11px] font-mono tabular-nums bg-neutral-100 dark:bg-neutral-800 muted',
    'segmented': 'inline-flex gap-0.5 p-0.5 rounded-lg bg-neutral-200/60 dark:bg-neutral-800/80',
    'segment': 'px-3 py-1.5 rounded-md text-sm whitespace-nowrap cursor-pointer muted hover:text-neutral-900 dark:hover:text-neutral-100',
    'segment-active': 'bg-white text-neutral-900 shadow-sm dark:bg-neutral-950 dark:text-neutral-100',
    'tab': 'shrink-0 px-2.5 py-1 rounded-full text-xs font-mono border border-base cursor-pointer muted hover:text-neutral-900 dark:hover:text-neutral-100',
    'tab-active': 'border-neutral-900 bg-neutral-900 text-white hover:text-white dark:border-neutral-100 dark:bg-neutral-100 dark:text-neutral-900 dark:hover:text-neutral-900',
    'field': 'h-11 px-3 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white text-neutral-900 dark:bg-neutral-900 dark:text-neutral-100 text-base sm:text-sm outline-none focus:border-teal-600 focus:ring-3 focus:ring-teal-600/15',
    'section-title': 'text-sm font-semibold',
    'scroll-x': 'overflow-x-auto overscroll-x-contain [scrollbar-width:none] [&::-webkit-scrollbar]:hidden',
  },
})
