import { defineConfig, presetWind3 } from 'unocss'

export default defineConfig({
  presets: [
    presetWind3({ dark: 'media' }),
  ],
  shortcuts: {
    'panel': 'border border-neutral-200 dark:border-neutral-800 rounded-lg bg-white dark:bg-neutral-900',
    'muted': 'text-neutral-500 dark:text-neutral-400',
    'chip': 'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-mono border border-neutral-200 dark:border-neutral-700',
    'tab': 'px-3 py-1.5 rounded-md text-sm cursor-pointer hover:bg-neutral-100 dark:hover:bg-neutral-800',
    'tab-active': 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900 hover:bg-neutral-900 dark:hover:bg-neutral-100',
    'field': 'px-3 py-2 rounded-md border border-neutral-300 dark:border-neutral-700 bg-transparent outline-none focus:border-teal-600',
  },
})
