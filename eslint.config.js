// @ts-check
import antfu from '@antfu/eslint-config'

export default antfu(
  {
    type: 'lib',
    pnpm: true,
    antislop: true,
    vue: true,
    ignores: ['playground/index.html'],
  },
  {
    // The training package holds command line scripts, which report progress on stdout
    files: ['training/**/*.ts'],
    rules: {
      'no-console': 'off',
    },
  },
)
