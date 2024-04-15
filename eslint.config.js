import antfu from '@antfu/eslint-config'

export default antfu({
  react: true,
}, {
  rules: {
    'padding-line-between-statements': [
      'error',
      { blankLine: 'always', prev: '*', next: 'return' },
    ],
  },
})
