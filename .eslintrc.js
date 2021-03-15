module.exports = {
  root: true, // Make sure eslint picks up the config at the root of the directory
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module', // Allows using import/export statements
    ecmaFeatures: {
      impliedStrict: true,
      jsx: true,
    },
  },
  settings: {
    react: {
      version: 'detect', // Automatically detect the react version
    },
  },
  env: {
    browser: true, // Enables browser globals like window and document
    amd: true, // Enables require() and define() as global variables as per the amd spec.
    node: true, // Enables Node.js global variables and Node.js scoping.
  },
  extends: [
    'eslint:recommended',
    'plugin:prettier/recommended',
    'plugin:@typescript-eslint/eslint-recommended',
    'plugin:@typescript-eslint/recommended',

    'plugin:import/errors',
    'plugin:import/warnings',
  ],
  rules: {
    'prettier/prettier': ['off', {}, {usePrettierrc: true}], // Use our .prettierrc file as source
    'linebreak-style': ['error', 'unix'],
    '@typescript-eslint/no-namespace': 'off',
    'import/no-cycle': 'error',
    'import/no-unresolved': 'off',
    'import/named': 'off',
  },
};
