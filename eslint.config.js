import js from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  { ignores: ['dist', 'node_modules'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    rules: {
      // TSL node objects are loosely typed; `.toReadOnly()` and friends are not in
      // the published d.ts yet, so a few `any` casts are unavoidable in the sim layer.
      '@typescript-eslint/no-explicit-any': 'off',
    },
  },
);
