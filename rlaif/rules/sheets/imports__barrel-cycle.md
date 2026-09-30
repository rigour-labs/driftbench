# imports/barrel-cycle

fixes scanned 1634, catches 2, HEAD hits 1 in 301710 lines (0.02 per 5k)

## Catches (fires before the fix, less after)

- https://github.com/solidjs/solid-start/commit/65eee5171602946a1dff32cfcad035f647ad9b36 `packages/start/data/index.tsx:3`
- https://github.com/solidjs/solid-start/commit/65eee5171602946a1dff32cfcad035f647ad9b36 `packages/start/index.tsx:18`

## HEAD sample: mark [y] right or [n] wrong

- [ ] https://github.com/honojs/hono/blob/6abd35b0a5f35f67b6417627d5b0a6c2d266ac04/src/helper/streaming/index.ts#L9 This re-export of `./text` closes an import cycle back to this file; its exports can be undefined at first use and HMR updates loop.
