# paths/unnormalized-module-key

fixes scanned 1634, catches 0, HEAD hits 1 in 301710 lines (0.02 per 5k)

## Catches (fires before the fix, less after)


## HEAD sample: mark [y] right or [n] wrong

- [ ] https://github.com/web-infra-dev/rsbuild/blob/a3035109a86d471c732d53f64fd245493b947e99/packages/core/src/loader/transformLoader.ts#L43 `result` holds a bundler module path in OS form and is used as a lookup key; on Windows it has backslashes and never matches.
