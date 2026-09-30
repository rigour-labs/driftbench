# solid/jsx-and-conditional

fixes scanned 1634, catches 0, HEAD hits 2 in 301710 lines (0.03 per 5k)

## Catches (fires before the fix, less after)


## HEAD sample: mark [y] right or [n] wrong

- [ ] https://github.com/solidjs/solid-start/blob/a4ed2a4cd811eb8767fc71dcf0125f1cafac6195/packages/start/src/shared/dev-toolbar/functions/SerovalViewer.tsx#L602 `cond && <X/>` is the React idiom; in Solid the branch is not keyed or disposed when `cond` changes.
- [ ] https://github.com/solidjs/solid-start/blob/a4ed2a4cd811eb8767fc71dcf0125f1cafac6195/packages/start/src/shared/dev-toolbar/functions/SerovalViewer.tsx#L601 `cond && <X/>` is the React idiom; in Solid the branch is not keyed or disposed when `cond` changes.
