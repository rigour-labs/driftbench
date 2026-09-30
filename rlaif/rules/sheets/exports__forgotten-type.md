# exports/forgotten-type

fixes scanned 1634, catches 0, HEAD hits 41 in 301710 lines (0.68 per 5k)

## Catches (fires before the fix, less after)


## HEAD sample: mark [y] right or [n] wrong

- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/integrations/index.ts#L12 `useSortable` is exported, but the signature uses `MaybeElement`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/shared/index.ts#L74 `whenever` is exported, but the signature uses `Truthy`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/math/index.ts#L13 `useMin` is exported, but the signature uses `MaybeComputedRefArgs`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/solidjs/solid-router/blob/be412bca1e7cf700090a029cc73cd3b7670aeb04/src/index.tsx#L3 `createBeforeLeave` is exported, but the signature uses `BeforeLeaveLifecycle`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/solidjs/solid-router/blob/be412bca1e7cf700090a029cc73cd3b7670aeb04/src/index.tsx#L19 `useSubmission`, `useSubmissions` are exported, but their signatures use `NarrowResponse`, `SubmissionStub`, which this entry does not export; consumers cannot name them.
- [ ] https://github.com/honojs/hono/blob/6abd35b0a5f35f67b6417627d5b0a6c2d266ac04/adapters/aws-lambda/src/index.ts#L7 `getConnInfo` is exported, but the signature uses `Env`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vuejs/pinia/blob/98587ca465b2c45e4053548261e769cad380ba5a/packages/pinia/src/index.ts#L63 `storeToRefs` is exported, but the signature uses `StoreToRefs`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/math/index.ts#L11 `useMath` is exported, but their signatures use `ArgumentsType`, `Reactified`, which this entry does not export; consumers cannot name them.
- [ ] https://github.com/honojs/hono/blob/6abd35b0a5f35f67b6417627d5b0a6c2d266ac04/adapters/lambda-edge/src/index.ts#L6 `handle` is exported, but their signatures use `CloudFrontContext`, `CloudFrontResult`, which this entry does not export; consumers cannot name them.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/core/index.ts#L144 `useWebWorker` is exported, but the signature uses `WorkerFn`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/solidjs/solid-router/blob/be412bca1e7cf700090a029cc73cd3b7670aeb04/src/index.tsx#L18 `_mergeSearchString` is exported, but the signature uses `SetSearchParams`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/solidjs/solid-router/blob/be412bca1e7cf700090a029cc73cd3b7670aeb04/src/index.tsx#L1 `createRouter` is exported, but the signature uses `RouterContext`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/rxjs/index.ts#L7 `watchExtractedObservable` is exported, but their signatures use `MapSources`, `MapOldSources`, which this entry does not export; consumers cannot name them.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/router/index.ts#L2 `useRouteParams` is exported, but the signature uses `ReactiveRouteOptionsWithTransform`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/router/index.ts#L3 `useRouteQuery` is exported, but the signature uses `ReactiveRouteOptionsWithTransform`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/shared/index.ts#L3 `createDisposableDirective` is exported, but the signature uses `originDirective`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/math/index.ts#L17 `useSum` is exported, but the signature uses `MaybeComputedRefArgs`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/integrations/index.ts#L2 `useAxios` is exported, but the signature uses `OverallUseAxiosReturn`, which this entry does not export; consumers cannot name it.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/router/index.ts#L1 `useRouteHash` is exported, but their signatures use `RouteHashValueRaw`, `ReactiveRouteOptions`, which this entry does not export; consumers cannot name them.
- [ ] https://github.com/vueuse/vueuse/blob/efdd69a1481205051e85d9c815eaa5840237f2d1/packages/rxjs/index.ts#L3 `useExtractedObservable` is exported, but their signatures use `MapSources`, `MapOldSources`, which this entry does not export; consumers cannot name them.
