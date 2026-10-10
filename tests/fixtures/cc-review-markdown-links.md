### Code review

- [pkg/worker/loop.go:42](pkg/worker/loop.go#L42): the retry loop never stops after shutdown
- [window check](./pkg/worker/window.go#L10-L14): off-by-one in the boundary
