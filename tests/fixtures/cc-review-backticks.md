## Code review

Found 2 issues:

1. The retry loop never stops after shutdown (`pkg/worker/loop.go:42`)
2. Off-by-one in the window check: `pkg/worker/window.go:L10-L14`

Also see pkg/worker/loop.go:42 above (same issue).
