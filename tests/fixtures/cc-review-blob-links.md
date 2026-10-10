### Code review

Found 2 issues:

1. The retry loop never stops after shutdown

https://github.com/example-org/example-repo/blob/0123456789abcdef0123456789abcdef01234567/pkg/worker/loop.go#L40-L44

2. Off-by-one in the window check

https://github.com/example-org/example-repo/blob/0123456789abcdef0123456789abcdef01234567/pkg/worker/window.go#L12
