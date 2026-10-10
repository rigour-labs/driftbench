"""How each repository builds and tests offline (docs/BUILD_TRACK.md, "Arms").

`prepare` runs before the agent, with the network, to fill the dependency
cache. The agent then runs with `env` (no fetching) and may use only the
`commands` as shell. `test` runs the hidden tests, given their packages.
"""
from __future__ import annotations

import dataclasses


class ToolchainError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Toolchain:
    prepare: tuple[str, ...]
    env: dict[str, str]
    commands: tuple[str, ...]          # the agent's shell, as Claude Code tool patterns
    test: tuple[str, ...]              # followed by the hidden tests' packages

    def test_command(self, test_files: list[str]) -> list[str]:
        return [*self.test, *packages(test_files)]


def packages(test_files: list[str]) -> list[str]:
    """Go packages of the test files, as ./dir paths, in order and without repeats."""
    seen: list[str] = []
    for path in test_files:
        package = "./" + path.rsplit("/", 1)[0] if "/" in path else "."
        if package not in seen:
            seen.append(package)
    return seen


GO = Toolchain(
    prepare=("go", "mod", "download"),
    env={"GOPROXY": "off", "GOFLAGS": "-mod=mod", "GOTOOLCHAIN": "local", "GOSUMDB": "off", "CGO_ENABLED": "0"},
    commands=("Bash(go build:*)", "Bash(go test:*)", "Bash(go vet:*)", "Bash(gofmt:*)"),
    test=("go", "test", "-count=1"),
)

TOOLCHAINS = {"tailscale/tailscale": GO}


def toolchain(repo: str) -> Toolchain:
    try:
        return TOOLCHAINS[repo]
    except KeyError as exc:
        raise ToolchainError(f"no offline toolchain for {repo}; known: {', '.join(sorted(TOOLCHAINS))}") from exc
