"""Which review instructions each paid entrant ran, by hash only (docs/ENTRANTS.md, "Instructions").

Claude Code's `/code-review` is built into its platform-native binary, which
is proprietary: its text is never extracted, copied or published. Rigour's
reviewer prompt is pinned the same way, so the two entrants are recorded
symmetrically:

- in run.json (the start job): each package's name, version and npm
  integrity, and Rigour's own `PROMPT_VERSION` (a sha256 of its rendered
  reviewer prompt);
- per review job, after the reviews: the sha256 of each file that holds the
  instructions as installed (`python -m bench.harness.prompts installed`).
Anyone can install the same versions and compare.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from bench.adapters.claude_cli import CLAUDE_CODE_VERSION
from bench.adapters.rigour import VERSION as RIGOUR_VERSION

Runner = Callable[[list[str]], str]
CODE_REVIEW = "/code-review"
ARCH = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
PROMPT_JS = Path("dist") / "review" / "reviewer" / "prompt.js"


def run_text(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"{' '.join(args[:3])}: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def native_package(system: str | None = None, machine: str | None = None) -> str:
    """Claude Code's native package for this platform, e.g. @anthropic-ai/claude-code-linux-x64."""
    name = (system or platform.system()).lower()
    os_tag = {"darwin": "darwin", "linux": "linux", "windows": "win32"}.get(name, name)
    return f"@anthropic-ai/claude-code-{os_tag}-{ARCH.get((machine or platform.machine()).lower(), machine)}"


def integrity(spec: str, run: Runner = run_text) -> str:
    found = run(["npm", "view", spec, "dist.integrity"])
    if not found.startswith("sha"):
        raise ValueError(f"no npm integrity for {spec}")
    return found


def rigour_prompt_version(core_dir: Path, run: Runner = run_text) -> str:
    """Rigour's PROMPT_VERSION, imported from the installed core at the pinned version."""
    url = (core_dir / PROMPT_JS).resolve().as_uri()
    return run(["node", "--input-type=module", "-e", f"console.log((await import({json.dumps(url)})).PROMPT_VERSION)"])


def prompt_record(names: list[str], rigour_core: Path | None, run: Runner | None = None) -> dict:
    """run.json's `prompts`: per paid entrant, the package that holds its instructions and how to check it."""
    run = run or run_text
    record = {}
    if "claude-code-review" in names:
        spec = f"{native_package()}@{CLAUDE_CODE_VERSION}"
        record["claude-code-review"] = {"command": CODE_REVIEW, "package": spec, "integrity": integrity(spec, run),
                                        "text": "built into the native binary; pinned by its integrity, never extracted"}
    if any(name.startswith("rigour-reviewer") for name in names):
        if rigour_core is None:
            raise ValueError("--rigour-core (the installed @rigour-labs/core) is required to record Rigour's prompt")
        spec = f"@rigour-labs/core@{RIGOUR_VERSION}"
        record["rigour-reviewer"] = {"package": spec, "integrity": integrity(spec, run),
                                     "prompt_version": rigour_prompt_version(rigour_core, run)}
    return record


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed(global_root: Path, npx_cache: Path) -> dict:
    """The sha256 of each installed file that holds an entrant's instructions."""
    found = {}
    native = global_root / "@anthropic-ai" / "claude-code" / "node_modules" / native_package()
    for root in (native, global_root / native_package()):
        files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else []
        if files:
            found["claude-code-review"] = {str(p.relative_to(root)): sha256_file(p) for p in files}
            break
    for core in sorted(npx_cache.glob("*/node_modules/@rigour-labs/core")):
        try:
            package = json.loads((core / "package.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"unreadable {core / 'package.json'}: {exc}") from exc
        if package.get("version") == RIGOUR_VERSION and (core / PROMPT_JS).is_file():
            found["rigour-reviewer"] = {str(PROMPT_JS): sha256_file(core / PROMPT_JS)}
            break
    return found


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "installed":
        print("usage: python -m bench.harness.prompts installed <out.json>", file=sys.stderr)
        return 2
    try:
        found = installed(Path(run_text(["npm", "root", "-g"])), Path.home() / ".npm" / "_npx")
    except (OSError, ValueError) as exc:
        print(f"prompts: {exc}", file=sys.stderr)
        return 1
    Path(argv[1]).write_text(json.dumps(found, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
