"""Before/after trees of fix commits, laid out so one Rigour run covers many fixes.

Each changed JS/TS file of a fix is written at `<fix sha>/before/<path>` and
`<fix sha>/after/<path>`, with the nearest package.json and tsconfig.json of each
side, so rules gated by declared dependencies or compiler options see what the
repository saw at that commit.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

from arena.gitrepo import Commit, Git
from arena.score import in_scope
from arena.szz import MAX_FIX_FILES, is_fix

TRAINING = Path(__file__).resolve().parent / "repos_training.json"
JS_TS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
MANIFESTS = ("package.json", "tsconfig.json")
SIDES = ("before", "after")
MAX_SIBLINGS = 80


def refuse_eval_repo(repo: str) -> None:
    """Evaluation repositories (arena corpora among them) never become training or validation data."""
    blocked = json.loads(TRAINING.read_text()).get("_eval_repos_DO_NOT_ADD", [])
    if repo.lower() in {r.lower() for r in blocked}:
        raise SystemExit(f"{repo} is an evaluation repository; it must not become training data")


def rigour_command() -> list[str]:
    cli = os.environ.get("RIGOUR_CLI")
    if cli:
        return ["node", cli]
    return ["npx", "--yes", f"@rigour-labs/cli@{os.environ.get('RIGOUR_VERSION', 'latest')}"]


def run_rigour(tree: Path, args: list[str]) -> list[dict]:
    """JSON lines a Rigour export command prints, run inside `tree`."""
    result = subprocess.run([*rigour_command(), *args], cwd=tree, capture_output=True, text=True, check=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def mineable(commit: Commit) -> bool:
    return is_fix(commit.subject) and 0 < len(commit.files) <= MAX_FIX_FILES and bool(code_files(commit))


def code_files(commit: Commit) -> list[str]:
    return [f for f in commit.files if f.endswith(JS_TS) and in_scope(f, "code")]


def write_pair(git: Git, tree: Path, fix: Commit, path: str, context: bool = False) -> list[str]:
    """Both versions of a changed file with their manifests. Returns the written
    changed files, relative to `tree`: the files to scan.

    A file the fix added has no pair. With `context`, a file the fix deleted keeps
    its before side (a rule that fired on it was fixed by the deletion), and the
    files beside it are written too, unscanned, so rules that compare a file with
    its siblings (entries, the types they export) see the package as it was.
    """
    shas = {"before": fix.parent, "after": fix.sha}
    texts = {side: git.run("show", f"{sha}:{path}", check=False) for side, sha in shas.items()}
    if not texts["before"] or not (texts["after"] or context):
        return []
    written = []
    for side in SIDES:
        if not texts[side]:
            continue
        root = tree / fix.sha / side
        _write(root / path, texts[side])
        written.append(str((root / path).relative_to(tree)))
        for extra in [*nearest_manifests(git, shas[side], path), *(siblings(git, shas[side], path) if context else [])]:
            if not (root / extra).exists():
                _write(root / extra, git.run("show", f"{shas[side]}:{extra}"))
    return written


def siblings(git: Git, sha: str, path: str) -> list[str]:
    """Non-test JS/TS files in the same directory as `path` at `sha`, capped."""
    directory = str(PurePosixPath(path).parent)
    listed = git.run("ls-tree", "--name-only", sha, f"{directory}/" if directory != "." else ".", check=False).splitlines()
    return [f for f in listed if f != path and f.endswith(JS_TS) and in_scope(f, "code")][:MAX_SIBLINGS]


def nearest_manifests(git: Git, sha: str, path: str) -> list[str]:
    """The closest package.json and tsconfig.json above `path` at `sha`, and the tsconfig's
    relative `extends` chain (compiler options such as stripInternal often live in a base)."""
    found: dict[str, str] = {}
    for directory in PurePosixPath(path).parents:
        for name in MANIFESTS:
            candidate = name if str(directory) == "." else f"{directory}/{name}"
            if name not in found and git.exists(sha, candidate):
                found[name] = candidate
    return [*found.values(), *_extends_chain(git, sha, found.get("tsconfig.json"))]


def _extends_chain(git: Git, sha: str, tsconfig: str | None, hops: int = 3) -> list[str]:
    chain: list[str] = []
    while tsconfig and hops:
        try:
            base = json.loads(_strip_comments(git.run("show", f"{sha}:{tsconfig}"))).get("extends")
        except (json.JSONDecodeError, RuntimeError):
            break
        if not isinstance(base, str) or not base.startswith("."):
            break
        target = os.path.normpath(str(PurePosixPath(tsconfig).parent / base))
        target = target if target.endswith(".json") else f"{target}.json"
        if not git.exists(sha, target):
            break
        chain.append(target)
        tsconfig, hops = target, hops - 1
    return chain


def _strip_comments(text: str) -> str:
    """tsconfig allows // and /* */ comments and trailing commas; JSON does not."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)^\s*//.*$", "", text)
    return re.sub(r",(\s*[}\]])", r"\1", text)


def split(file: str) -> tuple[str, str, str]:
    """(fix sha, side, original path) from a tree path `<sha>/<side>/<path>`."""
    sha, side, path = file.split("/", 2)
    return sha, side, path


def _write(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
