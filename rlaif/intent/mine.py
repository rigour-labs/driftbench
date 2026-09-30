"""Mine "tolerate or propagate?" examples from a repository's fix commits.

For every first-parent fix commit (arena.szz.is_fix) that edits TypeScript or
JavaScript source, the changed files before and after the fix are written to a
temporary tree and Rigour's `export-training-sites` extracts the awaited call
sites of both sides in one run per batch; pairing.label_fix turns each file's
pair into examples. Rigour's own engine does the extraction, so training data
is built the way Rigour will read code at runtime.

    RIGOUR_CLI=/path/to/rigour/packages/rigour-cli/dist/cli.js \\
        python -m rlaif.intent.mine --repo TanStack/router

Evaluation repositories (repos_training.json `_eval_repos_DO_NOT_ADD`) are refused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from arena.gitrepo import Commit, Git
from arena.repos import clone
from arena.score import in_scope
from arena.szz import MAX_FIX_FILES, is_fix
from rlaif.intent.pairing import label_fix

ROOT = Path(__file__).resolve().parent
JS_TS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts")
BATCH_FIXES = 50


def rigour_command() -> list[str]:
    cli = os.environ.get("RIGOUR_CLI")
    if cli:
        return ["node", cli]
    return ["npx", "--yes", f"@rigour-labs/cli@{os.environ.get('RIGOUR_VERSION', 'latest')}"]


def eval_repos() -> set[str]:
    data = json.loads((ROOT.parent / "repos_training.json").read_text())
    blocked = data.get("_eval_repos_DO_NOT_ADD", []) if isinstance(data, dict) else []
    return {r.lower() for r in blocked}


def mine(repo: str, max_fixes: int | None = None) -> list[dict]:
    if repo.lower() in eval_repos():
        raise SystemExit(f"{repo} is an evaluation repository; it must not become training data")
    git = clone(repo)
    root = git.run("rev-list", "--max-parents=0", "HEAD").split()[-1]
    fixes = [c for c in git.first_parent_history(root, "HEAD") if _mineable(c)]
    if max_fixes:
        fixes = fixes[-max_fixes:]
    examples: list[dict] = []
    for start in range(0, len(fixes), BATCH_FIXES):
        examples.extend(_mine_batch(git, repo, fixes[start:start + BATCH_FIXES]))
    return examples


def _mineable(commit: Commit) -> bool:
    return is_fix(commit.subject) and 0 < len(commit.files) <= MAX_FIX_FILES and bool(_code_files(commit))


def _code_files(commit: Commit) -> list[str]:
    return [f for f in commit.files if f.endswith(JS_TS) and in_scope(f, "code")]


def _mine_batch(git: Git, repo: str, fixes: list[Commit]) -> list[dict]:
    tree = Path(tempfile.mkdtemp(prefix="intent-mine-"))
    try:
        written = [w for fix in fixes for path in _code_files(fix) for w in _write_pair(git, tree, fix, path)]
        sites = _export(tree, written)
        subjects = {fix.sha: fix.subject for fix in fixes}
        return [
            {"repo": repo, "fix": sha, "subject": subjects[sha], **asdict(example), "file": path}
            for (sha, path), sides in _group(sites).items()
            for example in label_fix(sides.get("before", []), sides.get("after", []))
        ]
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def _write_pair(git: Git, tree: Path, fix: Commit, path: str) -> list[str]:
    """Both versions of a file, or nothing when the fix added or deleted it."""
    before = git.run("show", f"{fix.parent}:{path}", check=False)
    after = git.run("show", f"{fix.sha}:{path}", check=False)
    if not before or not after:
        return []
    written = []
    for side, text in (("before", before), ("after", after)):
        target = tree / fix.sha / side / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        written.append(str(target.relative_to(tree)))
    return written


def _export(tree: Path, files: list[str]) -> list[dict]:
    if not files:
        return []
    result = subprocess.run([*rigour_command(), "export-training-sites", *files], cwd=tree,
                            capture_output=True, text=True, check=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]


def _group(sites: list[dict]) -> dict[tuple[str, str], dict[str, list[dict]]]:
    """Sites keyed by (fix sha, original path) and side; paths are `<sha>/<side>/<path>`."""
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for site in sites:
        sha, side, path = site["file"].split("/", 2)
        grouped.setdefault((sha, path), {}).setdefault(side, []).append(site)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine tolerate/propagate examples from fix commits")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--max-fixes", type=int, default=0, help="Only the newest N fix commits")
    parser.add_argument("--out", default="", help="Output JSONL (default rlaif/intent/data/<owner>__<name>.jsonl)")
    args = parser.parse_args()
    examples = mine(args.repo, args.max_fixes or None)
    out = Path(args.out or ROOT / "data" / f"{args.repo.replace('/', '__')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e) + "\n" for e in examples))
    tolerate = sum(1 for e in examples if e["label"] == "tolerate")
    print(f"{args.repo}: {len(examples)} examples ({tolerate} tolerate, {len(examples) - tolerate} propagate) -> {out}")


if __name__ == "__main__":
    main()
