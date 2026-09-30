"""Review training pairs from a repository's own history, labelled by its later fixes.

A fix commit's removed or replaced lines are blamed just before it; the commit
that wrote them introduced the defect. That commit's diff is what a reviewer
saw, so it becomes a prompt, built by Rigour's `export-review-context` (the
same code the max tier reviews with), and the target is the finding the fix
implies: the blamed line and the fix's own description. Commits whose changed
lines no fix ever touched, old enough to have been fixed, are the negatives:
their target is no findings.

    RIGOUR_CLI=.../rigour-cli/dist/cli.js python -m rlaif.review.mine_pairs --repo owner/name

Labels cost nothing: they come from git history. Evaluation repositories are
refused, so the arena stays unseen.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from arena.gitrepo import Commit, Git
from arena.repos import clone
from arena.szz import TRIVIAL_LINE, is_fix
from rlaif.fixtrees import code_files, mineable, refuse_eval_repo, run_rigour

ROOT = Path(__file__).resolve().parent
CLEAN_AFTER_DAYS = 180
MAX_FILES_PER_COMMIT = 8
PREFIX = re.compile(r"^(?:fix(?:es|ed)?|bug(?:fix)?|hotfix|revert)(?:\([^)]*\))?!?:\s*", re.IGNORECASE)


@dataclass
class Site:
    """Lines one commit wrote in one file that a later fix changed."""
    commit: str
    path: str
    lines: set[int] = field(default_factory=set)
    fixes: list[Commit] = field(default_factory=list)
    #: What the first fix wrote in place of these lines.
    replacement: list[str] = field(default_factory=list)


def introducing_sites(git: Git, history: list[Commit]) -> dict[tuple[str, str], Site]:
    """(commit, path) -> the lines it wrote that later fixes changed, from every fix in `history`."""
    first_parent = {c.sha for c in history}
    sites: dict[tuple[str, str], Site] = {}
    for fix in history:
        if not mineable(fix):
            continue
        for path in code_files(fix):
            hunks = [h for h in git.hunks(fix.parent, fix.sha, path) if h.old_count > 0]
            if not hunks:
                continue
            after = git.run("show", f"{fix.sha}:{path}", check=False).splitlines()
            for hunk in hunks:  # blamed per hunk, so each line knows what the fix wrote in its place
                replacement = after[hunk.new_start - 1:hunk.new_start - 1 + min(hunk.new_count, 3)]
                for origin, line, text, origin_path in git.blame_ranges(fix.parent, path, [(hunk.old_start, hunk.old_end)]):
                    if origin not in first_parent or origin_path != path or TRIVIAL_LINE.match(text):
                        continue
                    site = sites.setdefault((origin, path), Site(origin, path))
                    site.lines.add(line)
                    if fix not in site.fixes:
                        site.fixes.append(fix)
                        site.replacement = replacement
    return sites


def clean_commits(git: Git, history: list[Commit], buggy: set[str], now: datetime) -> list[Commit]:
    """Commits changing code that no fix traced back to, old enough that a bug would likely have been fixed."""
    cutoff = now - timedelta(days=CLEAN_AFTER_DAYS)
    dates = _commit_dates(git, [c.sha for c in history])
    return [c for c in history if c.sha not in buggy and not is_fix(c.subject) and code_files(c)
            and dates.get(c.sha, now) <= cutoff]


def target(site: Site | None, path: str) -> dict:
    """The review a model should give: the fix-implied finding, or nothing.

    The description is the fix's own subject without its conventional-commit
    prefix ("fix(router): preserve raw params" -> "Preserve raw params."), and the
    suggestion is what the fix wrote there: the words of the people who fixed it.
    """
    if site is None:
        return {"findings": []}
    subject = PREFIX.sub("", site.fixes[0].subject).strip().rstrip(".")
    subject = re.sub(r"\s*\(#\d+\)$", "", subject)
    return {"findings": [{"category": "correctness", "severity": "high", "file": path, "line": min(site.lines),
                          "description": f"{subject[:1].upper()}{subject[1:]}.",
                          "suggestion": "\n".join(site.replacement) or subject, "confidence": 0.8}]}


def mine(repo: str, max_commits: int, seed: int = 7) -> list[dict]:
    refuse_eval_repo(repo)
    git = clone(repo)
    head = git.run("rev-parse", "origin/HEAD").strip()
    history = git.first_parent_history(git.run("rev-list", "--max-parents=0", head).split()[-1], head)
    sites = introducing_sites(git, history)
    by_commit: dict[str, dict[str, Site]] = defaultdict(dict)
    for (sha, path), site in sites.items():
        by_commit[sha][path] = site
    rng = random.Random(seed)
    buggy = sorted(by_commit)
    rng.shuffle(buggy)
    clean = [c.sha for c in clean_commits(git, history, set(by_commit), datetime.now(timezone.utc))]
    rng.shuffle(clean)
    half = max_commits // 2
    examples: list[dict] = []
    for sha in buggy[:half] + clean[:half]:
        examples.extend(_examples_at(git, repo, sha, by_commit.get(sha, {})))
    return balance(examples, rng)


def balance(examples: list[dict], rng: random.Random, clean_per_bug: int = 1) -> list[dict]:
    """Every bug example and at most `clean_per_bug` clean ones per bug: a clean
    commit touches many files, and a model trained mostly on clean files learns
    to say nothing."""
    bugs = [e for e in examples if e["label"] == "bug"]
    clean = [e for e in examples if e["label"] == "clean"]
    rng.shuffle(clean)
    return bugs + clean[:clean_per_bug * len(bugs)]


def _examples_at(git: Git, repo: str, sha: str, sites: dict[str, Site]) -> list[dict]:
    """One example per changed code file of the commit, reviewed as it was merged."""
    workdir = Path(tempfile.mkdtemp(prefix="review-pairs-"))
    tree = workdir / "tree"
    try:
        git.run("worktree", "add", "-q", "--detach", str(tree), sha)
        parent = git.parent(sha)
        files = [f for f in git.changed_files(parent, sha) if f.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs"))][:MAX_FILES_PER_COMMIT]
        if not files:
            return []
        diff = workdir / "change.diff"
        diff.write_text(git.run("diff", "--no-color", parent, sha, "--", *files))
        return [{"repo": repo, "commit": sha, "file": c["file"], "label": "bug" if c["file"] in sites else "clean",
                 "fix": sites[c["file"]].fixes[0].sha if c["file"] in sites else None,
                 "prompt": c["prompt"], "target": json.dumps(target(sites.get(c["file"]), c["file"]))}
                for c in run_rigour(tree, ["export-review-context", "--diff", str(diff)])]
    except Exception as error:  # a commit that no longer checks out or builds a context is skipped, not fatal
        print(f"skip {sha[:10]}: {str(error)[:120]}")
        return []
    finally:
        git.run("worktree", "remove", "--force", str(tree), check=False)
        shutil.rmtree(workdir, ignore_errors=True)


def _commit_dates(git: Git, shas: list[str]) -> dict[str, datetime]:
    out = git.run("show", "-s", "--format=%H %cI", *shas) if shas else ""
    return {sha: datetime.fromisoformat(date) for sha, date in (line.split(" ", 1) for line in out.splitlines() if " " in line)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine review training pairs labelled by later fixes")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--max-commits", type=int, default=400, help="Half with later-fixed lines, half clean")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    examples = mine(args.repo, args.max_commits)
    out = Path(args.out or ROOT / "data" / f"{args.repo.replace('/', '__')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e) + "\n" for e in examples))
    bugs = sum(1 for e in examples if e["label"] == "bug")
    print(f"{args.repo}: {len(examples)} examples ({bugs} bug, {len(examples) - bugs} clean) -> {out}")


if __name__ == "__main__":
    main()
