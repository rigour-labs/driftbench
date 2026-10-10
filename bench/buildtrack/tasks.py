"""Which merged pull requests become build-track tasks (docs/BUILD_TRACK.md, "Tasks").

A task is a merged pull request from the frozen corpus that changed at least
one test file (the hidden check) and at least one other file (the work), had
at least one acted-on human point, and has a usable statement. Its parent is
the merge base of its first head and its base commit, found at run time.
The draw is a seeded order of every eligible pull request; a run takes the
first `count` whose hidden tests discriminate (fail at the parent, pass on the
merged change), checked on the runner before any agent spends. The task file
keeps IDs, SHAs, paths and the statement's hash, never its text.
"""
from __future__ import annotations

import random
import re

from bench.buildtrack.statement import statement
from bench.collect.github import GitHubClient

TEST_RE = re.compile(r"(_test\.go|_test\.py|(^|/)test_[^/]*\.py|\.(test|spec)\.[cm]?[jt]sx?)$|(^|/)(tests?|__tests__)/")


def is_test(path: str) -> bool:
    return bool(TEST_RE.search(path))


def acted_points(points_file: dict) -> dict[int, list[str]]:
    by_pr: dict[int, list[str]] = {}
    for p in points_file["points"]:
        if p.get("acted_on") and not p.get("dropped"):
            by_pr.setdefault(p["pr"], []).append(p["id"])
    return by_pr


def candidate(pr: dict, files: list[dict], points: list[str], max_files: int | None = None) -> dict | str:
    """A task without its statement, or why the pull request is not one."""
    paths = [f["filename"] for f in files if f.get("status") != "removed"]
    tests = sorted(p for p in paths if is_test(p))
    if not pr.get("merged_at"):
        return "not merged"
    if not tests:
        return "changed no test file"
    if len(tests) == len(paths):
        return "changed only tests"
    if not points:
        return "no acted-on human point"
    if max_files and len(paths) > max_files:
        return f"changed more than {max_files} files"
    first_head = (pr.get("head_history") or [{"sha": pr["head_sha"]}])[0]["sha"]
    return {"pr": pr["number"], "base_sha": pr["base_sha"], "first_head": first_head,
            "merged_head": pr["head_sha"], "created_at": pr["created_at"], "test_files": tests,
            "changed_files": len(paths), "points": sorted(points)}


def select(client: GitHubClient, corpus: dict, points_file: dict, count: int, seed: int,
           max_files: int | None = None) -> dict:
    """{repo, seed, count, rule, tasks, excluded}: every eligible pull request in an order drawn with `seed`."""
    acted = acted_points(points_file)
    eligible, excluded = [], []
    for pr in sorted(corpus["prs"], key=lambda p: p["number"]):
        files = client.get_all(f"repos/{corpus['repo']}/pulls/{pr['number']}/files")
        task = candidate(pr, files, acted.get(pr["number"], []), max_files)
        if isinstance(task, str):
            excluded.append({"pr": pr["number"], "reason": task})
            continue
        found = statement(client, corpus["repo"], pr["number"])
        if "excluded" in found:
            excluded.append({"pr": pr["number"], "reason": found["excluded"]})
            continue
        eligible.append({**task, "statement": {k: found[k] for k in ("source", "chars", "sha256")}})
    order = random.Random(seed).sample(eligible, len(eligible))
    return {"repo": corpus["repo"], "seed": seed, "count": count, "eligible": len(eligible),
            "max_changed_files": max_files,
            "rule": "merged corpus pull requests that changed a test file and another file (at most "
                    "max_changed_files in all), with an acted-on human point and a statement as first written; "
                    "in an order drawn with the seed; a run takes the first `count` whose hidden tests fail at the "
                    "parent and pass on the merged change",
            "tasks": order, "excluded": excluded}
