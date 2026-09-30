"""Pre-emption: how much of CodeRabbit's useful review a tool would have raised before the PR.

A CodeRabbit comment the developer acted on (the flagged lines changed before
merge) is review work someone valued. If a tool run during development flags
the same place, that comment never needs to be written: the PR arrives clean.

For each PR the tool reviews the change as it stood at the commit CodeRabbit
reviewed (the one with the most acted-on comments), against the PR's base, so
it sees exactly what CodeRabbit saw. A target is pre-empted when a finding
lands in the same file within `tolerance` lines of it. Proximity is an upper
bound: a sampled set of matches is judged (`preempt-sample`) to report how many
describe the same issue.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from arena.corpus import BotComment, Pr
from arena.diffmap import touches
from arena.gitrepo import Git
from arena.repos import ensure_commits
from arena.score import DEFAULT_TOLERANCE, BOOTSTRAP_ROUNDS, Metric, _interval, in_scope
from arena.tools.rigour import RigourConfig, review_range


@dataclass(frozen=True)
class Target:
    id: str
    path: str
    start: int
    end: int
    severity: str


def acted_on_targets(git: Git, pr: Pr) -> tuple[str, list[Target]]:
    """(reviewed commit, its acted-on code comments). Empty when nothing was acted on."""
    acted: dict[str, list[Target]] = {}
    for comment in pr.comments:
        if not in_scope(comment.path, "code") or not _placeable(git, pr, comment):
            continue
        if touches(git.hunks(comment.commit, pr.merge_sha, comment.path), comment.start, comment.end):
            acted.setdefault(comment.commit, []).append(
                Target(str(comment.id), comment.path, comment.start, comment.end, comment.severity or ""))
    if not acted:
        return "", []
    commit = max(sorted(acted), key=lambda sha: len(acted[sha]))
    return commit, acted[commit]


def _placeable(git: Git, pr: Pr, comment: BotComment) -> bool:
    return git.has_commit(comment.commit) and git.exists(comment.commit, comment.path) and git.exists(pr.merge_sha, comment.path)


def run(git: Git, prs: list[Pr], cfg: RigourConfig) -> dict:
    """Review each PR at its reviewed commit; record targets and findings."""
    out: dict[str, dict] = {}
    for pr in prs:
        ensure_commits(git, pr.number, [c.commit for c in pr.comments])
        commit, targets = acted_on_targets(git, pr)
        if not targets:
            continue
        base = git.run("merge-base", git.parent(pr.merge_sha), commit).strip()
        result = review_range(git, base, commit, cfg)
        out[str(pr.number)] = {
            "commit": commit, "base": base, "status": result.status, "error": result.error, "seconds": round(result.seconds, 2),
            "targets": [t.__dict__ for t in targets],
            "findings": [{"path": f.path, "line": f.line, "id": f.id, "message": f.message} for f in result.findings],
        }
    return {"tool": cfg.name, "stage": "pre-pr", "repo": "", "prs": out}


def matches(entry: dict, tolerance: int = DEFAULT_TOLERANCE) -> list[tuple[dict, dict]]:
    """(target, finding) pairs, one-to-one: each finding pre-empts at most one comment."""
    used: set[int] = set()
    pairs = []
    for target in entry["targets"]:
        for index, finding in enumerate(entry["findings"]):
            near = target["start"] - tolerance <= finding["line"] <= target["end"] + tolerance
            if index not in used and finding["path"] == target["path"] and near:
                used.add(index)
                pairs.append((target, finding))
                break
    return pairs


@dataclass
class Preemption:
    prs: int
    targets: int
    preempted: int
    rate: Metric
    findings_per_pr: Metric
    by_severity: dict[str, tuple[int, int]]


def score(results: dict, seed: int = 7) -> Preemption:
    rows = [(len(e["targets"]), len(matches(e)), len(e["findings"]))
            for e in results["prs"].values() if e["status"] in ("PASS", "FAIL")]
    ratios = lambda sample: (sum(r[1] for r in sample) / t if (t := sum(r[0] for r in sample)) else None,  # noqa: E731
                             sum(r[2] for r in sample) / len(sample) if sample else 0.0)
    point = ratios(rows)
    rng = random.Random(seed)
    draws = [ratios([rows[rng.randrange(len(rows))] for _ in rows]) for _ in range(BOOTSTRAP_ROUNDS)] if rows else []
    severity: Counter = Counter()
    hit: Counter = Counter()
    for entry in results["prs"].values():
        if entry["status"] not in ("PASS", "FAIL"):
            continue
        for target in entry["targets"]:
            severity[target["severity"]] += 1
        for target, _ in matches(entry):
            hit[target["severity"]] += 1
    return Preemption(len(rows), sum(r[0] for r in rows), sum(r[1] for r in rows),
                      _interval(point[0], [d[0] for d in draws]), _interval(point[1], [d[1] for d in draws]),
                      {s: (hit[s], n) for s, n in severity.most_common()})


def sample(results: dict, size: int, seed: int = 7) -> list[dict]:
    """Random matched pairs for judging whether the finding raises the same issue."""
    pairs = [{"pr": pr, "commit": e["commit"], "comment": t["id"], "path": t["path"], "lines": [t["start"], t["end"]],
              "finding": f["id"], "line": f["line"], "message": f["message"]}
             for pr, e in results["prs"].items() for t, f in matches(e)]
    random.Random(seed).shuffle(pairs)
    return pairs[:size]
