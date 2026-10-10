"""Which pull requests a run reviews when it can't afford the whole corpus (docs/SPEC.md, "Subsamples").

The unit is the whole pull request, never a single head: a point on a later
round is matched against earlier heads of the same pull request, and false
blocks are judged on its approved head. Per repository the rule is either
the full corpus, or whole pull requests drawn until about `target` heads:
grouped by their largest diff (changed lines, 0-50 / 50-200 / 200-600 /
600+), each group's share of the target in proportion to its share of the
repository's heads, pull requests within a group in a seeded random order.
A repository can also be left out of a round, with the reason, which the
page then states.

The selection is written once to a committed file, with its seed and the
source of the diff sizes, and never redrawn; a run records the file's hash,
and scoring and the report use only the selected pull requests.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import yaml

from bench.harness.cases import cases_for_pr, heads_to_run

BUCKETS = ((0, 50), (50, 200), (200, 600), (600, None))


class SubsampleError(ValueError):
    pass


def bucket_of(lines: int) -> int:
    return next(i for i, (low, high) in enumerate(BUCKETS) if lines >= low and (high is None or lines < high))


def pr_heads(pr: dict) -> list[str]:
    return list(heads_to_run(cases_for_pr(pr)))


def shares(counts: list[int], target: int) -> list[int]:
    """`target` split across groups in proportion to `counts`, by largest remainder; empty groups get nothing."""
    total = sum(counts)
    if not total:
        return [0] * len(counts)
    raw = [target * c / total for c in counts]
    whole = [int(r) for r in raw]
    by_remainder = sorted((i for i in range(len(raw)) if counts[i]), key=lambda i: raw[i] - whole[i], reverse=True)
    for i in by_remainder[:target - sum(whole)]:
        whole[i] += 1
    return whole


SKIPPED = "not run in this round"


def skipped(reason: str) -> dict:
    return {"rule": f"{SKIPPED}: {reason}", "prs": [], "heads": 0}


def draw_repo(corpus: dict, sizes: dict[str, int], target: int | None, rng: random.Random) -> dict:
    prs = sorted(corpus["prs"], key=lambda pr: pr["number"])
    heads = {pr["number"]: pr_heads(pr) for pr in prs}
    if target is None:
        return {"rule": "full corpus", "prs": [pr["number"] for pr in prs], "heads": sum(map(len, heads.values()))}
    missing = [h for hs in heads.values() for h in hs if h not in sizes]
    if missing:
        raise SubsampleError(f"{corpus['repo']}: no diff size for {len(missing)} head(s), e.g. {missing[0][:12]}")
    groups: list[list[int]] = [[] for _ in BUCKETS]
    for pr in prs:
        groups[bucket_of(max(sizes[h] for h in heads[pr["number"]]))].append(pr["number"])
    quotas = shares([sum(len(heads[n]) for n in group) for group in groups], target)
    chosen: list[int] = []
    for group, quota in zip(groups, quotas):
        taken = 0
        for number in rng.sample(group, len(group)):
            if taken >= quota:
                break
            chosen.append(number)
            taken += len(heads[number])
    chosen.sort()
    return {"rule": f"whole pull requests until about {target} heads, stratified by largest diff, seeded",
            "prs": chosen, "heads": sum(len(heads[n]) for n in chosen)}


def draw(corpora: list[dict], sizes: dict[tuple[str, str], int], targets: dict[str, int | None], seed: int,
         source: str, skips: dict[str, str] | None = None) -> dict:
    """`targets`: repo -> heads to aim for, or None for the full corpus; `skips`: repo -> why it isn't run.
    `sizes`: (repo, head) -> changed lines."""
    rng = random.Random(seed)
    repos = {}
    for corpus in sorted(corpora, key=lambda c: c["repo"]):
        if corpus["repo"] in (skips or {}):
            repos[corpus["repo"]] = skipped(skips[corpus["repo"]])
            continue
        if corpus["repo"] not in targets:
            raise SubsampleError(f"{corpus['repo']}: no selection rule given")
        own = {head: lines for (repo, head), lines in sizes.items() if repo == corpus["repo"]}
        repos[corpus["repo"]] = draw_repo(corpus, own, targets[corpus["repo"]], rng)
    return {"seed": seed, "buckets": [list(b) for b in BUCKETS], "sizes_from": source, "repos": repos}


def sizes_from_run(run_dir: Path) -> dict[tuple[str, str], int]:
    """Changed lines per head, from an earlier run's records (every entrant sees the same diff)."""
    sizes = {}
    for path in run_dir.glob("*/*/*/*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SubsampleError(f"unreadable record {path}: {exc}") from exc
        if isinstance(record.get("changed_lines"), int):
            sizes[(record["repo"], record["head_sha"])] = record["changed_lines"]
    return sizes


def write_selection(selection: dict, path: Path, replace: bool) -> None:
    if path.exists() and not replace:
        raise SubsampleError(f"{path} exists; a selection is never redrawn (pass --replace before any run uses it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(selection, sort_keys=False), encoding="utf-8")


def read_selection(path: Path) -> dict:
    try:
        selection = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SubsampleError(f"cannot read selection {path}: {exc}") from exc
    if not isinstance(selection.get("repos"), dict):
        raise SubsampleError(f"{path}: not a selection file")
    return selection


def recorded_selection(manifest: dict) -> dict | None:
    """The selection a run's record names, refused if its file changed since the run began."""
    record = manifest.get("subsample")
    if not record:
        return None
    path = Path(record["file"])
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SubsampleError(f"{path} is missing or differs from the selection the run recorded")
    return read_selection(path)


def restrict_points(points_file: dict, corpus: dict) -> dict:
    """Only the points of the pull requests a (restricted) corpus still holds."""
    numbers = {pr["number"] for pr in corpus["prs"]}
    return {**points_file, "points": [p for p in points_file["points"] if p["pr"] in numbers]}


def cap_shares(selection: dict | None, repos: list[str], max_usd: float) -> dict[str, float]:
    """Each repo job's share of the cap: in proportion to its selected heads, or even without a selection."""
    if selection is None:
        return {repo: round(max_usd / len(repos), 4) for repo in repos}
    heads = {repo: (selection["repos"].get(repo) or {}).get("heads", 0) for repo in repos}
    total = sum(heads.values())
    if not total:
        raise SubsampleError("the selection has no heads to run in these repos")
    return {repo: round(max_usd * n / total, 4) for repo, n in heads.items()}


def selection_record(path: Path, selection: dict) -> dict:
    """What run.json keeps: the file, its hash, the seed, and each repo's rule and counts."""
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "seed": selection["seed"],
            "repos": {repo: {"rule": r["rule"], "prs": len(r["prs"]), "heads": r["heads"]}
                      for repo, r in selection["repos"].items()}}


def restrict(corpus: dict, selection: dict | None) -> dict:
    """The corpus with only the selected pull requests (unchanged without a selection)."""
    if selection is None:
        return corpus
    chosen = set((selection["repos"].get(corpus["repo"]) or {}).get("prs", []))
    return {**corpus, "prs": [pr for pr in corpus["prs"] if pr["number"] in chosen]}
