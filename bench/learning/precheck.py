"""The pre-check: how many heads a learned review would actually change (docs/LEARNING.md).

A head served no lesson gets the cold reviewer's exact input, so only heads
with at least one lesson served can differ. Numbers and lesson ids only:
lesson text is review text and stays out of anything published.
"""
from __future__ import annotations

import json
from pathlib import Path

from bench.learning.leaks import check_store

MODES = ("verified", "all")


class PrecheckError(ValueError):
    pass


def read_store(path: Path) -> list[dict]:
    """A store's lessons; none when the learner wrote no store."""
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("lessons") or []
    except (OSError, ValueError, AttributeError) as exc:
        raise PrecheckError(f"unreadable store {path}: {exc}") from exc


def store_counts(lessons: list[dict]) -> dict:
    states = [lesson.get("state") for lesson in lessons]
    return {"lessons": len(lessons), "verified": states.count("verified"), "candidate": states.count("candidate")}


def check_pr(pr: dict, out_dir: Path, crawl: dict) -> dict:
    """One pull request: its store's counts and leak check, and each head's served lesson ids."""
    lessons = read_store(out_dir / pr["store"])
    check = check_store(lessons, pr["pr"], pr["cutoff"], crawl)
    heads = {}
    for sha, served in pr["heads"].items():
        if "error" in served:
            heads[sha] = {"error": served["error"]}
        elif check["leaks"]:
            heads[sha] = {"error": "the store leaks; never served"}
        else:
            heads[sha] = {mode: served[mode] for mode in MODES}
    return {"pr": pr["pr"], "cutoff": pr["cutoff"], **store_counts(lessons), "learned": pr.get("learned"),
            **check, "heads": heads}


def totals(prs: list[dict]) -> dict:
    heads = [h for pr in prs for h in pr["heads"].values()]
    ok = [h for h in heads if "error" not in h]
    return {"prs": len(prs), "heads": len(heads), "errors": len(heads) - len(ok),
            "leaking_prs": sum(1 for pr in prs if pr["leaks"]),
            "edited_after_lessons": sum(len(pr["edited_after"]) for pr in prs),
            **{f"heads_served_{mode}": sum(1 for h in ok if h[mode]) for mode in MODES},
            **{f"lessons_served_{mode}": sum(len(h[mode]) for h in ok) for mode in MODES}}


def precheck(served: dict, out_dir: Path, crawl: dict) -> dict:
    prs = [check_pr(pr, out_dir, crawl) for pr in served["prs"]]
    return {"repo": served["repo"], "core": served["core"], "limits": served["limits"],
            "crawl": {"prs": len(crawl["prs"]), "limit": crawl["limit"]}, "totals": totals(prs), "prs": prs}
