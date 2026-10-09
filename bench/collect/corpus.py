"""Collect one repository's frozen corpus: the selected PRs and their review rounds."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bench.collect.commits import CommitDates, known_dates, resolve_review
from bench.collect.github import GitHubClient
from bench.collect.record import PrFetch, pr_record
from bench.collect.rounds import build_rounds
from bench.collect.select import (candidate_prs, human_reviews, parse_time, substantive_conversation,
                                  substantive_ids)
from bench.repos import PinnedRepo, slug_of

CORPUS_SCHEMA = 2


def collect_repo(client: GitHubClient, repo: PinnedRepo, max_prs: int, max_listed: int) -> dict:
    """Walk candidates in order and keep the first `max_prs` substantively reviewed ones."""
    candidates = candidate_prs(client, repo.name, repo.pinned_at, max_listed)
    records = []
    for pr in candidates:
        if len(records) >= max_prs:
            break
        record = collect_pr(client, repo.name, pr)
        if record:
            records.append(record)
    return {
        "schema": CORPUS_SCHEMA,
        "repo": repo.name,
        "pin": repo.pin,
        "pinned_at": repo.pinned_at.isoformat(),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection": {"max_prs": max_prs, "max_listed": max_listed, "merged_candidates": len(candidates)},
        "prs": records,
    }


def collect_pr(client: GitHubClient, repo: str, pr: dict) -> dict | None:
    """The PR's record, or None if no human reviewed it substantively (review or conversation)."""
    base = f"repos/{repo}/pulls/{pr['number']}"
    author = (pr.get("user") or {}).get("login", "")
    merged_at = parse_time(pr["merged_at"])
    comments = client.get_all(f"{base}/comments")
    reviews = human_reviews(client.get_all(f"{base}/reviews"), author, merged_at)
    conversation = client.get_all(f"repos/{repo}/issues/{pr['number']}/comments")
    substantive = substantive_ids(reviews, comments)
    if not substantive and not substantive_conversation(conversation, author, merged_at):
        return None
    commits = client.get_all(f"{base}/commits")
    timeline = client.get_all(f"repos/{repo}/issues/{pr['number']}/timeline")
    dates = CommitDates(client, repo, known_dates(commits, timeline))
    resolved = [resolve_review(review, comments, dates) for review in reviews]
    fetch = PrFetch(pr, resolved, comments, conversation, commits, timeline)
    rounds = build_rounds([r for r in resolved if r["id"] in substantive and r["commit_id"]])
    return pr_record(fetch, substantive, rounds)


class CorpusError(ValueError):
    pass


def read_corpus(path: Path) -> dict:
    """A frozen corpus file, checked for the schema this code understands."""
    try:
        corpus = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"cannot read corpus {path}: {exc}") from exc
    if not isinstance(corpus, dict) or corpus.get("schema") != CORPUS_SCHEMA:
        raise CorpusError(f"{path}: expected corpus schema {CORPUS_SCHEMA}")
    return corpus


def write_corpus(corpus: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug_of(corpus['repo'])}.json"
    path.write_text(json.dumps(corpus, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path
