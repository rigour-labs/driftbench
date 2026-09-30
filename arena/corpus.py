"""The arena corpus: PRs a reviewer bot actually reviewed, pinned to commits.

Only facts are stored (PR numbers, commits, comment locations, category and
severity labels, links), not comment text, so the published corpus carries
no third-party prose. A PR counts only when the bot submitted a review on
it: a "review skipped" notice is not a review.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from arena import github

BOT = "coderabbitai[bot]"
SEARCH_PAGE = 100
SEARCH_CAP = 1000
_HEADER = re.compile(r"_([^_]+)_")


@dataclass
class BotComment:
    id: int
    url: str
    path: str
    #: Range on the commit the comment was made on (start == end for one line).
    start: int
    end: int
    commit: str
    category: str = ""
    severity: str = ""


@dataclass
class Pr:
    number: int
    url: str
    merge_sha: str
    head_sha: str
    merged_at: str
    reviewed_commits: list[str] = field(default_factory=list)
    comments: list[BotComment] = field(default_factory=list)


@dataclass
class Corpus:
    repo: str
    snapshot_sha: str
    mined_at: str
    min_age_days: int
    prs: list[Pr] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1) + "\n")

    @staticmethod
    def load(path: Path) -> "Corpus":
        raw = json.loads(path.read_text())
        prs = [Pr(**{**p, "comments": [BotComment(**c) for c in p["comments"]]}) for p in raw.pop("prs")]
        return Corpus(**raw, prs=prs)


def mine(repo: str, limit: int, min_age_days: int) -> Corpus:
    """Newest `limit` merged PRs the bot reviewed, merged at least `min_age_days` before now."""
    info = github.api(f"repos/{repo}")
    snapshot = github.api(f"repos/{repo}/commits/{info['default_branch']}")["sha"]
    cutoff = datetime.now(timezone.utc) - timedelta(days=min_age_days)
    corpus = Corpus(repo, snapshot, datetime.now(timezone.utc).isoformat(timespec="seconds"), min_age_days)
    for number in _reviewed_prs(repo, cutoff):
        pr = _pr(repo, number, info["default_branch"])
        if pr:
            corpus.prs.append(pr)
        if len(corpus.prs) >= limit:
            break
    return corpus


def _reviewed_prs(repo: str, cutoff: datetime):
    query = f"repo:{repo} reviewed-by:{BOT} is:pr is:merged merged:<{cutoff.date().isoformat()}"
    for page in range(1, SEARCH_CAP // SEARCH_PAGE + 1):
        result = github.api("search/issues", params={"q": query, "per_page": str(SEARCH_PAGE), "page": str(page), "sort": "created", "order": "desc"})
        items = result.get("items", [])
        yield from (item["number"] for item in items)
        if len(items) < SEARCH_PAGE:
            return


def _pr(repo: str, number: int, default_branch: str) -> Pr | None:
    data = github.api(f"repos/{repo}/pulls/{number}")
    if not data.get("merged_at") or data["base"]["ref"] != default_branch:
        return None
    reviews = [r for r in github.api(f"repos/{repo}/pulls/{number}/reviews", paginate=True) if r["user"]["login"] == BOT]
    if not reviews:
        return None
    comments = [c for c in github.api(f"repos/{repo}/pulls/{number}/comments", paginate=True) if _is_line_comment(c)]
    return Pr(
        number, data["html_url"], data["merge_commit_sha"], data["head"]["sha"], data["merged_at"],
        sorted({r["commit_id"] for r in reviews if r.get("commit_id")}),
        [_comment(c) for c in comments],
    )


def _is_line_comment(c: dict) -> bool:
    return (c["user"]["login"] == BOT and c.get("in_reply_to_id") is None
            and c.get("side", "RIGHT") == "RIGHT" and c.get("original_line") is not None)


def _comment(c: dict) -> BotComment:
    end = c["original_line"]
    start = c.get("original_start_line") or end
    labels = _HEADER.findall(c["body"].split("\n", 1)[0])
    return BotComment(
        c["id"], c["html_url"], c["path"], min(start, end), max(start, end), c["original_commit_id"],
        category=labels[0].strip() if labels else "", severity=labels[1].strip() if len(labels) > 1 else "",
    )
