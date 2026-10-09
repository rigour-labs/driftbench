"""Review and comment text by ID, checked against the frozen corpus.

The corpus holds only a SHA-256 and length per text. Text is read from the
same API lists the collector used (so a warm cache answers without a call),
and a text whose hash no longer matches is reported as changed (rule TEXT-1),
never silently used.
"""
from __future__ import annotations

import hashlib

from bench.collect.github import GitHubClient

KIND_PATHS = {
    "review": "repos/{repo}/pulls/{pr}/reviews",
    "inline": "repos/{repo}/pulls/{pr}/comments",
    "conversation": "repos/{repo}/issues/{pr}/comments",
}


class TextSource:
    def __init__(self, client: GitHubClient, repo: str):
        self.client = client
        self.repo = repo
        self.loaded: dict[tuple[int, str], dict[int, str]] = {}

    def verified_text(self, pr: int, kind: str, entry: dict) -> str | None:
        """The entry's text if it still matches its frozen hash, else None (TEXT-1)."""
        body = self.bodies_by_id(pr, kind).get(entry["id"])
        if body is None or hashlib.sha256(body.encode("utf-8")).hexdigest() != entry["body_sha256"]:
            return None
        return body

    def bodies_by_id(self, pr: int, kind: str) -> dict[int, str]:
        key = (pr, kind)
        if key not in self.loaded:
            items = self.client.get_all(KIND_PATHS[kind].format(repo=self.repo, pr=pr))
            self.loaded[key] = {item["id"]: item.get("body") or "" for item in items}
        return self.loaded[key]


POINT_KINDS = {"inline": "inline", "body": "review", "conversation": "conversation"}


def point_text(texts: TextSource, point: dict) -> str | None:
    """A point's own text (its span of the source), or None if the source changed (TEXT-1)."""
    source = {"id": point["source_id"], "body_sha256": point["body_sha256"]}
    text = texts.verified_text(point["pr"], POINT_KINDS[point["kind"]], source)
    if text is None or not point.get("span"):
        return text
    start, end = point["span"]
    return text[start:end]
