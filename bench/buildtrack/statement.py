"""A task's statement as first written (docs/BUILD_TRACK.md, "Tasks").

The issue the pull request formally closes, when there is one and it was
opened before the pull request; else the pull request's description. Either
is taken as first written, from its edit history: a later edit can describe
what review asked for. GitHub lists edits newest first, and the oldest entry
holds the original text.
"""
from __future__ import annotations

import hashlib
import re

from bench.collect.github import GitHubClient

MIN_CHARS = 80
DIFF_LINE_RE = re.compile(r"^[+-](?![+-])\s*\S", re.MULTILINE)
MAX_DIFF_LINES = 3
EDITS = "userContentEdits(first: 100) { nodes { editedAt diff } }"


def query(repo: str, number: int) -> str:
    owner, name = repo.split("/")
    return (f'query {{ repository(owner: "{owner}", name: "{name}") {{ pullRequest(number: {number}) {{ '
            f"createdAt body {EDITS} closingIssuesReferences(first: 5) {{ nodes {{ number createdAt title body "
            f"{EDITS} }} }} }} }} }}")


def first_written(body: str | None, edits: list[dict]) -> str:
    """The text as created: the oldest edit's content when it was ever edited, else the body."""
    versions = [e for e in edits if isinstance(e.get("diff"), str) and e.get("editedAt")]
    return min(versions, key=lambda e: e["editedAt"])["diff"] if versions else (body or "")


def exclusion(text: str) -> str | None:
    """Why a statement can't be a task: too short to state one, or it carries the change itself."""
    if len(text.strip()) < MIN_CHARS:
        return f"statement shorter than {MIN_CHARS} characters"
    if "```diff" in text or len(DIFF_LINE_RE.findall(text)) >= MAX_DIFF_LINES:
        return "statement quotes a diff"
    return None


def statement(client: GitHubClient, repo: str, number: int) -> dict:
    """{source, text, sha256, chars} or {source, excluded}; the text is never written anywhere published."""
    pr = client.graphql(query(repo, number))["repository"]["pullRequest"]
    issues = [i for i in pr["closingIssuesReferences"]["nodes"] if i["createdAt"] < pr["createdAt"]]
    if issues:
        issue = min(issues, key=lambda i: i["number"])
        source = f"issue #{issue['number']}"
        text = f"{issue['title']}\n\n{first_written(issue['body'], issue['userContentEdits']['nodes'])}"
    else:
        source = "pull request description"
        text = first_written(pr["body"], pr["userContentEdits"]["nodes"])
    reason = exclusion(text)
    if reason:
        return {"source": source, "excluded": reason}
    return {"source": source, "text": text, "chars": len(text), "sha256": hashlib.sha256(text.encode()).hexdigest()}
