"""Which merged pull requests become build-track tasks, and their statements as first written."""
from __future__ import annotations

import json

from bench.buildtrack.statement import exclusion, first_written, statement
from bench.buildtrack.tasks import candidate, is_test, select

PR = {"number": 7, "merged_at": "2026-01-09T00:00:00Z", "base_sha": "b" * 40, "head_sha": "h2" * 20,
      "created_at": "2026-01-02T00:00:00Z", "head_history": [{"sha": "h1" * 20}, {"sha": "h2" * 20}]}
FILES = [{"filename": "net/dns.go", "status": "modified"}, {"filename": "net/dns_test.go", "status": "modified"}]
LONG = "The resolver retries forever when the upstream answers SERVFAIL; it should give up after three tries."


def test_test_files_are_recognised_across_languages():
    assert all(map(is_test, ["a/b_test.go", "x/test_y.py", "src/a.test.ts", "web/__tests__/z.js", "tests/u.py"]))
    assert not any(map(is_test, ["a/b.go", "contest.py", "src/attest.ts", "latest/x.go"]))


def test_a_task_needs_a_test_another_file_and_an_acted_on_point():
    task = candidate(PR, FILES, ["7-inline-1-0"])
    assert task["first_head"] == "h1" * 20 and task["merged_head"] == "h2" * 20 and task["test_files"] == ["net/dns_test.go"]
    assert candidate(PR, FILES[:1], ["p"]) == "changed no test file"
    assert candidate(PR, FILES[1:], ["p"]) == "changed only tests"
    assert candidate(PR, FILES, []) == "no acted-on human point"
    assert candidate({**PR, "merged_at": None}, FILES, ["p"]) == "not merged"
    assert candidate(PR, FILES, ["p"], max_files=1) == "changed more than 1 files"


def test_the_statement_is_the_text_as_first_written():
    edits = [{"editedAt": "2026-01-03T00:00:00Z", "diff": "edited after review"},
             {"editedAt": "2026-01-02T00:00:00Z", "diff": "as first written"}]
    assert first_written("edited after review", edits) == "as first written"
    assert first_written("never edited", []) == "never edited"
    assert exclusion("too short") and exclusion(LONG + "\n```diff\n-a\n+b\n```") == "statement quotes a diff"
    assert exclusion(LONG + "\n-old line\n+new line\n+another\n") == "statement quotes a diff"
    assert exclusion(LONG + "\n- a bullet, not a diff\n") is None


class Client:
    def __init__(self, issues: list[dict], body: str = LONG):
        self.issues, self.body = issues, body

    def graphql(self, query):
        return {"repository": {"pullRequest": {"createdAt": "2026-01-02T00:00:00Z", "body": self.body,
                                               "userContentEdits": {"nodes": []},
                                               "closingIssuesReferences": {"nodes": self.issues}}}}

    def get_all(self, path):
        return FILES


def test_a_closing_issue_opened_first_is_the_statement_and_no_text_reaches_the_task_file():
    issue = {"number": 3, "createdAt": "2026-01-01T00:00:00Z", "title": "Retries never stop", "body": LONG,
             "userContentEdits": {"nodes": []}}
    later = {**issue, "number": 4, "createdAt": "2026-01-05T00:00:00Z"}
    assert statement(Client([later, issue]), "o/r", 7)["source"] == "issue #3"
    assert statement(Client([later]), "o/r", 7)["source"] == "pull request description"
    corpus = {"repo": "o/r", "prs": [PR, {**PR, "number": 8}]}
    points = {"points": [{"pr": 7, "id": "7-inline-1-0", "acted_on": True}]}
    drawn = select(Client([issue]), corpus, points, count=5, seed=1)
    assert drawn["count"] == 5 and drawn["eligible"] == 1
    assert [t["pr"] for t in drawn["tasks"]] == [7] and drawn["excluded"] == [{"pr": 8, "reason": "no acted-on human point"}]
    assert LONG not in json.dumps(drawn) and set(drawn["tasks"][0]["statement"]) == {"source", "chars", "sha256"}
