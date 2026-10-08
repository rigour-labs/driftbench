from bench.collect.timeline import head_at, head_history
from tests.github_fakes import make_commit, make_push


def test_history_from_pushes_and_commits_oldest_first():
    timeline = [
        make_push("B", "2026-02-03T00:00:00Z"),
        {"event": "committed", "sha": "A", "committer": {"date": "2026-02-01T00:00:00Z"}},
        {"event": "reviewed", "commit_id": "A", "submitted_at": "2026-02-02T00:00:00Z"},
    ]
    history = head_history(timeline, [make_commit("ignored", "2026-01-01T00:00:00Z")], [])
    assert [(h["sha"], h["source"]) for h in history] == [("A", "commit_date"), ("B", "push")]


def test_history_falls_back_to_the_commit_list():
    history = head_history([{"event": "labeled"}], [make_commit("A", "2026-02-01T00:00:00Z")], [])
    assert history == [{"sha": "A", "at": "2026-02-01T00:00:00Z", "source": "commit_date"}]


def test_head_at_review_then_push_then_comment():
    """Review on A, B pushed, a comment after B: the comment was written against B."""
    history = head_history([make_push("A", "2026-02-01T00:00:00Z"), make_push("B", "2026-02-03T00:00:00Z")], [], [])
    assert head_at(history, "2026-02-04T00:00:00Z")["sha"] == "B"
    assert head_at(history, "2026-02-02T00:00:00Z")["sha"] == "A"
    assert head_at(history, "2026-01-01T00:00:00Z") is None


def test_reviews_fill_heads_the_timeline_lost():
    """A rebased-away head appears only through the review made on it."""
    reviews = [{"commit_id": "OLD", "submitted_at": "2026-02-02T00:00:00Z"},
               {"commit_id": None, "submitted_at": "2026-02-02T06:00:00Z"}]
    history = head_history([make_push("NEW", "2026-02-05T00:00:00Z")], [], reviews)
    assert [(h["sha"], h["source"]) for h in history] == [("OLD", "review"), ("NEW", "push")]
    assert head_at(history, "2026-02-03T00:00:00Z")["sha"] == "OLD"
