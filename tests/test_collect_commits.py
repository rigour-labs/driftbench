from bench.collect.commits import CommitDates, known_dates, resolve_review
from tests.github_fakes import FakeClient, commit_objects, make_comment, make_commit, make_review

DATES = {"A": "2026-02-01T00:00:00Z", "LATER": "2026-02-09T00:00:00Z"}


def dates_with(known: dict[str, str]) -> tuple[CommitDates, FakeClient]:
    client = FakeClient({}, {}, commit_objects("o/r", DATES))
    return CommitDates(client, "o/r", known), client


def test_reported_commit_that_predates_the_review_is_trusted():
    dates, client = dates_with({})
    review = resolve_review(make_review(1, "APPROVED", "A", "2026-02-02T00:00:00Z"), [], dates)
    assert (review["commit_id"], review["commit_check"]) == ("A", "ok")
    assert client.calls == ["repos/o/r/commits/A"]


def test_commit_dated_after_the_review_falls_back_to_its_inline_comments():
    """GitHub reported a later head for this approval; its comment was written on A."""
    dates, _ = dates_with({})
    review = make_review(1, "APPROVED", "LATER", "2026-02-02T00:00:00Z")
    comments = [make_comment(9, 1, "2026-02-01T23:00:00Z", original_commit_id="A"),
                make_comment(8, 2, "2026-02-01T23:00:00Z", original_commit_id="LATER")]
    resolved = resolve_review(review, comments, dates)
    assert (resolved["commit_id"], resolved["commit_check"], resolved["reported_commit_id"]) == ("A", "from_comment", "LATER")


def test_untrusted_when_nothing_predates_the_review():
    dates, _ = dates_with({})
    for sha in ("LATER", "GONE"):
        resolved = resolve_review(make_review(1, "APPROVED", sha, "2026-02-02T00:00:00Z"), [], dates)
        assert (resolved["commit_id"], resolved["commit_check"]) == (None, "untrusted")


def test_known_dates_avoid_lookups_and_are_fetched_once():
    timeline = [{"event": "committed", "sha": "B", "committer": {"date": "2026-01-05T00:00:00Z"}}]
    dates, client = dates_with(known_dates([make_commit("C", "2026-01-06T00:00:00Z")], timeline))
    assert dates.date("B") and dates.date("C") and client.calls == []
    dates.date("A"), dates.date("A")
    assert client.calls == ["repos/o/r/commits/A"]
