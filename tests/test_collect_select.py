from datetime import datetime, timezone

from bench.collect.rounds import build_rounds
from bench.collect.select import candidate_prs, substantive_reviews
from tests.github_fakes import AUTHOR, BOT, FakeClient, make_comment, make_pr, make_review

PIN = datetime(2026, 6, 1, tzinfo=timezone.utc)
MERGED = datetime(2026, 3, 1, tzinfo=timezone.utc)
LIST = "repos/o/r/pulls"


def test_candidates_keep_merged_before_pin_sorted_by_updated_then_merged():
    page = [
        make_pr(1, "2026-02-01T00:00:00Z", "2026-05-01T00:00:00Z"),
        make_pr(2, None, "2026-05-09T00:00:00Z"),                    # closed, not merged
        make_pr(3, "2026-07-01T00:00:00Z", "2026-07-02T00:00:00Z"),  # merged after the pin
        make_pr(4, "2026-04-01T00:00:00Z", "2026-05-01T00:00:00Z"),  # same updated as 1, merged later
        make_pr(5, "2026-01-01T00:00:00Z", "2026-05-20T00:00:00Z"),
    ]
    prs = candidate_prs(FakeClient({LIST: [page]}, {}), "o/r", PIN, limit=50)
    assert [pr["number"] for pr in prs] == [5, 4, 1]


def test_candidates_stop_at_the_listing_limit():
    pages = [[make_pr(n, "2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z") for n in range(k * 100, k * 100 + 100)]
             for k in range(3)]
    client = FakeClient({LIST: pages}, {})
    assert len(candidate_prs(client, "o/r", PIN, limit=150)) == 150
    assert client.calls == [LIST, LIST]


def test_substantive_reviews_rules():
    reviews = [
        make_review(1, "APPROVED", "h1", "2026-02-01T00:00:00Z"),                         # bare approval
        make_review(2, "APPROVED", "h1", "2026-02-01T00:00:00Z", body="  "),              # whitespace body
        make_review(3, "CHANGES_REQUESTED", "h1", "2026-02-01T00:00:00Z"),                # kept
        make_review(4, "COMMENTED", "h1", "2026-02-02T00:00:00Z", body="needs a test"),   # kept
        make_review(5, "APPROVED", "h2", "2026-02-03T00:00:00Z"),                         # kept: inline
        make_review(6, "COMMENTED", "h2", "2026-02-03T00:00:00Z", body="x", user=AUTHOR),  # author
        make_review(7, "COMMENTED", "h2", "2026-02-03T00:00:00Z", body="x", user=BOT),     # bot
        make_review(8, "COMMENTED", "h2", "2026-04-01T00:00:00Z", body="late"),           # after merge
        make_review(9, "COMMENTED", None, "2026-02-03T00:00:00Z", body="no commit"),
        make_review(10, "APPROVED", "h2", "2026-02-03T00:00:00Z", body="Thank you!"),    # ACK-1
        make_review(11, "APPROVED", "h2", "2026-02-03T00:00:00Z"),                       # inline ack only
    ]
    comments = [make_comment(50, 5, "2026-02-03T00:00:00Z"),
                make_comment(51, 11, "2026-02-03T00:00:00Z", body="LGTM")]
    kept = substantive_reviews(reviews, comments, "author", MERGED)
    assert [r["id"] for r in kept] == [3, 4, 5]


def test_rounds_group_by_head_in_first_review_order():
    reviews = [
        make_review(3, "COMMENTED", "h2", "2026-02-05T00:00:00Z", body="b"),
        make_review(1, "CHANGES_REQUESTED", "h1", "2026-02-01T00:00:00Z"),
        make_review(2, "COMMENTED", "h1", "2026-02-02T00:00:00Z", body="a"),
        make_review(4, "COMMENTED", "h1", "2026-02-06T00:00:00Z", body="back on h1"),
    ]
    rounds = build_rounds(reviews)
    assert [(r.index, r.head_sha, r.review_ids) for r in rounds] == [(1, "h1", (1, 2, 4)), (2, "h2", (3,))]
    assert rounds[0].first_review_at == "2026-02-01T00:00:00Z"
