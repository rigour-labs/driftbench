from datetime import datetime, timezone

from bench.collect.rounds import build_rounds, round_for_head
from bench.collect.select import (approval_overridden, approved_head, candidate_prs, human_reviews, substantive_conversation,
                                  substantive_ids)
from tests.github_fakes import AUTHOR, BOT, FakeClient, make_comment, make_conversation, make_pr, make_review

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
    humans = human_reviews(reviews, "author", MERGED)
    assert [r["id"] for r in humans] == [1, 2, 3, 4, 5, 10, 11]
    assert substantive_ids(humans, comments) == {3, 4, 5}


def test_approved_head_is_the_last_approval():
    reviews = [
        make_review(1, "CHANGES_REQUESTED", "A", "2026-02-01T00:00:00Z"),
        make_review(2, "APPROVED", "B", "2026-02-03T00:00:00Z"),
        make_review(3, "APPROVED", "A", "2026-02-02T00:00:00Z"),
    ]
    assert approved_head(reviews) == "B"
    assert approved_head(reviews[:1]) is None


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


def test_round_for_head():
    rounds = build_rounds([make_review(1, "COMMENTED", "h1", "2026-02-01T00:00:00Z", body="a")])
    assert round_for_head(rounds, "h1") == 1
    assert round_for_head(rounds, "h2") is None and round_for_head(rounds, None) is None


def test_substantive_conversation_rules():
    conversation = [
        make_conversation(1, "2026-02-01T00:00:00Z", "this drops the error"),        # kept
        make_conversation(2, "2026-02-01T00:00:00Z", "LGTM"),                        # ACK-1
        make_conversation(3, "2026-02-01T00:00:00Z", "/retest"),                     # CMD-1
        make_conversation(4, "2026-02-01T00:00:00Z", "fixed it", user=AUTHOR),       # author
        make_conversation(5, "2026-02-01T00:00:00Z", "coverage dropped", user=BOT),  # bot
        make_conversation(6, "2026-04-01T00:00:00Z", "post-merge question"),         # after merge
    ]
    assert substantive_conversation(conversation, "author", MERGED) == {1}


def test_approval_overridden_by_a_later_change_request():
    approve = make_review(1, "APPROVED", "X", "2026-02-01T00:00:00Z")
    later_request = make_review(2, "CHANGES_REQUESTED", "X", "2026-02-02T00:00:00Z")
    earlier_request = make_review(3, "CHANGES_REQUESTED", "W", "2026-01-31T00:00:00Z")
    untrusted_request = {**later_request, "commit_id": None}
    assert approval_overridden([approve, later_request])
    assert not approval_overridden([earlier_request, approve])
    assert not approval_overridden([approve, untrusted_request])
    assert not approval_overridden([later_request])
