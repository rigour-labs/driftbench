from bench.score.match import closest_match, distance, eligible_heads, location_scorable
from tests.score_fixtures import SameFileVersions, finding, inline_point

ROUNDS = [{"index": 1, "head_sha": "h1"}, {"index": 2, "head_sha": "h2"}, {"index": 3, "head_sha": "h1"}]


def test_location_scorable_needs_inline_new_side_line_and_a_round():
    point = inline_point("p", 1, 1, 10, True)
    assert location_scorable(point)
    assert not location_scorable({**point, "scorable": False})
    assert not location_scorable({**point, "kind": "body"})
    assert not location_scorable({**point, "anchor": {**point["anchor"], "side": "LEFT"}})
    assert not location_scorable({**point, "anchor": {**point["anchor"], "line": None}})


def test_distance_to_a_line_or_a_range():
    assert distance(10, {"line": 10}) == 0 and distance(13, {"line": 10}) == 3
    assert distance(6, {"line": 10, "start_line": 7}) == 1 and distance(8, {"line": 10, "start_line": 7}) == 0


def test_distance_from_a_ranged_finding():
    assert distance(1, {"line": 30}, end_line=60) == 0      # 1..60 contains line 30
    assert distance(40, {"line": 30}, end_line=60) == 10
    assert distance(1, {"line": 30}, end_line=25) == 5
    assert distance(1, {"line": 30, "start_line": 26}, end_line=27) == 0


def test_eligible_heads_are_rounds_up_to_the_points_own():
    assert eligible_heads(inline_point("p", 1, 1, 10, True), ROUNDS) == ["h1"]
    assert eligible_heads(inline_point("p", 1, 3, 10, True), ROUNDS) == ["h1", "h2"]


def test_closest_match_ignores_later_heads_other_files_errors_and_lineless():
    point = inline_point("p", 1, 1, 10, True)
    records = {
        "h1": {"verdict": "pass", "findings": [finding(30), finding(12), finding(10, path="other.py"), finding(None)]},
        "h2": {"verdict": "pass", "findings": [finding(10)]},           # round 2: after the point; never used
    }
    match = closest_match(point, ["h1"], records, SameFileVersions())
    assert (match.head_sha, match.finding_index, match.distance) == ("h1", 1, 2)
    assert closest_match(point, ["h1"], {"h1": {"verdict": "error", "findings": [finding(10)]}},
                         SameFileVersions()) is None


def test_every_hunk_range_catches_a_point_inside_a_long_hunk():
    point = inline_point("p", 1, 1, 30, True)
    records = {"h1": {"verdict": "pass", "findings": [{**finding(5), "end_line": 64}]}}
    assert closest_match(point, ["h1"], records, SameFileVersions()).distance == 0


def test_blocking_only_skips_advisory_findings():
    point = inline_point("p", 1, 1, 10, True)
    records = {"h1": {"verdict": "fail", "findings": [finding(10), finding(15, blocking=True)]}}
    assert closest_match(point, ["h1"], records, SameFileVersions()).distance == 0
    assert closest_match(point, ["h1"], records, SameFileVersions(), blocking_only=True).distance == 5
