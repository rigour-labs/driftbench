import hashlib

import pytest

from bench.report.markdown import selection_line
from bench.subsample import (SubsampleError, draw, read_selection, recorded_selection, restrict, restrict_points,
                             selection_record, shares, write_selection)


def pr(number: int, heads: int) -> dict:
    rounds = [{"index": i + 1, "head_sha": f"{number}h{i}"} for i in range(heads)]
    return {"number": number, "rounds": rounds, "head_sha": rounds[-1]["head_sha"],
            "approved_head_sha": rounds[-1]["head_sha"], "approval_overridden": False}


def corpus(repo: str, n: int) -> dict:
    return {"repo": repo, "prs": [pr(i, 1 + i % 2) for i in range(1, n + 1)]}


def sizes_for(c: dict) -> dict:
    """PR i's heads change 10 * i lines: small PRs first, a long tail of large ones."""
    return {(c["repo"], r["head_sha"]): 10 * p["number"] for p in c["prs"] for r in p["rounds"]}


def test_shares_split_a_target_in_proportion_by_largest_remainder():
    assert shares([111, 84, 96, 34], 40) == [14, 10, 12, 4] and sum(shares([1, 1, 1], 10)) == 10
    assert shares([0, 0], 5) == [0, 0]


def test_a_draw_is_seeded_stratified_whole_prs_and_full_where_asked():
    a, b = corpus("o/a", 60), corpus("o/b", 30)
    sizes = {**sizes_for(a), **sizes_for(b)}
    first = draw([a, b], sizes, {"o/a": 20, "o/b": None}, seed=7, source="test")
    assert first == draw([a, b], sizes, {"o/a": 20, "o/b": None}, seed=7, source="test")
    assert first != draw([a, b], sizes, {"o/a": 20, "o/b": None}, seed=8, source="test")
    sampled, full = first["repos"]["o/a"], first["repos"]["o/b"]
    assert full == {"rule": "full corpus", "prs": list(range(1, 31)), "heads": 45}
    assert 20 <= sampled["heads"] <= 24 and sampled["prs"] == sorted(sampled["prs"])
    groups = [sum(1 for n in sampled["prs"] if lo <= 10 * n < (hi or 10**9)) for lo, hi in first["buckets"]]
    assert all(groups[:3]) and groups[3] == 0    # the 600+ group is one 1-head PR: its share of 20 rounds to 0
    with pytest.raises(SubsampleError, match="no diff size"):
        draw([a], {}, {"o/a": 20}, seed=1, source="test")
    with pytest.raises(SubsampleError, match="no selection rule"):
        draw([a], sizes, {}, seed=1, source="test")


def test_a_selection_is_written_once_recorded_by_hash_and_restricts_corpus_and_points(tmp_path):
    a = corpus("o/a", 10)
    selection = draw([a], sizes_for(a), {"o/a": 5}, seed=1, source="test")
    path = tmp_path / "run-2.yaml"
    write_selection(selection, path, replace=False)
    with pytest.raises(SubsampleError, match="never redrawn"):
        write_selection(selection, path, replace=False)
    record = selection_record(path, read_selection(path))
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() and record["repos"]["o/a"]["prs"] >= 2
    assert recorded_selection({"subsample": record}) == read_selection(path)
    assert recorded_selection({}) is None
    chosen = restrict(a, selection)
    assert [p["number"] for p in chosen["prs"]] == selection["repos"]["o/a"]["prs"]
    points = {"points": [{"id": f"p{n}", "pr": n} for n in range(1, 11)]}
    assert {p["pr"] for p in restrict_points(points, chosen)["points"]} == set(selection["repos"]["o/a"]["prs"])
    path.write_text(path.read_text() + "\n# edited\n")
    with pytest.raises(SubsampleError, match="differs from the selection the run recorded"):
        recorded_selection({"subsample": record})


def test_the_page_states_each_repos_selection():
    assert selection_line({}) == ""
    assert selection_line({"subsample": {"rule": "full corpus", "prs": 60, "heads": 134}}) == "Selection: full corpus. "
    line = selection_line({"subsample": {"rule": "whole pull requests until about 40 heads, stratified by largest "
                                                  "diff, seeded", "prs": 23, "heads": 44}})
    assert "seeded subsample of 23 PRs and 44 heads" in line and "intervals are wider" in line


def test_a_skipped_repo_is_stated_and_gets_no_share_of_the_cap():
    from bench.report.markdown import repo_section
    from bench.subsample import cap_shares
    a, b = corpus("o/a", 20), corpus("o/b", 20)
    sizes = {**sizes_for(a), **sizes_for(b)}
    selection = draw([a, b], sizes, {"o/a": 10}, seed=3, source="t", skips={"o/b": "budget: reason"})
    assert selection["repos"]["o/b"] == {"rule": "not run in this round: budget: reason", "prs": [], "heads": 0}
    shares_now = cap_shares(selection, ["o/a"], 50.0)
    assert shares_now == {"o/a": 50.0}
    two = {"repos": {"o/a": {"heads": 39}, "o/b": {"heads": 33}}}
    assert cap_shares(two, ["o/a", "o/b"], 50.0) == {"o/a": 27.0833, "o/b": 22.9167}
    assert cap_shares(None, ["o/a", "o/b"], 50.0) == {"o/a": 25.0, "o/b": 25.0}
    with pytest.raises(SubsampleError, match="no heads"):
        cap_shares(selection, ["o/b"], 50.0)
    section = "\n".join(repo_section({"repo": "o/b", "pin": "p" * 40, "reportable": False,
                                      "subsample": {"rule": "not run in this round: budget: reason", "prs": 0,
                                                    "heads": 0}}, {}))
    assert "not run in this round: budget: reason" in section and "Insufficient data" not in section


def test_run_json_records_the_selection_and_the_cap_shares(tmp_path):
    from bench.__main__ import main
    from bench.harness.cli import read_manifest
    a, b = corpus("o/a", 20), corpus("o/b", 20)
    path = tmp_path / "sel.yaml"
    write_selection(draw([a, b], {**sizes_for(a), **sizes_for(b)}, {"o/a": 10, "o/b": 8}, seed=1, source="t"),
                    path, replace=False)
    out = tmp_path / "run"
    assert main(["manifest", "--entrants", "claude-code-review", "--out", str(out), "--labels", str(tmp_path),
                 "--model", "m", "--max-usd", "50", "--head-bound", "claude-code-review=1",
                 "--subsample", str(path)]) == 0
    manifest = read_manifest(out / "run.json")
    assert manifest["subsample"]["sha256"] and set(manifest["subsample"]["repos"]) == {"o/a", "o/b"}
    shares_now = manifest["paid"]["cap_shares"]
    assert round(sum(shares_now.values()), 2) == 50.0 and shares_now["o/a"] > shares_now["o/b"]


def test_an_explicit_selection_lists_exact_heads_and_their_prs():
    from bench.subsample import explicit, selected_heads
    a, b = corpus("o/a", 6), corpus("o/b", 4)
    wanted = {"o/a": {"2h0", "5h0", "5h1"}}                  # PR 2 has one head, PR 5 has two
    selection = explicit([a, b], wanted, "why the reviewer missed these")
    assert selection["repos"]["o/a"] == {"rule": "explicit heads: why the reviewer missed these", "prs": [2, 5],
                                         "heads": 3, "head_shas": ["2h0", "5h0", "5h1"]}
    assert selection["repos"]["o/b"]["rule"].startswith("not run in this round")
    assert selected_heads(selection) == frozenset({"2h0", "5h0", "5h1"})
    assert selected_heads({"repos": {"o/a": {"prs": [1], "heads": 1}}}) is None
    with pytest.raises(SubsampleError, match="not in the corpus"):
        explicit([a], {"o/a": {"nope"}}, "x")
    with pytest.raises(SubsampleError, match="repos not in the corpus"):
        explicit([a], {"o/z": {"1h0"}}, "x")
