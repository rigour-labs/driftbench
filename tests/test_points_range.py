import base64

from bench.points.acted_on import acted_on
from bench.points.range_basis import MISSING, own_patch, parse_own_patch
from tests.github_fakes import FakeClient

ANCHOR = {"path": "a.py", "line": 11, "start_line": None, "side": "RIGHT", "commit_sha": "A"}
PR = {"head_sha": "H", "base_sha": "B", "commits": [{}]}
REBASED = {"status": "diverged", "ahead_by": 40, "files": []}
# The PR's own change at A: adds lines 10-12 to a.py.
OWN_A = "@@ -9,2 +9,5 @@\n x = 1\n+def f():\n+    return compute(a, b)\n+\n y = 2\n"


def compare_with(patch, path="a.py", **entry):
    return {"status": "diverged", "files": [{"filename": path, "patch": patch, **entry}]}


def contents(text):
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def run(objects):
    return acted_on(FakeClient({}, {}, {"repos/o/r/compare/A...H": REBASED, **objects}), "o/r", ANCHOR, PR)


def test_parse_own_patch_tracks_head_side_lines():
    own = parse_own_patch(OWN_A)
    assert own.added == [(10, "def f():"), (11, "    return compute(a, b)"), (12, "")]
    assert own.removed == []


def test_author_fixed_the_anchored_line_after_a_rebase():
    # at H (rebased onto newer upstream; its own patch shifted by 30 lines) the anchored line was rewritten
    own_h = "@@ -39,2 +39,5 @@\n x = 1\n+def f():\n+    return compute(a, b, timeout=5)\n+\n y = 2\n"
    verdict = run({"repos/o/r/compare/B...A": compare_with(OWN_A), "repos/o/r/compare/B...H": compare_with(own_h)})
    assert verdict == (True, "range")


def test_only_upstream_changed_nearby():
    # the PR's own lines are unchanged at H; upstream edits around them are not in either own patch
    own_h = "@@ -50,2 +50,5 @@\n x = 1\n+def f():\n+    return compute(a, b)\n+\n y = 2\n"
    verdict = run({"repos/o/r/compare/B...A": compare_with(OWN_A), "repos/o/r/compare/B...H": compare_with(own_h)})
    assert verdict == (False, "range")


def test_renamed_and_rewritten_file_is_unknown():
    gone = {"status": "diverged", "files": [{"filename": "b.py", "previous_filename": "a.py"}]}  # no patch
    verdict = run({"repos/o/r/compare/B...A": compare_with(OWN_A), "repos/o/r/compare/B...H": gone})
    assert verdict == (None, "rebased")


def test_pr_took_its_change_out():
    untouched = {"status": "diverged", "files": [{"filename": "other.py", "patch": "@@ -1 +1 @@\n-a\n+b\n"}]}
    verdict = run({"repos/o/r/compare/B...A": compare_with(OWN_A), "repos/o/r/compare/B...H": untouched})
    assert verdict == (True, "range")


def test_context_only_anchor_uses_fresh_changes_near_its_place_at_h():
    # anchor line 11 ("z") is context; the PR's own added line ("w", line 15) is outside the 8..14 window
    anchor_on_context = "@@ -9,7 +9,8 @@\n x\n y\n z\n a\n b\n c\n+w\n q\n"
    file_a = "".join(f"l{n}\n" for n in range(1, 9)) + "x\ny\nz\na\nb\nc\nw\nq\n"
    file_h = "up\n" * 20 + file_a                                    # rebased: 20 upstream lines above
    near = "@@ -29,7 +29,9 @@\n x\n y\n+guard()\n z\n a\n b\n c\n+w\n q\n"  # new PR line at 31
    far = "@@ -29,7 +29,8 @@\n x\n y\n z\n a\n b\n c\n+w\n q\n@@ -80 +81,2 @@\n end\n+extra()\n"
    files = {"repos/o/r/contents/a.py?ref=A": contents(file_a), "repos/o/r/contents/a.py?ref=H": contents(file_h)}
    common = {"repos/o/r/compare/B...A": compare_with(anchor_on_context), **files}
    assert run({**common, "repos/o/r/compare/B...H": compare_with(near)}) == (True, "range")
    assert run({**common, "repos/o/r/compare/B...H": compare_with(far)}) == (False, "range")


def test_author_and_upstream_both_changed_nearby_counts_only_the_pr_own_change():
    """Rule: true only if the PR's own patch differs near the anchor; upstream edits never count."""
    own_h_same = "@@ -39,2 +39,5 @@\n x = 1\n+def f():\n+    return compute(a, b)\n+\n y = 2\n"
    own_h_fixed = "@@ -39,2 +39,5 @@\n x = 1\n+def f():\n+    return compute(a, b) or 0\n+\n y = 2\n"
    base = {"repos/o/r/compare/B...A": compare_with(OWN_A)}
    assert run({**base, "repos/o/r/compare/B...H": compare_with(own_h_same)}) == (False, "range")
    assert run({**base, "repos/o/r/compare/B...H": compare_with(own_h_fixed)}) == (True, "range")


def test_squash_merged_head_without_a_readable_fork_is_unknown_never_false():
    assert run({"repos/o/r/compare/B...A": compare_with(OWN_A)}) == (None, "rebased")   # B...H is a 404


def test_force_pushed_round_fetchable_works_and_unfetchable_is_unknown():
    own_h = "@@ -39,2 +39,5 @@\n x = 1\n+def f():\n+    return compute(a, b, retries=3)\n+\n y = 2\n"
    assert run({"repos/o/r/compare/B...A": compare_with(OWN_A),
                "repos/o/r/compare/B...H": compare_with(own_h)}) == (True, "range")
    gone = acted_on(FakeClient({}, {}, {}), "o/r", ANCHOR, PR)       # anchor commit no longer served
    assert gone[0] is None


def test_own_patch_states():
    client = FakeClient({}, {}, {"repos/o/r/compare/B...X": {"files": [{"filename": "z.py", "patch": ""}]},
                                 "repos/o/r/compare/B...T": {"files": [{"filename": f"f{n}"} for n in range(300)]}})
    assert own_patch(client, "o/r", "B", "X", "a.py") is MISSING
    assert own_patch(client, "o/r", "B", "T", "a.py") is None      # truncated list: can't tell
    assert own_patch(client, "o/r", "B", "GONE", "a.py") is None   # compare 404
