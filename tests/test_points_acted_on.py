import base64

from bench.points.acted_on import acted_on, changed_old_lines, from_patch
from tests.github_fakes import FakeClient

PATCH = "@@ -10,4 +10,4 @@ func f() {\n ctx := a\n-old := b\n+new := b\n keep\n keep\n@@ -40,2 +40,3 @@\n x\n+inserted\n y\n"
ANCHOR = {"path": "a.go", "line": 14, "start_line": None, "side": "RIGHT", "commit_sha": "A"}
OLD = "".join(f"line {n}\n" for n in range(1, 31))


def compare(status: str = "ahead", ahead_by: int = 1, files: list | None = None) -> dict:
    if files is None:
        files = [{"filename": "a.go", "status": "modified", "patch": PATCH}]
    return {"status": status, "ahead_by": ahead_by, "files": files}


def contents(text: str) -> dict:
    return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}


def test_changed_old_lines_marks_removals_and_insertions():
    assert changed_old_lines(PATCH) == {11, 40}


def test_patch_change_within_three_lines_counts():
    assert from_patch(compare(), ANCHOR) is True                      # 11 is within 14 - 3
    assert from_patch(compare(), {**ANCHOR, "line": 20}) is False      # nothing near line 20
    assert from_patch(compare(), {**ANCHOR, "line": 44, "start_line": 43}) is True


def test_patch_file_untouched_removed_renamed_or_without_patch():
    assert from_patch(compare(files=[{"filename": "b.go", "patch": PATCH}]), ANCHOR) is False
    assert from_patch(compare(files=[{"filename": "a.go", "status": "removed"}]), ANCHOR) is True
    assert from_patch(compare(files=[{"filename": "c.go", "previous_filename": "a.go", "patch": PATCH}]), ANCHOR) is True
    assert from_patch(compare(files=[{"filename": "a.go", "status": "modified"}]), ANCHOR) is None
    truncated = [{"filename": f"f{n}.go", "patch": PATCH} for n in range(300)]
    assert from_patch(compare(files=truncated), ANCHOR) is None


def test_ancestor_falls_back_to_contents_when_the_patch_cant_tell():
    new = OLD.replace("line 13\n", "line 13 changed\n")
    file_contents = {"repos/o/r/contents/a.go?ref=A": contents(OLD), "repos/o/r/contents/a.go?ref=H": contents(new)}
    truncated = compare(files=[{"filename": f"f{n}.go", "patch": PATCH} for n in range(300)])
    verdict, client = run({"repos/o/r/compare/A...H": truncated, **file_contents})
    assert verdict == (True, "ancestor") and "repos/o/r/contents/a.go?ref=H" in client.calls
    no_patch = compare(files=[{"filename": "a.go", "status": "modified"}])
    assert run({"repos/o/r/compare/A...H": no_patch, **file_contents})[0] == (True, "ancestor")


def run(objects: dict, anchor: dict = ANCHOR, pr_commits: int = 1):
    client = FakeClient({}, {}, objects)
    pr = {"head_sha": "H", "base_sha": "B", "commits": [{}] * pr_commits}
    return acted_on(client, "o/r", anchor, pr), client


def test_ancestor_uses_the_compare_patch():
    assert run({"repos/o/r/compare/A...H": compare()})[0] == (True, "ancestor")


def test_amended_in_place_diffs_the_file_directly():
    new = OLD.replace("line 15\n", "line 15 changed\n")
    objects = {"repos/o/r/compare/A...H": compare(status="diverged"),
               "repos/o/r/contents/a.go?ref=A": contents(OLD), "repos/o/r/contents/a.go?ref=H": contents(new)}
    assert run(objects)[0] == (True, "direct")
    assert run(objects, {**ANCHOR, "line": 25})[0] == (False, "direct")


def test_direct_file_removed_or_unreadable():
    removed = {"repos/o/r/compare/A...H": compare(status="diverged"), "repos/o/r/contents/a.go?ref=A": contents(OLD)}
    assert run(removed)[0] == (True, "direct")
    too_big = {**removed, "repos/o/r/contents/a.go?ref=H": {"encoding": "none", "content": ""}}
    assert run(too_big)[0] == (None, "direct")


def test_rebased_without_readable_own_patches_is_unknown():
    verdict, client = run({"repos/o/r/compare/A...H": compare(status="diverged", ahead_by=40)}, pr_commits=2)
    assert verdict == (None, "rebased")
    assert "repos/o/r/compare/B...A" in client.calls       # the range basis was tried first


def test_skips_left_side_lineless_missing_commit_and_final_head():
    assert run({}, {**ANCHOR, "side": "LEFT"})[0] == (None, None)
    assert run({}, {**ANCHOR, "line": None})[0] == (None, None)
    assert run({})[0] == (None, None)                      # compare 404
    verdict, client = run({}, {**ANCHOR, "commit_sha": "H"})
    assert verdict == (False, "ancestor") and client.calls == []
