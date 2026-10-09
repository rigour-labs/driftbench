import base64

from bench.points.file_at import changed_lines_between, file_at
from tests.github_fakes import FakeClient


def test_changed_lines_between_matches_the_patch_convention():
    old = "a\nb\nc\nd\n"
    assert changed_lines_between(old, "a\nB\nc\nd\n") == {2}
    assert changed_lines_between(old, "a\nc\nd\n") == {2}
    assert changed_lines_between(old, "a\nb\nnew\nc\nd\n") == {2}
    assert changed_lines_between(old, old) == set()


def test_file_at_found_missing_and_unreadable():
    good = {"encoding": "base64", "content": base64.b64encode(b"x = 1\n").decode()}
    client = FakeClient({}, {}, {"repos/o/r/contents/dir/my%20file.py?ref=S": good,
                                 "repos/o/r/contents/big.bin?ref=S": {"encoding": "none", "content": ""}})
    assert file_at(client, "o/r", "dir/my file.py", "S").text == "x = 1\n"
    assert not file_at(client, "o/r", "gone.py", "S").found
    big = file_at(client, "o/r", "big.bin", "S")
    assert big.found and big.text is None
