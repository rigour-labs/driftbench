import base64

from bench.score.linemap import FileVersions, map_line
from tests.github_fakes import FakeClient

OLD = "a\nb\nc\nd\ne\n"


def test_map_line_through_edits():
    assert map_line(OLD, OLD, 3) == 3
    assert map_line(OLD, "new\nnew\na\nb\nc\nd\ne\n", 3) == 5   # shifted by two inserted lines
    assert map_line(OLD, "a\nB\nC\nd\ne\n", 3) == 2             # inside a changed region: its start
    assert map_line(OLD, "a\nd\ne\n", 3) == 2                   # deleted: where the gap is
    assert map_line(OLD, OLD, 0) is None and map_line(OLD, OLD, 99) is None


def test_an_insert_never_takes_the_line_that_follows_it():
    assert map_line("a\nb", "a\nINSERT\nb", 2) == 3                  # mid-file
    assert map_line("a\nb", "a\nINSERT\nb", 1) == 1
    assert map_line("a\nb", "TOP\na\nb", 1) == 2                     # at the start
    assert map_line("a\nb\nc", "a\nb\nINSERT\nc", 3) == 4           # right before the last line
    assert map_line("a\nb\nc", "a\nc", 3) == 2                       # after a delete
    assert map_line("a\nb\nc", "a\nB\nc", 2) == 2                   # inside a replace
    assert map_line("a\nb\nc", "a\nB\nc", 3) == 3


def test_a_line_of_an_empty_file_does_not_exist():
    assert map_line("", "a\nb", 1) is None


def test_carry_fetches_once_and_handles_unreadable():
    def contents(text):
        return {"encoding": "base64", "content": base64.b64encode(text.encode()).decode()}
    client = FakeClient({}, {}, {"repos/o/r/contents/f.py?ref=A": contents(OLD),
                                 "repos/o/r/contents/f.py?ref=B": contents("x\n" + OLD)})
    versions = FileVersions(client, "o/r")
    assert versions.carry("f.py", 2, "A", "A") == 2 and client.calls == []
    assert versions.carry("f.py", 2, "A", "B") == 3
    assert versions.carry("f.py", 3, "A", "B") == 4 and len(client.calls) == 2
    assert versions.carry("f.py", 2, "A", "GONE") is None
