from pathlib import Path

import pytest

from bench.adapters.citations import changed_path, citations, diagnostics

FIXTURES = Path(__file__).parent / "fixtures"
CHANGED = {"pkg/worker/loop.go", "pkg/worker/window.go", "README.md"}


def cited(name: str) -> list[tuple[str, int, int | None]]:
    text = (FIXTURES / name).read_text()
    return [(f.path, f.line, f.end_line) for f in citations(text, CHANGED)]


@pytest.mark.parametrize("name, expected", [
    ("cc-review-backticks.md", [("pkg/worker/loop.go", 42, None), ("pkg/worker/window.go", 10, 14)]),
    ("cc-review-markdown-links.md", [("pkg/worker/loop.go", 42, None), ("pkg/worker/window.go", 10, 14)]),
    ("cc-review-blob-links.md", [("pkg/worker/loop.go", 40, 44), ("pkg/worker/window.go", 12, None)]),
    ("cc-review-no-findings.md", []),
])
def test_every_citation_shape_is_read_once(name, expected):
    assert cited(name) == expected


def test_only_changed_files_count_and_prefixes_are_resolved():
    assert changed_path("./pkg/worker/loop.go", CHANGED) == "pkg/worker/loop.go"
    assert changed_path("//github.com/o/r/blob/abc/pkg/worker/loop.go", CHANGED) == "pkg/worker/loop.go"
    assert changed_path("other/loop.go", CHANGED) is None and changed_path("worker/loop.go", CHANGED) is None
    assert citations("see vendor/lib.go:10", CHANGED) == []


def test_diagnostics_are_numbers_only_and_tell_a_miss_from_silence():
    blob = diagnostics((FIXTURES / "cc-review-blob-links.md").read_text(), CHANGED, 7)
    assert blob == {"result_chars": blob["result_chars"], "citation_like": 2, "link_like": 2, "cited_changed": 2,
                    "files_named": 2, "num_turns": 7}
    silent = diagnostics((FIXTURES / "cc-review-no-findings.md").read_text(), CHANGED, 3)
    assert silent["citation_like"] == 0 and silent["cited_changed"] == 0 and silent["result_chars"] > 0
    assert all(isinstance(v, int) for v in blob.values())


def test_files_named_counts_changed_files_mentioned_without_a_line():
    from bench.adapters.citations import files_named
    text = "The retry loop in loop.go never stops; window.go looks fine. Unrelated: myloop.gopher."
    assert files_named(text, CHANGED) == 2
    assert files_named("pkg/worker/loop.go is touched", CHANGED) == 1 and files_named("nothing here", CHANGED) == 0
