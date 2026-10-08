import subprocess
from pathlib import Path

import pytest

from bench import name_guard
from bench.__main__ import main


@pytest.fixture
def names_file(tmp_path: Path) -> Path:
    path = tmp_path / "blocked.txt"
    path.write_text("# comment\nExamplecorp\nWidget Works\n", encoding="utf-8")
    return path


def test_matches_whole_words_ignoring_case(names_file):
    pattern = name_guard.load_pattern(names_file)
    hits = name_guard.scan_text(pattern, "f", "ok\nfrom EXAMPLECORP data\nwidget  works here\n")
    assert [h.line for h in hits] == [2, 3]


def test_does_not_match_inside_longer_words(names_file):
    pattern = name_guard.load_pattern(names_file)
    assert name_guard.scan_text(pattern, "f", "examplecorps\nsubexamplecorp\n") == []


def test_missing_or_empty_list_fails_closed(tmp_path):
    with pytest.raises(name_guard.GuardConfigError):
        name_guard.load_pattern(tmp_path / "absent.txt")
    empty = tmp_path / "empty.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(name_guard.GuardConfigError):
        name_guard.load_pattern(empty)


def test_scans_directories_and_skips_binary(names_file, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "verdicts.json").write_text('{"note": "Examplecorp"}', encoding="utf-8")
    (assets / "blob.bin").write_bytes(b"\0Examplecorp")
    hits = name_guard.scan_paths(name_guard.load_pattern(names_file), [assets])
    assert [Path(h.where).name for h in hits] == ["verdicts.json"]


def test_unreadable_file_fails_closed(names_file, tmp_path):
    with pytest.raises(name_guard.GuardConfigError, match="cannot read"):
        name_guard.scan_paths(name_guard.load_pattern(names_file), [tmp_path / "gone.json"])


def test_commit_messages_and_authors_are_scanned(names_file, tmp_path):
    repo = tmp_path / "repo"
    git = ["git", "-C", str(repo), "-c", "user.name=A", "-c", "user.email=a@x.test"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "fix for widget works"], check=True)
    text = name_guard.commit_messages(repo, "HEAD")
    assert name_guard.scan_text(name_guard.load_pattern(names_file), "log", text)


def test_cli_exit_codes(names_file, tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_BLOCKED_NAMES", str(tmp_path / "absent.txt"))
    assert main(["guard"]) == 2
    monkeypatch.setenv("BENCH_BLOCKED_NAMES", str(names_file))
    leak = tmp_path / "leak.md"
    leak.write_text("Examplecorp\n", encoding="utf-8")
    assert main(["guard", str(leak)]) == 1
