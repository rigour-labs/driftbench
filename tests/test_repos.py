from pathlib import Path

import pytest

from bench.repos import RepoListError, load_repos

ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 40


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "repos.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _entry(name: str = "o/r", pin: str = SHA, at: str = "2026-10-08T00:00:00Z", extra: str = "") -> str:
    return (
        f"  - name: {name}\n    licence: MIT\n    default_branch: main\n"
        f"    pin: {pin}\n    pinned_at: \"{at}\"\n{extra}"
    )


def test_shipped_repo_list_is_valid():
    repos = load_repos(ROOT / "repos.yaml", enabled_only=False)
    assert {r.name for r in repos} >= {"tailscale/tailscale", "zulip/zulip", "immich-app/immich"}
    assert all(len(r.pin) == 40 for r in repos)


def test_disabled_repos_are_skipped_by_default(tmp_path):
    path = _write(tmp_path, "repos:\n" + _entry("o/a") + _entry("o/b", extra="    enabled: false\n"))
    assert [r.name for r in load_repos(path)] == ["o/a"]
    assert len(load_repos(path, enabled_only=False)) == 2


def test_slug_is_filesystem_safe(tmp_path):
    path = _write(tmp_path, "repos:\n" + _entry("zulip/zulip"))
    assert load_repos(path)[0].slug == "zulip__zulip"


@pytest.mark.parametrize(
    "body, message",
    [
        ("repos: []\n", "non-empty"),
        ("repos:\n" + _entry(pin="main"), "40-character"),
        ("repos:\n" + _entry(name="no-slash"), "owner/repo"),
        ("repos:\n" + _entry(at="yesterday"), "ISO timestamp"),
        ("repos:\n" + _entry(at="2026-10-08T00:00:00"), "timezone"),
        ("repos:\n" + _entry() + _entry(), "listed twice"),
        ("repos:\n  - name: o/r\n", "missing licence"),
    ],
)
def test_invalid_entries_are_rejected(tmp_path, body, message):
    with pytest.raises(RepoListError, match=message):
        load_repos(_write(tmp_path, body))
