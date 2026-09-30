import pytest

from rlaif.intent import mine


def test_sites_group_by_fix_and_original_path_with_nested_directories():
    sites = [
        {"file": "abc/before/src/a/b.ts", "function": "f"},
        {"file": "abc/after/src/a/b.ts", "function": "f"},
        {"file": "def/before/c.ts", "function": "g"},
    ]
    grouped = mine._group(sites)
    assert set(grouped) == {("abc", "src/a/b.ts"), ("def", "c.ts")}
    assert [s["function"] for s in grouped[("abc", "src/a/b.ts")]["after"]] == ["f"]
    assert "after" not in grouped[("def", "c.ts")]


def test_evaluation_repositories_are_refused_before_cloning(monkeypatch):
    monkeypatch.setattr(mine, "clone", lambda repo: pytest.fail("cloned an evaluation repository"))
    with pytest.raises(SystemExit, match="evaluation repository"):
        mine.mine("Lodash/Lodash")


def test_rigour_cli_override_runs_the_local_build(monkeypatch):
    monkeypatch.setenv("RIGOUR_CLI", "/x/cli.js")
    assert mine.rigour_command() == ["node", "/x/cli.js"]
