import json
import subprocess
from pathlib import Path

import pytest

from arena.gitrepo import Git
from rlaif import fixtrees


def _commit(repo: Path, message: str, files: dict[str, str]) -> None:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True)


@pytest.fixture
def git(tmp_path: Path) -> Git:
    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    for key, value in [("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")]:
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    _commit(repo, "init", {
        "package.json": '{"name": "root"}',
        "tsconfig.base.json": '{\n  // shared\n  "compilerOptions": {"stripInternal": true,},\n}',
        "packages/core/package.json": '{"dependencies": {"solid-js": "1"}}',
        "packages/core/tsconfig.json": '{"extends": "../../tsconfig.base.json"}',
        "packages/core/src/a.ts": "export const a = 1;\n",
    })
    _commit(repo, "fix: a", {"packages/core/src/a.ts": "export const a = 2;\n", "packages/core/src/new.ts": "x\n"})
    return Git(repo)


def _fix(git: Git):
    return git.first_parent_history(git.run("rev-list", "--max-parents=0", "HEAD").strip(), "HEAD")[-1]


def test_a_pair_carries_its_nearest_manifests_and_tsconfig_base(git: Git, tmp_path: Path):
    fix, tree = _fix(git), tmp_path / "tree"
    written = fixtrees.write_pair(git, tree, fix, "packages/core/src/a.ts")
    assert [fixtrees.split(w)[1:] for w in written] == [("before", "packages/core/src/a.ts"), ("after", "packages/core/src/a.ts")]
    side = tree / fix.sha / "before"
    assert json.loads((side / "packages/core/package.json").read_text())["dependencies"] == {"solid-js": "1"}
    assert (side / "packages/core/tsconfig.json").exists() and (side / "tsconfig.base.json").exists()
    assert not (side / "package.json").exists()  # the nearest package.json wins


def test_a_file_the_fix_added_has_no_pair(git: Git, tmp_path: Path):
    assert fixtrees.write_pair(git, tmp_path / "tree", _fix(git), "packages/core/src/new.ts") == []


def test_evaluation_repositories_are_refused():
    with pytest.raises(SystemExit, match="evaluation repository"):
        fixtrees.refuse_eval_repo("SUPABASE/supabase")


def test_rigour_cli_override_runs_the_local_build(monkeypatch):
    monkeypatch.setenv("RIGOUR_CLI", "/x/cli.js")
    assert fixtrees.rigour_command() == ["node", "/x/cli.js"]
