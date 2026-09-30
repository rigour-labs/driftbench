import subprocess
from pathlib import Path

import pytest

from arena.gitrepo import Git
from arena.szz import bugs_introduced, is_fix


def _commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    for key, value in [("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")]:
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    return tmp_path


BASE = "export function total(items: number[]) {\n  let sum = 0;\n  return sum;\n}\n"
PR = ("export function total(items: number[]) {\n  let sum = 0;\n"
      "  for (const i of items) sum += i;\n  const limit = items.length - 1;\n  return sum;\n}\n")


def test_attributes_a_later_fix_to_the_line_the_pr_wrote(repo: Path):
    _commit(repo, "init", {"src/total.ts": BASE, "src/other.ts": "export const x = 1;\n"})
    merge = _commit(repo, "feat: sum items (#12)", {"src/total.ts": PR})
    _commit(repo, "fix: other thing", {"src/other.ts": "export const x = 2;\n"})
    _commit(repo, "refactor: rename sum", {"src/total.ts": PR.replace("  return sum;", "  return sum; // total")})
    head = _commit(repo, "fix: off-by-one in total limit", {"src/total.ts": PR.replace("items.length - 1", "items.length").replace("  return sum;", "  return sum; // total")})

    git = Git(repo)
    bugs = bugs_introduced(git, merge, set(), head)
    shared = bugs_introduced(git, merge, set(), head, git.first_parent_history(git.parent(merge), head))

    assert [(b.path, b.lines, b.fix_subject) for b in bugs] == [("src/total.ts", [4], "fix: off-by-one in total limit")]
    assert shared == bugs


def test_does_not_blame_the_pr_for_lines_it_did_not_write(repo: Path):
    _commit(repo, "init", {"src/total.ts": BASE})
    merge = _commit(repo, "feat: loop (#13)", {"src/total.ts": PR})
    head = _commit(repo, "fix: initial value", {"src/total.ts": PR.replace("let sum = 0;", "let sum = 0n as unknown as number;")})
    assert bugs_introduced(Git(repo), merge, set(), head) == []


def test_classifies_fix_subjects_conservatively():
    assert is_fix("fix(router): preserve search params on redirect")
    assert is_fix("Revert \"feat: cache loader\"")
    assert not is_fix("fix typo in README")
    assert not is_fix("fix(deps): bump vite")
    assert not is_fix("feat: add loader")


def test_first_parent_history_lists_subjects_parents_and_files(repo: Path):
    first = _commit(repo, "init", {"a.ts": "1\n"})
    second = _commit(repo, "fix: b", {"a.ts": "2\n", "dir/b.ts": "x\n"})
    [commit] = Git(repo).first_parent_history(first, second)
    assert (commit.sha, commit.parent, commit.subject, commit.files) == (second, first, "fix: b", ("a.ts", "dir/b.ts"))
