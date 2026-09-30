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


def test_a_fix_is_blamed_once_for_every_pr_that_shares_it(repo: Path):
    _commit(repo, "init", {"src/a.ts": "a1\na2\n", "src/b.ts": "b1\n"})
    first = _commit(repo, "feat: a (#1)", {"src/a.ts": "a1\nA2 = 1\n"})
    second = _commit(repo, "feat: a again (#2)", {"src/a.ts": "a1\nA2 = 1\nA3 = 2\n"})
    head = _commit(repo, "fix: a values", {"src/a.ts": "a1\nA2 = 10\nA3 = 20\n"})

    class Counting(Git):
        blames = 0

        def blame_ranges(self, *args):
            Counting.blames += 1
            return super().blame_ranges(*args)

    git = Counting(repo)
    history, cache = git.first_parent_history(git.parent(first), head), {}
    bugs = [bugs_introduced(git, merge, set(), head, history, cache) for merge in (first, second)]

    assert [[(b.path, b.lines) for b in found] for found in bugs] == [[("src/a.ts", [2])], [("src/a.ts", [3])]]
    assert Counting.blames == 1


FN = ("export function normalizeRoutePath(dir: string, routePath: string) {\n"
      "  const joined = `/${dir}${routePath}`.replace(/\\/+/g, '/');\n"
      "  return joined.endsWith('/') && joined.length > 1 ? joined.slice(0, -1) : joined;\n"
      "}\n")


def test_a_function_the_pr_moved_between_files_is_not_blamed_on_the_pr(repo: Path):
    _commit(repo, "init", {"src/a.ts": "export const a = 1;\n" + FN, "src/b.ts": "export const b = 2;\n"})
    merge = _commit(repo, "refactor: move path helpers (#20)", {"src/a.ts": "export const a = 1;\n",
                                                                 "src/b.ts": "export const b = 2;\n" + FN})
    head = _commit(repo, "fix: keep trailing slash for root", {"src/b.ts": "export const b = 2;\n" + FN.replace("joined.length > 1", "joined.length > 2")})
    assert bugs_introduced(Git(repo), merge, set(), head) == []


def test_code_the_pr_copied_unchanged_is_not_blamed_but_its_edited_line_is(repo: Path):
    _commit(repo, "init", {"src/a.ts": FN})
    edited = FN.replace("replace(/\\/+/g, '/')", "replace(/\\/\\//g, '/')")
    merge = _commit(repo, "feat: second copy for virtual routes (#21)", {"src/virtual.ts": edited})
    fixed = edited.replace("replace(/\\/\\//g, '/')", "replace(/\\/+/g, '/')").replace("joined.length > 1", "joined.length > 2")
    head = _commit(repo, "fix: collapse repeated slashes in virtual routes", {"src/virtual.ts": fixed})
    bugs = bugs_introduced(Git(repo), merge, set(), head)
    assert [(b.path, b.lines) for b in bugs] == [("src/virtual.ts", [2])]


def test_a_real_merge_attributes_lines_written_on_its_branch(repo: Path):
    _commit(repo, "init", {"src/total.ts": BASE})
    subprocess.run(["git", "-C", str(repo), "checkout", "-qb", "feature"], check=True)
    _commit(repo, "sum items", {"src/total.ts": PR})
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "main"], check=True)
    _commit(repo, "chore: unrelated", {"README.md": "x\n"})
    subprocess.run(["git", "-C", str(repo), "merge", "-q", "--no-ff", "-m", "Merge pull request #13", "feature"], check=True)
    merge = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    head = _commit(repo, "fix: off-by-one in total limit", {"src/total.ts": PR.replace("items.length - 1", "items.length")})
    bugs = bugs_introduced(Git(repo), merge, set(), head)
    assert [(b.path, b.lines) for b in bugs] == [("src/total.ts", [4])]
