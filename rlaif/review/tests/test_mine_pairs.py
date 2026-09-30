import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from arena.gitrepo import Git
from rlaif.review import mine_pairs as m


def _commit(repo: Path, message: str, files: dict[str, str], days_ago: int) -> str:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    env = {**os.environ, "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True, env=env)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    for key, value in [("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")]:
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    shas = {
        "init": _commit(tmp_path, "init", {"src/total.ts": "export const zero = 0;\n"}, 400),
        "clean": _commit(tmp_path, "feat: add greeting", {"src/greet.ts": "export const hi = (n: string) => `hi ${n}`;\n"}, 390),
        "bug": _commit(tmp_path, "feat: sum items", {"src/total.ts": "export const zero = 0;\nexport const total = (xs: number[]) => xs.reduce((a, b) => a + b, 1);\n"}, 380),
        "recent": _commit(tmp_path, "feat: add farewell", {"src/bye.ts": "export const bye = 1;\n"}, 5),
    }
    shas["fix"] = _commit(tmp_path, "fix(total): start the sum at zero (#42)", {"src/total.ts": "export const zero = 0;\nexport const total = (xs: number[]) => xs.reduce((a, b) => a + b, 0);\n"}, 1)
    git = Git(tmp_path)
    return git, shas, git.first_parent_history(shas["init"], shas["fix"])


def test_a_fix_is_traced_to_the_commit_that_wrote_the_line_with_what_replaced_it(repo):
    git, shas, history = repo
    sites = m.introducing_sites(git, history)
    site = sites[(shas["bug"], "src/total.ts")]
    assert sorted(site.lines) == [2] and [f.sha for f in site.fixes] == [shas["fix"]]
    assert site.replacement == ["export const total = (xs: number[]) => xs.reduce((a, b) => a + b, 0);"]


def test_the_target_reads_like_a_review_comment_not_a_label(repo):
    git, shas, history = repo
    finding = m.target(m.introducing_sites(git, history)[(shas["bug"], "src/total.ts")], "src/total.ts")["findings"][0]
    assert finding["description"] == "Start the sum at zero."
    assert finding["line"] == 2 and "reduce((a, b) => a + b, 0)" in finding["suggestion"]
    assert m.target(None, "src/greet.ts") == {"findings": []}


def test_clean_commits_are_old_code_changes_no_fix_traced_back_to(repo):
    git, shas, history = repo
    clean = m.clean_commits(git, history, {shas["bug"]}, datetime.now(timezone.utc))
    assert [c.sha for c in clean] == [shas["clean"]]  # not the buggy one, not the fix, not the 5-day-old one


def test_evaluation_repositories_are_refused():
    with pytest.raises(SystemExit, match="evaluation repository"):
        m.mine("TanStack/router", 10)


def test_balance_keeps_every_bug_and_as_many_clean_examples():
    import random
    examples = [{"label": "bug", "i": 0}] + [{"label": "clean", "i": i} for i in range(1, 10)]
    kept = m.balance(examples, random.Random(1))
    assert [e["label"] for e in kept] == ["bug", "clean"]
