import subprocess
from pathlib import Path

from arena.corpus import BotComment, Pr
from arena.gitrepo import Git
from arena.tools.coderabbit import locate


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_locates_on_the_merge_and_drops_what_cannot_be_placed(tmp_path: Path):
    _git(tmp_path, "init", "-q", "-b", "main")
    for key, value in [("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")]:
        _git(tmp_path, "config", key, value)
    (tmp_path / "a.ts").write_text("one\ntwo\nthree\n")
    _git(tmp_path, "add", "-A"); _git(tmp_path, "commit", "-qm", "init")
    reviewed = _git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "a.ts").write_text("zero\none\nTWO\nthree\n")
    _git(tmp_path, "commit", "-qam", "address review")
    merge = _git(tmp_path, "rev-parse", "HEAD")

    comments = [
        BotComment(1, "u", "a.ts", 2, 2, reviewed),       # edited later: acted on
        BotComment(2, "u", "a.ts", 3, 3, reviewed),       # untouched, shifted down one line
        BotComment(3, "u", "a.ts", 1, 1, "f" * 40),        # commit force-pushed away and unfetchable
        BotComment(4, "u", "gone.ts", 1, 1, reviewed),     # file not in the merge
    ]
    located, dropped = locate(Git(tmp_path), Pr(9, "u", merge, merge, "2026-01-01", [reviewed], comments))

    assert [(l.finding.path, l.finding.line, l.acted_on) for l in located] == [("a.ts", 3, True), ("a.ts", 4, False)]
    assert dropped == 2
