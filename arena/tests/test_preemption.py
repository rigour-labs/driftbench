import subprocess
from pathlib import Path

from arena import preemption
from arena.corpus import BotComment, Pr
from arena.gitrepo import Git


def entry(targets, findings, status="FAIL"):
    return {"status": status, "commit": "c", "targets": targets, "findings": findings}


def t(id, path, start, end, severity="🟠 Major"):
    return {"id": id, "path": path, "start": start, "end": end, "severity": severity}


def f(path, line, id="r"):
    return {"path": path, "line": line, "id": id, "message": "m"}


def test_a_finding_near_an_acted_on_comment_pre_empts_it_once():
    e = entry([t("1", "a.ts", 10, 12), t("2", "a.ts", 14, 14), t("3", "b.ts", 5, 5)],
              [f("a.ts", 13), f("b.ts", 40)])
    assert [(tg["id"], fd["line"]) for tg, fd in preemption.matches(e)] == [("1", 13)]


def test_score_pools_prs_and_breaks_down_by_severity():
    results = {"prs": {
        "1": entry([t("1", "a.ts", 10, 10, "🔴 Critical"), t("2", "a.ts", 50, 50)], [f("a.ts", 11)]),
        "2": entry([t("3", "b.ts", 5, 5)], [f("c.ts", 1), f("c.ts", 2)]),
        "3": entry([t("4", "d.ts", 1, 1)], [f("d.ts", 1)], status="error"),  # a failed run is excluded, not a miss
    }}
    s = preemption.score(results)
    assert (s.prs, s.targets, s.preempted, s.rate.value, s.findings_per_pr.value) == (2, 3, 1, 1 / 3, 1.5)
    assert s.by_severity == {"🟠 Major": (0, 2), "🔴 Critical": (1, 1)}


def _commit(repo: Path, message: str, files: dict[str, str]) -> str:
    for name, body in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(body)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", message], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def test_targets_are_the_acted_on_code_comments_of_the_most_acted_on_reviewed_commit(tmp_path: Path):
    subprocess.run(["git", "init", "-q", "-b", "main", str(tmp_path)], check=True)
    for key, value in [("user.name", "t"), ("user.email", "t@example.com"), ("commit.gpgsign", "false")]:
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    _commit(tmp_path, "init", {"src/a.ts": "export const a = 1;\n"})
    reviewed = _commit(tmp_path, "feat: b", {"src/a.ts": "export const a = 1;\nexport const b = a / 0;\nexport const c = 3;\n",
                                              "README.md": "docs\n"})
    merged = _commit(tmp_path, "address review", {"src/a.ts": "export const a = 1;\nexport const b = a;\nexport const c = 3;\n"})
    comment = lambda id, path, line: BotComment(id, "", path, line, line, reviewed, "", "🟠 Major")
    pr = Pr(1, "", merged, merged, "", [reviewed], [comment(1, "src/a.ts", 2), comment(2, "src/a.ts", 3), comment(3, "README.md", 1)])
    commit, targets = preemption.acted_on_targets(Git(tmp_path), pr)
    assert commit == reviewed
    assert [(x.id, x.start) for x in targets] == [("1", 2)]  # line 3 was not changed; README is not code
