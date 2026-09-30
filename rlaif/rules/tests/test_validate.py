from pathlib import Path

from rlaif.rules import validate as v


def f(rule, sha, side, path, line=1):
    return {"rule": rule, "file": f"{sha}/{side}/{path}", "line": line, "message": "m"}


def test_a_catch_fires_before_the_fix_and_less_after_it():
    found = v.catches([
        f("r", "s1", "before", "a.ts", 3), f("r", "s1", "after", "a.ts", 9),  # same count: not fixed
        f("r", "s2", "before", "b.ts", 4),                                     # gone after: caught
        f("r", "s3", "after", "c.ts", 1),                                      # only after: introduced
    ])
    assert [(c["fix"], c["file"], c["line"]) for c in found] == [("s2", "b.ts", 4)]


def _sheet(monkeypatch, tmp_path: Path, marks: list[str], catches: int = 1, per_5k: float = 0.4) -> Path:
    report = v.Report("r", fixes_scanned=10, catches=[{"repo": "o/n", "fix": "s", "file": "a.ts", "line": 1}] * catches,
                      head_hits=[{"repo": "o/n", "sha": "h", "file": f"f{i}.ts", "line": i} for i in range(len(marks))],
                      head_loc=int(5000 * len(marks) / per_5k) if per_5k else 1)
    monkeypatch.setattr(v, "SHEETS", tmp_path)
    path = v.write_sheet(report)
    text = path.read_text()
    for mark in marks:
        text = text.replace("- [ ] ", f"- [{mark}] ", 1)
    path.write_text(text)
    return path


def test_a_rule_ships_only_when_it_catches_is_quiet_and_its_labelled_hits_are_right(monkeypatch, tmp_path: Path):
    assert v.check_sheet(_sheet(monkeypatch, tmp_path, ["y"] * 9 + ["n"]))["ships"] is True
    assert v.check_sheet(_sheet(monkeypatch, tmp_path, ["y"] * 7 + ["n"] * 3))["ships"] is False   # precision 70%
    assert v.check_sheet(_sheet(monkeypatch, tmp_path, ["y"] * 10, catches=0))["ships"] is False   # never caught a fix
    assert v.check_sheet(_sheet(monkeypatch, tmp_path, ["y"] * 10, per_5k=2.0))["ships"] is False  # too noisy
    assert v.check_sheet(_sheet(monkeypatch, tmp_path, ["y"] * 9 + [" "]))["ships"] is False       # sheet not finished
