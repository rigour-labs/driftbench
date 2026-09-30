"""Validate candidate Rigour rules on training repositories before they ship.

A rule ships only if, on repositories it was not designed from:
  - it catches real fixes: it fires on a file before a fix commit and less after it;
  - it is quiet on code as it stands: at most MAX_HITS_PER_5K_LOC hits at HEAD;
  - its HEAD hits are right: at least MIN_PRECISION of a hand-labelled sample.

    RIGOUR_CLI=.../rigour-cli/dist/cli.js python -m rlaif.rules.validate \\
        --rules solid/jsx-and-conditional --repos solidjs/solid-start --max-fixes 400
    python -m rlaif.rules.validate --check solid/jsx-and-conditional

Rigour's `scan-rules` command runs the rules (JSON lines: rule, file, line, message).
The sheet in rlaif/rules/sheets/ lists sampled HEAD hits as links for labelling;
it holds no copied code.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from arena.gitrepo import Commit, Git
from arena.repos import clone
from rlaif.fixtrees import code_files, mineable, refuse_eval_repo, run_rigour, split, write_pair

ROOT = Path(__file__).resolve().parent
SHEETS = ROOT / "sheets"
BATCH_FIXES = 50
SAMPLE = 20
MIN_PRECISION = 0.8
MAX_HITS_PER_5K_LOC = 1.0
LABEL = re.compile(r"^- \[([ yn])\] ")


@dataclass
class Report:
    rule: str
    fixes_scanned: int = 0
    catches: list[dict] = field(default_factory=list)
    head_hits: list[dict] = field(default_factory=list)
    head_loc: int = 0

    @property
    def hits_per_5k_loc(self) -> float:
        return 5000 * len(self.head_hits) / self.head_loc if self.head_loc else 0.0


def validate(rules: list[str], repos: list[str], max_fixes: int) -> dict[str, Report]:
    reports = {rule: Report(rule) for rule in rules}
    for repo in repos:
        refuse_eval_repo(repo)
        git = clone(repo)
        head = git.run("rev-parse", "origin/HEAD").strip()  # a reused clone's own HEAD is stale after fetch
        fixes = [c for c in git.first_parent_history(_root(git), head) if mineable(c)][-max_fixes:]
        for start in range(0, len(fixes), BATCH_FIXES):
            _scan_fixes(git, repo, fixes[start:start + BATCH_FIXES], reports)
        _scan_head(git, repo, head, reports)
    return reports


def catches(findings: list[dict]) -> list[dict]:
    """Findings on a fix's before-side file where the after side has fewer findings of that rule."""
    counts: Counter = Counter()
    before: dict[tuple, list[dict]] = {}
    for finding in findings:
        sha, side, path = split(finding["file"])
        key = (finding["rule"], sha, path)
        counts[(*key, side)] += 1
        if side == "before":
            before.setdefault(key, []).append({**finding, "fix": sha, "file": path})
    return [f for key, found in before.items() if counts[(*key, "before")] > counts[(*key, "after")] for f in found]


def _root(git: Git) -> str:
    return git.run("rev-list", "--max-parents=0", "origin/HEAD").split()[-1]


def _scan_fixes(git: Git, repo: str, fixes: list[Commit], reports: dict[str, Report]) -> None:
    tree = Path(tempfile.mkdtemp(prefix="rule-validate-"))
    try:
        written = [w for fix in fixes for path in code_files(fix) for w in write_pair(git, tree, fix, path)]
        findings = run_rigour(tree, ["scan-rules", "--rules", ",".join(reports), *written]) if written else []
        for report in reports.values():
            report.fixes_scanned += len(fixes)
        for found in catches(findings):
            reports[found["rule"]].catches.append({"repo": repo, **found})
    finally:
        shutil.rmtree(tree, ignore_errors=True)


def _scan_head(git: Git, repo: str, head: str, reports: dict[str, Report]) -> None:
    git.run("checkout", "-q", "--detach", head)
    files = [f for f in git.run("ls-files").splitlines() if f.endswith((".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs"))]
    loc = sum(_lines(Path(git.path) / f) for f in files)
    for finding in run_rigour(Path(git.path), ["scan-rules", "--rules", ",".join(reports)]):
        reports[finding["rule"]].head_hits.append({"repo": repo, "sha": head, **finding})
    for report in reports.values():
        report.head_loc += loc


def _lines(path: Path) -> int:
    try:
        return path.read_bytes().count(b"\n")
    except OSError:
        return 0


def write_sheet(report: Report, seed: int = 7) -> Path:
    """A labelling sheet of sampled HEAD hits: mark each [y] right or [n] wrong."""
    sample = random.Random(seed).sample(report.head_hits, min(SAMPLE, len(report.head_hits)))
    lines = [f"# {report.rule}", "",
             f"fixes scanned {report.fixes_scanned}, catches {len(report.catches)}, "
             f"HEAD hits {len(report.head_hits)} in {report.head_loc} lines ({report.hits_per_5k_loc:.2f} per 5k)", "",
             "## Catches (fires before the fix, less after)", ""]
    lines += [f"- https://github.com/{c['repo']}/commit/{c['fix']} `{c['file']}:{c['line']}`" for c in report.catches[:SAMPLE]]
    lines += ["", "## HEAD sample: mark [y] right or [n] wrong", ""]
    lines += [f"- [ ] https://github.com/{h['repo']}/blob/{h['sha']}/{h['file']}#L{h['line']} {h.get('message', '')}" for h in sample]
    path = SHEETS / f"{report.rule.replace('/', '__')}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def check_sheet(path: Path) -> dict:
    """Ship decision from a labelled sheet."""
    text = path.read_text()
    marks = [m.group(1) for m in map(LABEL.match, text.splitlines()) if m]
    labelled = [m for m in marks if m != " "]
    precision = labelled.count("y") / len(labelled) if labelled else None
    stats = re.search(r"catches (\d+), HEAD hits \d+ in \d+ lines \(([\d.]+) per 5k\)", text)
    caught, per_5k = (int(stats.group(1)), float(stats.group(2))) if stats else (0, float("inf"))
    ships = (caught >= 1 and per_5k <= MAX_HITS_PER_5K_LOC and precision is not None
             and precision >= MIN_PRECISION and len(labelled) == len(marks))
    return {"catches": caught, "hits_per_5k_loc": per_5k, "labelled": len(labelled), "of": len(marks),
            "precision": precision, "ships": ships}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate candidate Rigour rules on training repositories")
    parser.add_argument("--rules", default="", help="Comma-separated rule ids")
    parser.add_argument("--repos", nargs="*", default=[], help="Training repositories (owner/name)")
    parser.add_argument("--max-fixes", type=int, default=400, help="Newest N fix commits per repository")
    parser.add_argument("--check", default="", help="Report the ship decision from a labelled sheet")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_sheet(SHEETS / f"{args.check.replace('/', '__')}.md")))
        return
    for report in validate([r for r in args.rules.split(",") if r], args.repos, args.max_fixes).values():
        print(f"{report.rule}: {len(report.catches)} catches, {report.hits_per_5k_loc:.2f} hits/5k LOC -> {write_sheet(report)}")


if __name__ == "__main__":
    main()
