"""Arena pipeline: label PRs with later-fixed bugs, run tools, score them.

Files, all reproducible from the pinned corpus and a clone:
  arena/corpora/<owner>__<name>.json   PRs, commits, bot comment locations
  arena/labels/<owner>__<name>.json    SZZ bugs per PR (with parameters)
  arena/results/<owner>__<name>/<tool>.json   each tool's findings per PR
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from arena import szz
from arena.corpus import Corpus
from arena.repos import clone, ensure_commits
from arena.score import Finding, PrResult, ToolScore, in_scope, score
from arena.tools import coderabbit
from arena.tools.rigour import RigourConfig, review
from arena.verdicts import Verdicts

ROOT = Path(__file__).parent


def slug(repo: str) -> str:
    return repo.replace("/", "__")


def label(corpus: Corpus) -> dict:
    """SZZ bugs for every PR, fixed after merge up to the corpus snapshot."""
    git = clone(corpus.repo)
    oldest = min(corpus.prs, key=lambda p: p.merged_at).merge_sha
    history = git.first_parent_history(git.parent(oldest), corpus.snapshot_sha)
    prs, skipped, blame_cache = {}, [], {}
    for pr in corpus.prs:
        try:
            bugs = szz.bugs_introduced(git, pr.merge_sha, set(), corpus.snapshot_sha, history, blame_cache)
        except ValueError:
            skipped.append(pr.number)  # merge commit not on the default branch's first-parent chain
            continue
        prs[str(pr.number)] = [asdict(b) for b in bugs]
    return {
        "repo": corpus.repo, "snapshot_sha": corpus.snapshot_sha,
        "params": {"fix_subject": szz.FIX_SUBJECT.pattern, "not_a_defect": szz.NOT_A_DEFECT.pattern,
                   "max_fix_files": szz.MAX_FIX_FILES},
        "skipped": skipped,
        "prs": prs,
    }


def run_coderabbit(corpus: Corpus) -> dict:
    git = clone(corpus.repo)
    prs = {}
    for pr in corpus.prs:
        ensure_commits(git, pr.number, [c.commit for c in pr.comments])
        located, dropped = coderabbit.locate(git, pr)
        prs[str(pr.number)] = {
            "findings": [asdict(l.finding) | {"acted_on": l.acted_on} for l in located],
            "dropped": dropped,
        }
    return {"tool": "coderabbit", "prs": prs}


def run_rigour(corpus: Corpus, cfg: RigourConfig) -> dict:
    git = clone(corpus.repo)
    prs = {}
    for pr in corpus.prs:
        run = review(git, pr.merge_sha, cfg)
        prs[str(pr.number)] = {"findings": [asdict(f) for f in run.findings], "seconds": round(run.seconds, 2),
                               "status": run.status, "error": run.error}
    return {"tool": cfg.name, "flags": list(cfg.flags), "config": str(cfg.config or ""), "prs": prs}


#: Repos whose misses the rules were designed from: scored, never used for claims.
DESIGN_SETS = {"TanStack/router"}


def score_tool(labels: dict, results: dict, scope: str = "code", verdicts: Verdicts | None = None) -> ToolScore:
    """Proximity score, or with `verdicts` the judged score: real bugs only, hits only by findings
    judged to describe the bug (unjudged counts as not describing; `verdicts.unjudged` reports them)."""
    return pooled_score([(labels, results, verdicts)], scope)


def pooled_score(repos: list[tuple[dict, dict, Verdicts | None]], scope: str = "code") -> ToolScore:
    """One tool's score over several repos' PRs, bootstrapped over all of them together.

    Each entry is (labels, results, verdicts); verdicts must be given for all or none.
    """
    rows: list[PrResult] = []
    judges: dict[str, tuple[Verdicts, str]] = {}
    for index, (labels, results, verdicts) in enumerate(repos):
        for number, bugs in labels["prs"].items():
            entry = results["prs"].get(number)
            if entry is None or entry.get("status") == "error":
                continue
            findings = [Finding(f["path"], f["line"], f.get("correct"), str(f.get("id", "")), f.get("message", ""))
                        for f in entry["findings"] if in_scope(f["path"], scope)]
            located = [(b["path"], b["lines"], b["fix_sha"]) for b in bugs
                       if in_scope(b["path"], scope) and (verdicts is None or verdicts.is_real(b["fix_sha"]))]
            key = f"{index}#{number}"
            rows.append(PrResult(key, located, findings))
            if verdicts is not None:
                judges[key] = (verdicts, results["tool"])
    if not judges:
        return score(rows)

    def describes(key: str, finding: Finding, bug: tuple) -> bool:
        verdicts, tool = judges[key]
        return bool(verdicts.describes(tool, key.split("#", 1)[1], finding.id, bug[2]))
    return score(rows, describes=describes)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
