"""`bench issues diagnose`: which of an earlier run's missed points the reviewer considered and held back (paid).

Targets are the points an earlier issue judgment found raised by one entrant and not by the other
(`--raised-by`, `--missed-by`). Reads a diagnostic run's records, writes `--out` (saved after every point).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import sys
from collections import Counter
from pathlib import Path

import yaml

from bench.collect.corpus import CorpusError, read_corpus
from bench.collect.github import GitHubClient
from bench.harness.budget import Budget
from bench.harness.runner import read_record
from bench.issues.diagnose import ENTRANT, MAX_TOKENS, SYSTEM, candidates, diagnose_point, point_records
from bench.labels.openrouter import chat
from bench.labels.prelabel_cli import CLAUDE_MARKERS
from bench.points.points_file import read_points
from bench.points.texts import TextSource, point_text
from bench.repos import slug_of


def add_diagnose_action(actions: argparse._SubParsersAction, root: Path) -> None:
    diag = actions.add_parser("diagnose", help="did the reviewer hold back the issues it missed? (paid)")
    diag.add_argument("--run", type=Path, required=True, help="the diagnostic run directory")
    diag.add_argument("--judgments", type=Path, required=True, help="the earlier run's issue-judgments.yaml")
    diag.add_argument("--raised-by", default="claude-code-review")
    diag.add_argument("--missed-by", default="rigour-reviewer")
    diag.add_argument("--out", type=Path, required=True)
    diag.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    diag.add_argument("--points", type=Path, default=root / "work" / "points")
    diag.add_argument("--cache", type=Path, default=root / "work" / "cache")
    diag.add_argument("--model", required=True)
    diag.add_argument("--max-usd", type=float, required=True)
    diag.add_argument("--call-bound", type=float, default=0.05)
    diag.set_defaults(handler=cmd_diagnose)


def target_ids(judgments: dict, raised_by: str, missed_by: str) -> dict[str, str]:
    """point id -> repo, for points `raised_by` raised (yes or partly) and `missed_by` didn't."""
    return {pid: j["repo"] for pid, j in judgments.items() if not j.get("error")
            and j["verdicts"][raised_by]["verdict"] in ("yes", "partly") and j["verdicts"][missed_by]["verdict"] == "no"}


def summarise(points: dict) -> dict:
    return dict(sorted(Counter(p.get("bucket", "error") for p in points.values()).items()))


def cmd_diagnose(args: argparse.Namespace) -> int:
    if args.max_usd <= 0 or any(m in args.model.lower() for m in CLAUDE_MARKERS):
        print("diagnose needs --max-usd above zero and a model outside the Claude family", file=sys.stderr)
        return 1
    try:
        earlier = yaml.safe_load(args.judgments.read_text(encoding="utf-8"))["judgments"]
        targets = target_ids(earlier, args.raised_by, args.missed_by)
        out = yaml.safe_load(args.out.read_text(encoding="utf-8")) if args.out.exists() else {
            "model": args.model, "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
            "entrant": args.missed_by, "targets": len(targets), "spent_usd": 0.0, "points": {}}
        corpora = {c["repo"]: c for c in (read_corpus(p) for p in sorted(args.corpus.glob("*.json")))}
    except (OSError, KeyError, CorpusError, yaml.YAMLError) as exc:
        print(exc, file=sys.stderr)
        return 1
    budget, client = Budget(args.max_usd, {ENTRANT: args.call_bound}), GitHubClient(args.cache)
    call = functools.partial(chat, args.model, max_tokens=MAX_TOKENS)
    for repo in sorted(set(targets.values())):
        by_head = {r["head_sha"]: r for r in (read_record(p) for p in
                                              sorted((args.run / args.missed_by / slug_of(repo)).glob("*/*.json")))}
        points = {p["id"]: p for p in read_points(args.points / f"{slug_of(repo)}.json")["points"]}
        rounds = {pr["number"]: pr["rounds"] for pr in corpora[repo]["prs"]}
        texts = TextSource(client, repo)
        for pid in sorted(pid for pid, r in targets.items() if r == repo and pid not in out["points"]):
            point = points[pid]
            found = candidates(point_records(point, rounds[point["pr"]], by_head))
            result = diagnose_point(point, point_text(texts, point) or "", found, call, budget)
            out["points"][pid] = {"repo": repo, **result}
            out["spent_usd"] = round(out["spent_usd"] + result.get("cost_usd", 0), 6)
            args.out.write_text(yaml.safe_dump(out, sort_keys=True, allow_unicode=True), encoding="utf-8")
    print(f"{len(out['points'])} of {out['targets']} points: {summarise(out['points'])}; "
          f"reported spend ${out['spent_usd']:.4f} -> {args.out}")
    return 0
