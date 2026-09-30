"""python -m arena mine|label|run|score — see arena/README.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arena import pipeline
from arena.corpus import Corpus, mine
from arena.tools.rigour import RigourConfig

ROOT = pipeline.ROOT


def main() -> None:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("mine", help="Mine PRs a bot reviewed into a pinned corpus")
    m.add_argument("--repo", required=True)
    m.add_argument("--limit", type=int, default=50)
    m.add_argument("--min-age-days", type=int, default=60)
    for name in ("label", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True)
        if name != "score":
            p.add_argument("--sample", type=int, default=0, help="Only the first N PRs of the corpus (a smoke run)")
    sub.choices["run"].add_argument("--tool", required=True, help="coderabbit, or a name from arena/configs (e.g. rigour-semantic)")
    sub.choices["score"].add_argument("--scope", choices=["code", "all"], default="code")
    args = parser.parse_args()
    {"mine": _mine, "label": _label, "run": _run, "score": _score}[args.command](args)


def _paths(repo: str) -> tuple[Path, Path, Path]:
    s = pipeline.slug(repo)
    return ROOT / "corpora" / f"{s}.json", ROOT / "labels" / f"{s}.json", ROOT / "results" / s


def _mine(args) -> None:
    corpus = mine(args.repo, args.limit, args.min_age_days)
    corpus.save(_paths(args.repo)[0])
    print(f"{len(corpus.prs)} PRs, {sum(len(p.comments) for p in corpus.prs)} bot comments, snapshot {corpus.snapshot_sha[:10]}")


def _label(args) -> None:
    corpus_path, labels_path, _ = _paths(args.repo)
    labels = pipeline.label(_sampled(Corpus.load(corpus_path), args.sample))
    pipeline.write_json(labels_path, labels)
    print(f"{sum(len(b) for b in labels['prs'].values())} later-fixed bugs across {len(labels['prs'])} PRs")


def _run(args) -> None:
    corpus_path, _, results_dir = _paths(args.repo)
    corpus = _sampled(Corpus.load(corpus_path), args.sample)
    if args.tool == "coderabbit":
        results = pipeline.run_coderabbit(corpus)
    else:
        results = pipeline.run_rigour(corpus, _rigour_config(args.tool))
    pipeline.write_json(results_dir / f"{args.tool}.json", results)
    print(f"{args.tool}: {sum(len(p['findings']) for p in results['prs'].values())} findings on {len(results['prs'])} PRs")


def _sampled(corpus: Corpus, sample: int) -> Corpus:
    if sample > 0:
        corpus.prs = corpus.prs[:sample]
    return corpus


def _rigour_config(name: str) -> RigourConfig:
    spec = json.loads((ROOT / "configs" / f"{name}.json").read_text())
    config = spec.get("config")
    return RigourConfig(name, (ROOT / "configs" / config) if config else None, tuple(spec.get("flags", [])))


def _score(args) -> None:
    _, labels_path, results_dir = _paths(args.repo)
    labels = json.loads(labels_path.read_text())
    print(f"scope: {args.scope}")
    print(f"{'tool':<22}{'PRs':>5}{'bugs':>6}{'comments':>10}  {'recall':<20}{'hit rate':<20}{'comments/PR':<16}")
    for path in sorted(results_dir.glob("*.json")):
        s = pipeline.score_tool(labels, json.loads(path.read_text()), args.scope)
        print(f"{path.stem:<22}{s.prs:>5}{s.bugs:>6}{s.comments:>10}  {_fmt(s.recall):<20}{_fmt(s.hit_rate):<20}{_fmt(s.comments_per_pr, pct=False):<16}")


def _fmt(metric, pct: bool = True) -> str:
    if metric.value is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.1f}")
    return f"{f(metric.value)} [{f(metric.low)}–{f(metric.high)}]" if metric.low is not None else f(metric.value)


if __name__ == "__main__":
    main()
