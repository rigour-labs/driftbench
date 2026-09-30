"""python -m arena mine|label|run|score|judge-pack|judge-merge — see arena/README.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arena import pipeline, verdicts
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
        p.add_argument("--repo", required=name != "score")
        if name != "score":
            p.add_argument("--sample", type=int, default=0, help="Only the first N PRs of the corpus (a smoke run)")
    sub.choices["run"].add_argument("--tool", required=True, help="coderabbit, or a name from arena/configs (e.g. rigour-semantic)")
    sub.choices["score"].add_argument("--scope", choices=["code", "all"], default="code")
    sub.choices["score"].add_argument("--all", action="store_true", help="Every corpus, pooled (design sets excluded)")
    jp = sub.add_parser("judge-pack", help="Write what still needs a verdict, with instructions for the judge")
    jp.add_argument("--repo", required=True)
    jp.add_argument("--out", required=True, type=Path)
    jm = sub.add_parser("judge-merge", help="Validate judge answers into arena/verdicts/")
    jm.add_argument("--repo", required=True)
    jm.add_argument("answers", nargs="+", type=Path)
    args = parser.parse_args()
    handlers = {"mine": _mine, "label": _label, "run": _run, "score": _score,
                "judge-pack": _judge_pack, "judge-merge": _judge_merge}
    handlers[args.command](args)


def _paths(repo: str) -> tuple[Path, Path, Path]:
    s = pipeline.slug(repo)
    return ROOT / "corpora" / f"{s}.json", ROOT / "labels" / f"{s}.json", ROOT / "results" / s


def _verdicts_path(repo: str) -> Path:
    return ROOT / "verdicts" / f"{pipeline.slug(repo)}.json"


def _results(results_dir: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(results_dir.glob("*.json"))}


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
    if args.all:
        _score_all(args.scope)
        return
    if not args.repo:
        raise SystemExit("score needs --repo or --all")
    _, labels_path, results_dir = _paths(args.repo)
    labels = json.loads(labels_path.read_text())
    results = _results(results_dir)
    judged = verdicts.Verdicts.load(_verdicts_path(args.repo))
    print(f"scope: {args.scope} (proximity: any finding near a later-fixed line)")
    _table(labels, results, args.scope, None)
    if judged.fixes:
        fixes, pairs = verdicts.unjudged(labels, results, judged, args.scope)
        print(f"\nscope: {args.scope} (judged: real bugs, findings that describe them); unjudged: {fixes} fixes, {pairs} pairs")
        _table(labels, results, args.scope, judged)


def _table(labels: dict, results: dict[str, dict], scope: str, judged) -> None:
    print(f"{'tool':<22}{'PRs':>5}{'bugs':>6}{'comments':>10}  {'recall':<20}{'hit rate':<20}{'comments/PR':<16}")
    for tool, result in results.items():
        s = pipeline.score_tool(labels, result, scope, judged)
        print(f"{tool:<22}{s.prs:>5}{s.bugs:>6}{s.comments:>10}  {_fmt(s.recall):<20}{_fmt(s.hit_rate):<20}{_fmt(s.comments_per_pr, pct=False):<16}")


def _score_all(scope: str) -> None:
    """Per-repo judged recall, then every tool pooled over the eval repos (design sets excluded)."""
    repos = []
    for corpus_path in sorted((ROOT / "corpora").glob("*.json")):
        repo = json.loads(corpus_path.read_text())["repo"]
        _, labels_path, results_dir = _paths(repo)
        if labels_path.exists() and any(results_dir.glob("*.json")):
            repos.append((repo, json.loads(labels_path.read_text()), _results(results_dir),
                          verdicts.Verdicts.load(_verdicts_path(repo))))
    tools = sorted({tool for _, _, results, _ in repos for tool in results})
    print(f"scope: {scope}; design sets (never pooled): {', '.join(sorted(pipeline.DESIGN_SETS))}")
    print(f"{'repo':<32}{'tool':<22}{'bugs':>6}  {'judged recall':<20}unjudged fixes/pairs")
    for repo, labels, results, judged in repos:
        fixes, pairs = verdicts.unjudged(labels, results, judged, scope)
        for tool in tools:
            if tool in results:
                s = pipeline.score_tool(labels, results[tool], scope, judged if judged.fixes else None)
                print(f"{repo:<32}{tool:<22}{s.bugs:>6}  {_fmt(s.recall) if judged.fixes else 'unjudged':<20}{fixes}/{pairs}")
    pooled = [r for r in repos if r[0] not in pipeline.DESIGN_SETS]
    for title, judged_only in (("proximity", False), ("judged", True)):
        print(f"\npooled over {len(pooled)} eval repos ({title})")
        print(f"{'tool':<22}{'PRs':>5}{'bugs':>6}{'comments':>10}  {'recall':<20}{'hit rate':<20}{'comments/PR':<16}")
        for tool in tools:
            entries = [(labels, results[tool], judged if judged_only else None)
                       for _, labels, results, judged in pooled if tool in results and (judged.fixes or not judged_only)]
            if entries:
                s = pipeline.pooled_score(entries, scope)
                print(f"{tool:<22}{s.prs:>5}{s.bugs:>6}{s.comments:>10}  {_fmt(s.recall):<20}{_fmt(s.hit_rate):<20}{_fmt(s.comments_per_pr, pct=False):<16}")


def _judge_pack(args) -> None:
    corpus_path, labels_path, results_dir = _paths(args.repo)
    merges = {str(pr.number): pr.merge_sha for pr in Corpus.load(corpus_path).prs}
    todo = verdicts.pack(json.loads(labels_path.read_text()), _results(results_dir),
                         verdicts.Verdicts.load(_verdicts_path(args.repo)), merges=merges)
    args.out.mkdir(parents=True, exist_ok=True)
    pipeline.write_json(args.out / "fixes.json", {"repo": args.repo, "fixes": todo})
    (args.out / "INSTRUCTIONS.md").write_text(verdicts.INSTRUCTIONS)
    print(f"{sum(1 for f in todo if not f['judged'])} fixes and {sum(len(f['candidates']) for f in todo)} pairs to judge -> {args.out}")


def _judge_merge(args) -> None:
    path = _verdicts_path(args.repo)
    judged = verdicts.Verdicts.load(path)
    added = sum(verdicts.merge(judged, json.loads(answers.read_text())) for answers in args.answers)
    judged.save(path, args.repo)
    print(f"{added} verdicts merged -> {path}")


def _fmt(metric, pct: bool = True) -> str:
    if metric.value is None:
        return "n/a"
    f = (lambda v: f"{100 * v:.0f}%") if pct else (lambda v: f"{v:.1f}")
    return f"{f(metric.value)} [{f(metric.low)}–{f(metric.high)}]" if metric.low is not None else f(metric.value)


if __name__ == "__main__":
    main()
