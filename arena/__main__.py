"""python -m arena mine|label|run|score|judge-pack|judge-merge — see arena/README.md."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arena import pipeline, preemption, verdicts
from arena.repos import clone
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
    m.add_argument("--branch", default="", help="Branch the reviewed PRs merged into (default: the repo's default branch)")
    for name in ("label", "run", "score"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=name != "score")
        if name != "score":
            p.add_argument("--sample", type=int, default=0, help="Only the first N PRs of the corpus (a smoke run)")
    sub.choices["run"].add_argument("--tool", required=True, help="coderabbit, or a name from arena/configs (e.g. rigour-semantic)")
    sub.choices["score"].add_argument("--scope", choices=["code", "all"], default="code")
    sub.choices["score"].add_argument("--all", action="store_true", help="Every corpus, pooled (design sets excluded)")
    pr_ = sub.add_parser("preempt-run", help="Review each PR as CodeRabbit saw it; record its acted-on comments")
    pr_.add_argument("--repo", required=True)
    pr_.add_argument("--tool", required=True, help="A name from arena/configs")
    pr_.add_argument("--sample", type=int, default=0)
    ps = sub.add_parser("preempt-score", help="Share of CodeRabbit's acted-on comments each tool raised before the PR")
    ps.add_argument("--repo", default="", help="One repo; default: every repo with pre-emption results, pooled")
    pp = sub.add_parser("preempt-sample", help="Random matched pairs to judge")
    pp.add_argument("--size", type=int, default=50)
    pp.add_argument("--out", required=True, type=Path)
    jp = sub.add_parser("judge-pack", help="Write what still needs a verdict, with instructions for the judge")
    jp.add_argument("--repo", required=True)
    jp.add_argument("--out", required=True, type=Path)
    jm = sub.add_parser("judge-merge", help="Validate judge answers into arena/verdicts/")
    jm.add_argument("--repo", required=True)
    jm.add_argument("answers", nargs="+", type=Path)
    args = parser.parse_args()
    handlers = {"mine": _mine, "label": _label, "run": _run, "score": _score,
                "judge-pack": _judge_pack, "judge-merge": _judge_merge,
                "preempt-run": _preempt_run, "preempt-score": _preempt_score, "preempt-sample": _preempt_sample}
    handlers[args.command](args)


def _paths(repo: str) -> tuple[Path, Path, Path]:
    s = pipeline.slug(repo)
    return ROOT / "corpora" / f"{s}.json", ROOT / "labels" / f"{s}.json", ROOT / "results" / s


def _verdicts_path(repo: str) -> Path:
    return ROOT / "verdicts" / f"{pipeline.slug(repo)}.json"


def _results(results_dir: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text()) for path in sorted(results_dir.glob("*.json"))}


def _mine(args) -> None:
    corpus = mine(args.repo, args.limit, args.min_age_days, args.branch)
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
    return RigourConfig(name, (ROOT / "configs" / config) if config else None, _expand(spec.get("flags", [])))


def _expand(flags: list[str]) -> tuple[str, ...]:
    """Flags may name environment variables (`${RIGOUR_MODEL_PATH}`); a missing one is an error, not a literal."""
    expanded = tuple(os.path.expandvars(flag) for flag in flags)
    missing = [flag for flag in expanded if "${" in flag]
    if missing:
        raise SystemExit(f"unset environment variable in config flags: {missing}")
    return expanded


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


def _preempt_path(repo: str, tool: str) -> Path:
    return ROOT / "results" / pipeline.slug(repo) / "pre-pr" / f"{tool}.json"


def _preempt_run(args) -> None:
    corpus = _sampled(Corpus.load(_paths(args.repo)[0]), args.sample)
    results = preemption.run(clone(args.repo), corpus.prs, _rigour_config(args.tool)) | {"repo": args.repo}
    pipeline.write_json(_preempt_path(args.repo, args.tool), results)
    print(f"{args.tool}: {len(results['prs'])} PRs with acted-on comments reviewed as CodeRabbit saw them")


def _preempt_results() -> dict[str, list[dict]]:
    """tool -> results for every repo with pre-emption results (design sets excluded unless named)."""
    by_tool: dict[str, list[dict]] = {}
    for path in sorted((ROOT / "results").glob("*/pre-pr/*.json")):
        by_tool.setdefault(path.stem, []).append(json.loads(path.read_text()))
    return by_tool


def _preempt_score(args) -> None:
    print("pre-emption: CodeRabbit comments the developer acted on, raised by the tool before the PR (proximity, code scope)")
    print(f"{'tool':<22}{'PRs':>6}{'targets':>9}{'raised':>8}  {'rate':<20}{'findings/PR':<16}by severity (raised/targets)")
    paths = sorted((ROOT / "results").glob(f"{pipeline.slug(args.repo) if args.repo else '*'}/pre-pr/*.json"))
    pooled: dict[str, dict] = {}
    for path in paths:
        if not args.repo and json.loads(path.read_text()).get("repo") in pipeline.DESIGN_SETS:
            continue
        data = json.loads(path.read_text())
        merged = pooled.setdefault(path.stem, {"tool": path.stem, "prs": {}})
        merged["prs"].update({f"{path.parent.parent.name}#{k}": v for k, v in data["prs"].items()})
    for tool, results in pooled.items():
        s = preemption.score(results)
        severity = ", ".join(f"{k or '?'} {h}/{n}" for k, (h, n) in s.by_severity.items())
        print(f"{tool:<22}{s.prs:>6}{s.targets:>9}{s.preempted:>8}  {_fmt(s.rate):<20}{_fmt(s.findings_per_pr, pct=False):<16}{severity}")


def _preempt_sample(args) -> None:
    pairs = []
    for path in sorted((ROOT / "results").glob("*/pre-pr/*.json")):
        data = json.loads(path.read_text())
        repo = path.parent.parent.name.replace("__", "/", 1)
        pairs += [{"repo": repo, "tool": data["tool"], **p} for p in preemption.sample(data, 10_000)]
    import random
    random.Random(7).shuffle(pairs)
    pipeline.write_json(args.out, {"pairs": pairs[:args.size], "instructions": (
        "For each pair, read the CodeRabbit comment (gh api repos/<repo>/pulls/comments/<comment>) and the Rigour finding "
        "message. same_issue=true only if both point at the same problem. Answer [{comment, finding, same_issue, reason}].")})
    print(f"{min(args.size, len(pairs))} of {len(pairs)} matched pairs -> {args.out}")


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
