"""`bench run`: every chosen entrant over every frozen corpus."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from bench.adapters import PaidSettings, select_adapters
from bench.adapters.claude_cli import CLAUDE_CODE_VERSION, PROVIDERS
from bench.adapters.rigour import VERSION as RIGOUR_VERSION
from bench.adapters.tool_access import ToolAccessError, check_parity, pinned_rigour_source, tool_record
from bench.harness.budget import Budget, BudgetError
from bench.collect.corpus import CorpusError, read_corpus
from bench.harness.gitrepo import GitError, RepoCheckout
from bench.harness.runner import SMOKE_MIN_CHANGED_LINES, RunConfig, run_corpus
from bench.labels.fingerprint import label_fingerprint
from bench.labels.openrouter import OpenRouterError, key_usage
from bench.repos import slug_of
from bench.subsample import SubsampleError, cap_shares, read_selection, restrict, selection_record


def add_run_parser(commands: argparse._SubParsersAction, root: Path) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run = commands.add_parser("run", help="run entrants on every reviewed head of the frozen corpus")
    run.add_argument("--entrants", nargs="+", default=["free"], help="names, or `free` for every free entrant")
    run.add_argument("--repo", action="append", help="only this repo (repeatable)")
    run.add_argument("--corpus", type=Path, default=root / "work" / "corpus")
    run.add_argument("--out", type=Path, default=root / "work" / "runs" / today)
    run.add_argument("--repos-dir", type=Path, default=root / "work" / "repos", help="local clones")
    run.add_argument("--timeout", type=int, default=900, help="seconds per review")
    run.add_argument("--labels", type=Path, default=root / "labels", help="label files fixed at run start")
    add_paid_args(run)
    run.set_defaults(handler=cmd_run)

    manifest = commands.add_parser("manifest", help="write <out>/run.json once, before a run split across jobs")
    manifest.add_argument("--entrants", nargs="+", default=["free"])
    manifest.add_argument("--out", type=Path, required=True)
    manifest.add_argument("--labels", type=Path, default=root / "labels")
    manifest.add_argument("--timeout", type=int, default=900, help="seconds per review")
    add_paid_args(manifest)
    manifest.set_defaults(handler=cmd_manifest)


def add_paid_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="paid entrants: one model for all, by full ID")
    parser.add_argument("--max-usd", type=float, help="paid entrants: hard dollar cap for this invocation")
    parser.add_argument("--head-bound", action="append", default=[], metavar="ENTRANT=USD",
                        help="paid entrants: per-head upper bound from the run estimate (repeatable)")
    parser.add_argument("--provider", choices=PROVIDERS, default="anthropic",
                        help="paid entrants: one provider for all (openrouter reads OPENROUTER_API_KEY)")
    parser.add_argument("--subsample", type=Path, help="a committed selection file: only its pull requests run")
    parser.add_argument("--max-heads", type=int, default=0,
                        help=f"smoke run: only the first N heads per entrant per repo with at least "
                             f"{SMOKE_MIN_CHANGED_LINES} changed lines; never scored")


def paid_settings(args: argparse.Namespace) -> tuple[PaidSettings | None, Budget | None]:
    """PaidSettings and the Budget, or (None, None) for a free run; ValueError on bad input."""
    if args.max_usd is None:
        return None, None
    bounds = {}
    for item in args.head_bound:
        name, _, usd = item.partition("=")
        try:
            bounds[name] = float(usd)
        except ValueError as exc:
            raise ValueError(f"--head-bound {item!r}: expected ENTRANT=USD") from exc
    return (PaidSettings(model=args.model or "", max_usd=args.max_usd, head_bounds=bounds, provider=args.provider),
            Budget(args.max_usd, bounds))


def cmd_run(args: argparse.Namespace) -> int:
    try:
        paid, budget = paid_settings(args)
        adapters = select_adapters(args.entrants, paid)
        for adapter in (a for a in adapters if a.paid):
            budget.per_head_bound(adapter.name)  # BudgetError now, not mid-run, if a paid entrant has no bound
        if any(a.paid for a in adapters):
            check_parity(pinned_rigour_source(RIGOUR_VERSION, args.repos_dir.parent / "npm-cache"))
        selection = read_selection(args.subsample) if args.subsample else None
        corpora = [restrict(read_corpus(p), selection) for p in sorted(args.corpus.glob("*.json"))]
    except (ValueError, CorpusError, BudgetError, ToolAccessError, SubsampleError) as exc:
        print(exc, file=sys.stderr)
        return 1
    corpora = [c for c in corpora if not args.repo or c["repo"] in args.repo]
    if not corpora:
        print(f"no matching corpus in {args.corpus}; run `bench collect` first", file=sys.stderr)
        return 1
    try:
        started = run_manifest(args.out, adapters, args.labels, paid_record(args, adapters),
                               smoke_record(args), subsample_record(args))["run_started_at"]
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    config = RunConfig(out_dir=args.out, scratch_dir=args.out / "_scratch",
                       npm_cache=args.repos_dir.parent / "npm-cache", timeout_s=args.timeout, run_started_at=started,
                       budget=budget, max_heads=args.max_heads)
    usage_start = gateway_usage(args, adapters)
    for corpus in corpora:
        checkout = RepoCheckout(f"https://github.com/{corpus['repo']}.git", args.repos_dir / slug_of(corpus["repo"]))
        for adapter in adapters:
            try:
                counts = run_corpus(adapter, checkout, corpus, config)
            except GitError as exc:
                print(f"{corpus['repo']}: {exc}", file=sys.stderr)
                return 1
            print(f"{corpus['repo']} {adapter.name}: {counts}")
    if budget is not None:
        name = "+".join(slug_of(c["repo"]) for c in corpora)
        if usage_start is not None:
            write_usage(args.out / f"openrouter-usage-{name}.json", usage_start, gateway_usage(args, adapters))
        (args.out / f"budget-{name}.json").write_text(json.dumps(budget.as_record(), indent=1, sort_keys=True) + "\n",
                                                       encoding="utf-8")
    return 0


def gateway_usage(args: argparse.Namespace, adapters: list) -> dict | None:
    """The OpenRouter key's cumulative usage now, when paid entrants run through it; None otherwise.

    An unreadable usage never stops the run: it is recorded as unavailable, with the reason."""
    if args.provider != "openrouter" or not any(a.paid for a in adapters):
        return None
    reading = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "usage_usd": None}
    try:
        reading["usage_usd"] = key_usage()
    except OpenRouterError as exc:  # the billed figure becomes "unavailable"; the run goes on
        print(f"warning: OpenRouter usage unavailable: {exc}", file=sys.stderr)
        reading["error"] = str(exc)
    return reading


def write_usage(path: Path, start: dict, end: dict | None) -> None:
    path.write_text(json.dumps({"start": start, "end": end}, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def paid_record(args: argparse.Namespace, adapters: list) -> dict | None:
    """What makes the paid comparison fair, fixed in run.json: model, timeout, CLI version, cap, bounds."""
    if not any(a.paid for a in adapters):
        return None
    record = {"model": args.model, "provider": args.provider, "timeout_s": args.timeout,
              "claude_code": CLAUDE_CODE_VERSION, "max_usd": args.max_usd, "head_bounds": args.head_bound,
              "tools": tool_record()}
    if args.subsample and args.max_usd:
        selection = read_selection(args.subsample)
        running = [repo for repo, chosen in selection["repos"].items() if chosen["heads"]]
        record["cap_shares"] = cap_shares(selection, running, args.max_usd)
    return record


def smoke_record(args: argparse.Namespace) -> dict | None:
    """A smoke run's cut, fixed in run.json so it can never be taken for a full run."""
    if not args.max_heads:
        return None
    return {"max_heads": args.max_heads, "min_changed_lines": SMOKE_MIN_CHANGED_LINES,
            "rule": "the first heads in corpus order whose diff has at least min_changed_lines changed lines"}


def subsample_record(args: argparse.Namespace) -> dict | None:
    """The selection a run used: file, hash, seed, and each repo's rule and counts (bench/subsample.py)."""
    return selection_record(args.subsample, read_selection(args.subsample)) if args.subsample else None


def run_manifest(out: Path, adapters: list, labels_dir: Path, paid: dict | None = None,
                 smoke: dict | None = None, subsample: dict | None = None) -> dict:
    """`<out>/run.json`, written once at the first start; a resumed run keeps the first record.

    It fixes the labels by content (bench/labels/fingerprint.py): the HEAD
    commit and the blob hash of every label file at the moment the run began.
    A run split across jobs writes it once (`bench manifest`) and gives every
    job the same file, so nothing is recomputed per job.
    """
    path = out / "run.json"
    if path.exists():
        return read_manifest(path)
    manifest = {
        "run_started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "entrants": {adapter.name: adapter.version for adapter in adapters},
        "labels": label_fingerprint(labels_dir),
        **({"paid": paid} if paid else {}),
        **({"smoke": smoke} if smoke else {}),
        **({"subsample": subsample} if subsample else {}),
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def cmd_manifest(args: argparse.Namespace) -> int:
    try:
        paid, _ = paid_settings(args)
        adapters = select_adapters(args.entrants, paid)
        manifest = run_manifest(args.out, adapters, args.labels, paid_record(args, adapters), smoke_record(args),
                                subsample_record(args))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(manifest, sort_keys=True))
    return 0


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest {path}: {exc}") from exc
    if "run_started_at" not in manifest:
        raise ValueError(f"{path}: no run_started_at")
    return manifest
