"""`bench report` (the results page) and `bench calibrate draw | show` (the hand-checked sample)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.collect.github import GitHubClient, GitHubError
from bench.harness.runner import RecordError, read_record, record_path_for
from bench.harness.cli import read_manifest
from bench.labels.fingerprint import labels_unchanged
from bench.labels.sample import SampleError, points_sha256, read_sample, sample_path
from bench.labels.workspace import current_texts
from bench.labels.model_agreement import model_agreement
from bench.labels.model_file import model_path, read_model_file
from bench.labels.store import LabelError, effective_labels, labels_path, read_labels
from bench.points.points_file import PointsError, read_points
from bench.points.texts import TextSource, point_text
from bench.repos import slug_of
from bench.report.calibration import CalibrationError, draw, read_calibration, summarise_calibration, write_calibration
from bench.report.classes import per_class
from bench.report.markdown import render
from bench.score.cli import ScoreFileError, read_ledger, read_summary

HANDLED = (OSError, ValueError, CalibrationError, GitHubError, LabelError, PointsError, RecordError, SampleError,
           ScoreFileError)


def add_report_parsers(commands: argparse._SubParsersAction, root: Path) -> None:
    for name, help_text in (("report", "write results/<run>/summary.md from the score summaries"),
                            ("calibrate", "draw or show the hand-checked calibration sample")):
        parser = commands.add_parser(name, help=help_text)
        parser.add_argument("--run", type=Path, required=True, help="a run directory, e.g. work/runs/2026-10-08")
        parser.add_argument("--results", type=Path, help="default: results/<run name>")
        parser.add_argument("--points", type=Path, default=root / "work" / "points")
        parser.add_argument("--labels", type=Path, default=root / "labels")
        parser.add_argument("--cache", type=Path, default=root / "work" / "cache")
        if name == "calibrate":
            parser.add_argument("action", choices=("draw", "show"))
            parser.add_argument("--seed", type=int, default=2026)
            parser.add_argument("--replace", action="store_true", help="redraw over an existing sample")
        parser.set_defaults(handler=guarded(cmd_report if name == "report" else cmd_calibrate))


def guarded(handler):
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except HANDLED as exc:
            print(exc, file=sys.stderr)
            return 1
    return run


def results_dir(args: argparse.Namespace) -> Path:
    return args.results or Path("results") / args.run.name


def repo_points(args: argparse.Namespace, repo: str) -> dict:
    return read_points(args.points / f"{slug_of(repo)}.json")


def withheld_reason(args: argparse.Namespace, repo: str, manifest: dict | None, points_file: dict) -> str:
    """Why results by class can't be published for `repo`, or "" if they can."""
    if manifest is None:
        return "the run directory has no run.json, so the labels it used are unknown"
    sample = read_sample(sample_path(args.labels, repo))
    if sample is None:
        return "no labelled sample for this repository"
    if sample["points_sha256"] != points_sha256(points_file):
        return "the sample was drawn from a different points file than this run scored"
    ok, reason = labels_unchanged(args.labels, manifest.get("labels"), repo)
    return "" if ok else reason


def class_results(args: argparse.Namespace, repo: str, ledger: list[dict],
                  manifest: dict | None) -> tuple[dict, str, dict | None]:
    """Rates by class from the labelled sample and the model-human agreement on it, or ({}, the reason they
    are withheld, None)."""
    points_file = repo_points(args, repo)
    reason = withheld_reason(args, repo, manifest, points_file)
    if reason:
        return {}, reason, None
    sample = read_sample(sample_path(args.labels, repo))
    labels = read_labels(labels_path(args.labels, repo), repo)
    texts = current_texts(TextSource(GitHubClient(args.cache), repo), points_file, labels)
    in_sample = set(sample["point_ids"])
    usable = {pid: label for pid, label in effective_labels(labels, texts).items() if pid in in_sample}
    model_data = read_model_file(model_path(args.labels, repo), repo)
    agreement = model_agreement(labels["points"], usable, model_data, sample["point_ids"]) if model_data else None
    return per_class([row for row in ledger if row["repo"] == repo], usable), "", agreement


def load_manifest(run_dir: Path) -> dict | None:
    path = run_dir / "run.json"
    return read_manifest(path) if path.exists() else None


def cmd_report(args: argparse.Namespace) -> int:
    out = results_dir(args)
    summaries = [read_summary(p) for p in sorted(out.glob("*.json"))]
    if not summaries:
        print(f"no score summaries in {out}; run `bench score` first", file=sys.stderr)
        return 1
    ledger = read_ledger(args.run / "ledger.jsonl")
    manifest = load_manifest(args.run)
    classes, notes, models = {}, {}, {}
    for summary in (s for s in summaries if s["reportable"]):
        repo = summary["repo"]
        classes[repo], notes[repo], models[repo] = class_results(args, repo, ledger, manifest)
    calibration = summarise_calibration(read_calibration(out / "calibration.yaml"))
    page = render(args.run.name, summaries, classes, calibration, notes, models)
    (out / "summary.md").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'summary.md'}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    path = results_dir(args) / "calibration.yaml"
    if args.action == "show":
        return show_calibration(args, path)
    if path.exists() and not args.replace:
        print(f"{path} exists; pass --replace to redraw (the old draw stays in git history)", file=sys.stderr)
        return 1
    reportable = {s["repo"] for s in map(read_summary, sorted(results_dir(args).glob("*.json"))) if s["reportable"]}
    ledger = [row for row in read_ledger(args.run / "ledger.jsonl") if row["repo"] in reportable]
    points = [{**p, "repo": repo} for repo in sorted(reportable)
              for p in repo_points(args, repo)["points"] if not p["dropped"]]
    calibration = draw(ledger, points, args.seed)
    write_calibration(calibration, path)
    print(f"wrote {path} from {len(reportable)} reportable repo(s); {'; '.join(calibration['short']) or 'full'}")
    return 0


def show_calibration(args: argparse.Namespace, path: Path) -> int:
    calibration = read_calibration(path)
    if calibration is None:
        print(f"no sample at {path}; run `bench calibrate draw` first", file=sys.stderr)
        return 1
    client = GitHubClient(args.cache)
    for entry in (e for e in calibration["entries"] if e["verdict"] is None):
        point = next(p for p in repo_points(args, entry["repo"])["points"] if p["id"] == entry["point"])
        print(f"## {entry['kind']} {entry['point']} ({entry['repo']} #{entry['pr']})")
        print(point_text(TextSource(client, entry["repo"]), point))
        if entry["kind"] == "location":
            record = read_record(record_path_for(args.run, entry["tool"], entry["repo"], entry["pr"], entry["head_sha"]))
            finding = record["findings"][entry["finding"]]
            print(f"-> {entry['tool']} at {finding['path']}:{finding['line']}: {finding['message'][:300]}")
        else:
            print(f"-> acted on: {entry['acted_on']} ({entry['basis']}) at {point['anchor']['path']}:{point['anchor']['line']}")
        print()
    return 0
