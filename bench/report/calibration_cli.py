"""`bench calibrate show | ai | merge`: evidence for the judges, model verdicts, and merging them in.

`show` prints every open entry with its evidence (human and Claude judges
read this). `ai` asks the non-Claude model for a verdict on every entry
(paid: needs --model and --max-usd) and writes `calibration-model.yaml`.
`merge` writes verdicts into `calibration.yaml`: acted-on by consensus with
`calibration-claude.yaml`, location by the model alone, human verdicts kept.
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

import yaml

from bench.collect.corpus import read_corpus
from bench.collect.github import GitHubClient
from bench.harness.budget import Budget
from bench.harness.runner import read_record, record_path_for
from bench.labels.openrouter import chat
from bench.labels.prelabel_cli import CLAUDE_MARKERS
from bench.points.texts import TextSource, point_text
from bench.repos import slug_of
from bench.report.calibration import CalibrationError, read_calibration, write_calibration
from bench.report.calibration_ai import ENTRANT, MAX_TOKENS, entry_key, judge, merge_verdicts
from bench.report.calibration_evidence import acted_evidence, location_evidence


def evidence_for(args: argparse.Namespace, entry: dict, client: GitHubClient, points_of) -> str:
    point = next(p for p in points_of(entry["repo"])["points"] if p["id"] == entry["point"])
    text = point_text(TextSource(client, entry["repo"]), point) or "(text changed since the freeze)"
    if entry["kind"] == "location":
        record = read_record(record_path_for(args.run, entry["tool"], entry["repo"], entry["pr"], entry["head_sha"]))
        return location_evidence(client, entry["repo"], point, text, entry["tool"], record["findings"][entry["finding"]])
    corpus = read_corpus(args.corpus / f"{slug_of(entry['repo'])}.json")
    merged_head = next(pr["head_sha"] for pr in corpus["prs"] if pr["number"] == entry["pr"])
    return acted_evidence(client, entry["repo"], point, text, merged_head)


def read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"cannot read {path}: {exc}") from exc


def cmd_show(args: argparse.Namespace, path: Path, points_of) -> int:
    calibration = read_calibration(path)
    if calibration is None:
        raise CalibrationError(f"no sample at {path}; run `bench calibrate draw` first")
    client = GitHubClient(args.cache)
    for entry in (e for e in calibration["entries"] if e["verdict"] is None):
        print(f"## {entry_key(entry)} ({entry['repo']} #{entry['pr']})\n{evidence_for(args, entry, client, points_of)}\n")
    return 0


def cmd_ai(args: argparse.Namespace, path: Path, points_of) -> int:
    if not args.max_usd or args.max_usd <= 0 or not args.model:
        raise CalibrationError("`calibrate ai` is paid: it needs --model and an approved --max-usd")
    if any(marker in args.model.lower() for marker in CLAUDE_MARKERS):
        raise CalibrationError(f"{args.model}: calibration verdicts use a model outside the Claude family")
    calibration = read_calibration(path)
    if calibration is None:
        raise CalibrationError(f"no sample at {path}; run `bench calibrate draw` first")
    out = path.with_name("calibration-model.yaml")
    data = read_yaml(out) if out.exists() else {"model": args.model, "spent_usd": 0.0, "verdicts": {}, "unjudged": {}}
    if data["model"] != args.model:
        raise CalibrationError(f"{out} was made with {data['model']}; move it aside to change models")
    budget, client = Budget(args.max_usd, {ENTRANT: args.call_bound}), GitHubClient(args.cache)
    call = functools.partial(chat, args.model, max_tokens=MAX_TOKENS)
    for entry in calibration["entries"]:
        key = entry_key(entry)
        if key in data["verdicts"]:
            continue
        result, why = judge(entry, evidence_for(args, entry, client, points_of), call, budget)
        if result is not None:
            data["verdicts"][key] = result
            data["spent_usd"] = round(data["spent_usd"] + result["cost_usd"], 6)
        data["unjudged"] = {**{k: v for k, v in data["unjudged"].items() if k != key}, **({key: why} if why else {})}
        out.write_text(yaml.safe_dump(data, sort_keys=True), encoding="utf-8")
    print(f"{len(data['verdicts'])} judged, {len(data['unjudged'])} not; reported spend ${data['spent_usd']:.4f}; "
          f"budget {budget.as_record()} -> {out}")
    return 0


def cmd_merge(args: argparse.Namespace, path: Path) -> int:
    calibration = read_calibration(path)
    if calibration is None:
        raise CalibrationError(f"no sample at {path}")
    claude = read_yaml(path.with_name("calibration-claude.yaml")) if path.with_name("calibration-claude.yaml").exists() else {}
    model = read_yaml(path.with_name("calibration-model.yaml"))
    merged = merge_verdicts(calibration, claude, model)
    write_calibration(merged, path)
    open_entries = sum(1 for e in merged["entries"] if e["verdict"] is None and not e.get("disputed"))
    disputed = sum(1 for e in merged["entries"] if e.get("disputed"))
    print(f"merged into {path}: {len(merged['entries']) - open_entries - disputed} with a verdict, "
          f"{disputed} disputed, {open_entries} open")
    return 0
