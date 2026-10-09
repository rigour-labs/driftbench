"""`bench label prelabel`: model suggestions for the label samples, under one hard dollar cap.

Paid: it needs the maintainer's go, `--max-usd`, and OPENROUTER_API_KEY in the
environment. One invocation covers every `--repo` given, with one cap across
all of them.
"""
from __future__ import annotations

import argparse
import functools
from collections import Counter
from pathlib import Path

from bench.collect.github import GitHubClient
from bench.harness.budget import Budget
from bench.labels.model_file import empty_model_file, model_path, read_model_file, write_model_file
from bench.labels.openrouter import chat
from bench.labels.prelabel import (ENTRANT, MAX_TOKENS, anchored_hunk, guide_excerpt, messages_for, prompt_sha256,
                                   run_prelabels, system_prompt)
from bench.labels.sample import read_sample, sample_path
from bench.labels.store import LabelError
from bench.labels.workspace import points_for_repo
from bench.points.texts import TextSource, point_text

DEFAULT_CALL_BOUND = 0.05
CLAUDE_MARKERS = ("anthropic/", "claude")


def add_prelabel_action(actions: argparse._SubParsersAction, root: Path) -> None:
    pre = actions.add_parser("prelabel", help="model suggestions for the label samples (paid; needs --max-usd)")
    pre.add_argument("--repo", action="append", required=True, help="repeat for each repository")
    pre.add_argument("--model", required=True, help="full OpenRouter model ID, outside the Claude family")
    pre.add_argument("--max-usd", type=float, required=True, help="hard cap across every repo in this run")
    pre.add_argument("--call-bound", type=float, default=DEFAULT_CALL_BOUND,
                     help="upper bound on one call's cost before any has been seen")
    pre.add_argument("--guide", type=Path, default=root / "docs" / "LABELLING.md")
    pre.set_defaults(label_handler=cmd_prelabel)


def check_args(args: argparse.Namespace) -> None:
    if args.max_usd <= 0:
        raise LabelError("--max-usd must be above zero: a paid run needs an approved cap")
    if any(marker in args.model.lower() for marker in CLAUDE_MARKERS):
        raise LabelError(f"{args.model}: pre-labels use a model outside the Claude family; "
                         "both paid entrants run on Claude")


def model_data_for(path: Path, sample: dict, args: argparse.Namespace, system: str) -> dict:
    """The repo's model file, or a new one; refuse to mix models, prompts or samples in one file."""
    fresh = {**empty_model_file(sample, args.model), "prompt_sha256": prompt_sha256(system)}
    data = read_model_file(path, sample["repo"]) or fresh
    for key in ("model", "rules_version", "prompt_version", "prompt_sha256", "blind_ids"):
        if data.get(key) != fresh[key]:
            raise LabelError(f"{path}: its {key} differs from this run's; move the file aside to start over")
    return data


def sample_targets(sample: dict, points: dict, texts: TextSource, client: GitHubClient, system: str) -> list[dict]:
    """One target per point of the label sample, and no others (the calibration sample is never sent)."""
    targets = []
    for pid in sample["point_ids"]:
        point = points[pid]
        text = point_text(texts, point)
        comments = (client.get_all(f"repos/{sample['repo']}/pulls/{point['pr']}/comments")
                    if point["kind"] == "inline" else [])
        hunk = anchored_hunk(comments, point["source_id"])
        targets.append({"id": pid, "text": text, "messages": messages_for(system, text or "", hunk)})
    return targets


def cmd_prelabel(args: argparse.Namespace) -> int:
    check_args(args)
    system = system_prompt(guide_excerpt(args.guide))
    call = functools.partial(chat, args.model, max_tokens=MAX_TOKENS)
    budget = Budget(args.max_usd, {ENTRANT: args.call_bound})
    client = GitHubClient(args.cache)
    for repo in args.repo:
        sample = read_sample(sample_path(args.labels, repo))
        if sample is None:
            raise LabelError(f"no sample for {repo}; run `bench label sample` first")
        path = model_path(args.labels, repo)
        data = model_data_for(path, sample, args, system)
        points = {p["id"]: p for p in points_for_repo(args, repo)["points"]}
        targets = sample_targets(sample, points, TextSource(client, repo), client, system)
        data = run_prelabels(data, targets, call, budget, lambda d: write_model_file(d, path))
        reasons = Counter(why.split(":")[0] for why in data["unsuggested"].values())
        suggested = sum(1 for e in data["points"].values() if e.get("suggested"))
        print(f"{repo}: {suggested} of {len(sample['point_ids'])} suggested, reported spend ${data['spent_usd']:.4f}; "
              f"unsuggested: {dict(reasons) or 'none'} -> {path}")
    print(f"budget: {budget.as_record()}")
    return 0
