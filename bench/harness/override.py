"""A diagnostic run of the free Rigour entrant on another version or config (docs/ENTRANTS.md).

Only with --diagnostic, so it is never scored: checking what a release
candidate's new check would say on the frozen corpus, for example, without
moving the pinned version every scored run uses. The version, the config
file and its sha256 are fixed in run.json.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from bench.adapters.rigour import RigourDeterministic


def add_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rigour-version", help="diagnostic runs: the free Rigour entrant at this exact version")
    parser.add_argument("--rigour-config", type=Path, help="diagnostic runs: the free Rigour entrant with this rigour.yml")


def overridden(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "rigour_version", None) or getattr(args, "rigour_config", None))


def check(args: argparse.Namespace) -> None:
    if overridden(args) and not args.diagnostic:
        raise ValueError("--rigour-version and --rigour-config are for diagnostic runs only (--diagnostic)")
    if args.rigour_config and not args.rigour_config.is_file():
        raise ValueError(f"--rigour-config {args.rigour_config}: no such file")


def apply(args: argparse.Namespace, adapters: list) -> list:
    """The adapters, with the free Rigour entrant on the given version and config."""
    if not overridden(args):
        return adapters
    replaced = RigourDeterministic(args.rigour_version or RigourDeterministic.version,
                                   args.rigour_config.resolve() if args.rigour_config else None)
    return [replaced if a.name == RigourDeterministic.name else a for a in adapters]


def record(args: argparse.Namespace) -> dict | None:
    if not overridden(args):
        return None
    config = args.rigour_config
    return {"version": args.rigour_version or RigourDeterministic.version,
            **({"config": str(config), "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}
               if config else {})}
