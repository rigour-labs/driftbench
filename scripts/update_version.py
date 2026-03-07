#!/usr/bin/env python3
"""
Update latest_version.json on HuggingFace after successful training.

This file is read by Rigour's model-manager at startup to auto-download
the latest fine-tuned model (like antivirus signature updates).

Version format: SemVer (MAJOR.MINOR.PATCH)
  - MAJOR: Training data format change, base model change, pipeline architecture
  - MINOR: New training repos, updated dataset, hyperparameter improvements
  - PATCH: Bug fixes, retraining with same data/format

Usage:
    python scripts/update_version.py --version 2.0.0 --changelog "Aligned training prompts with inference, added enterprise repos"
    python scripts/update_version.py --version 2.1.0 --changelog "Added 15 new enterprise repos from auto-discovery"
    python scripts/update_version.py --bump minor --changelog "New repos added"  # auto-increments

Environment variables:
    HF_TOKEN: HuggingFace token with write access to the dataset repo
"""
import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone


def parse_semver(v: str) -> tuple[int, int, int]:
    """Parse a SemVer string into (major, minor, patch)."""
    # Strip leading 'v' if present
    v = v.lstrip("v")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", v)
    if not match:
        raise ValueError(f"Invalid SemVer: '{v}'. Expected format: MAJOR.MINOR.PATCH (e.g., 2.0.0)")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def bump_version(current: str, bump_type: str) -> str:
    """Bump a SemVer string by the given type."""
    major, minor, patch = parse_semver(current)
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Invalid bump type: {bump_type}")


def get_current_version(hf_repo: str, hf_token: str) -> str:
    """Fetch the current version from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(hf_repo, "latest_version.json",
                               repo_type="dataset", token=hf_token)
        with open(path) as f:
            data = json.load(f)
        v = data.get("version", "1.0.0")
        # Handle legacy integer versions
        if isinstance(v, int):
            return f"{v}.0.0"
        return str(v)
    except Exception:
        return "1.0.0"


def main():
    parser = argparse.ArgumentParser(description="Update latest model version on HuggingFace (SemVer)")
    version_group = parser.add_mutually_exclusive_group(required=True)
    version_group.add_argument("--version", type=str, help="Explicit SemVer (e.g., 2.0.0)")
    version_group.add_argument("--bump", choices=["major", "minor", "patch"],
                               help="Auto-bump from current version")
    parser.add_argument("--changelog", default="", help="What changed in this version")
    parser.add_argument("--hf-repo", default="rigour-labs/rigour-rlaif-data", help="HuggingFace dataset repo")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without doing it")
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: HF_TOKEN required")
        sys.exit(1)

    # Resolve version
    if args.version:
        version = args.version.lstrip("v")
        parse_semver(version)  # Validate
    else:
        current = get_current_version(args.hf_repo, args.hf_token)
        version = bump_version(current, args.bump)
        print(f"Current: {current} → Bumping {args.bump} → {version}")

    data = {
        "version": version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": "training-pipeline",
        "changelog": args.changelog,
    }

    if args.dry_run:
        print(f"Would upload to {args.hf_repo}/latest_version.json:")
        print(json.dumps(data, indent=2))
        sys.exit(0)

    from huggingface_hub import HfApi

    api = HfApi(token=args.hf_token)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp, indent=2)
    tmp.close()

    api.upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="latest_version.json",
        repo_id=args.hf_repo,
        repo_type="dataset",
    )
    print(f"Updated latest_version.json → v{version}")
    if args.changelog:
        print(f"  Changelog: {args.changelog}")
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()
