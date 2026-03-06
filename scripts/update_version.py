#!/usr/bin/env python3
"""
Update latest_version.json on HuggingFace after successful training.

This file is read by Rigour's model-manager at startup to auto-download
the latest fine-tuned model (like antivirus signature updates).

Usage:
    python scripts/update_version.py --version 6
    python scripts/update_version.py --version 6 --hf-repo rigour-labs/rigour-rlaif-data

Environment variables:
    HF_TOKEN: HuggingFace token with write access to the dataset repo
"""
import argparse
import json
import os
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Update latest model version on HuggingFace")
    parser.add_argument("--version", required=True, type=int, help="Model version number")
    parser.add_argument("--hf-repo", default="rigour-labs/rigour-rlaif-data", help="HuggingFace dataset repo")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""), help="HuggingFace token")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be uploaded without doing it")
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: HF_TOKEN required")
        sys.exit(1)

    data = {"version": args.version, "updated_by": "training-pipeline"}

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
    print(f"Updated latest_version.json → v{args.version}")
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()
