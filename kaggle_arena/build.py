"""Build the Kaggle kernel: run.py with its parameters filled in, and its metadata.

    python kaggle_arena/build.py --out /tmp/kernel

Environment: KAGGLE_USERNAME (the kernel's owner), MODELS (comma-separated
names, empty for all), PER_REPO, REPOS_OVERRIDE (comma-separated, empty for the
defaults), RIGOUR_REF, DRIFTBENCH_REF.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SLUG = "rigour-arena-gpu"


def params(env: dict) -> dict:
    base = json.loads((HERE / "params.json").read_text())
    wanted = [m.strip() for m in env.get("MODELS", "").split(",") if m.strip()]
    unknown = set(wanted) - {m["name"] for m in base["models"]}
    if unknown:
        raise SystemExit(f"unknown model(s): {sorted(unknown)}")
    if wanted:
        base["models"] = [m for m in base["models"] if m["name"] in wanted]
    if env.get("REPOS_OVERRIDE"):
        base["repos"] = [r.strip() for r in env["REPOS_OVERRIDE"].split(",") if r.strip()]
    base["per_repo"] = int(env.get("PER_REPO") or base["per_repo"])
    base["rigour_ref"] = env.get("RIGOUR_REF") or base["rigour_ref"]
    base["driftbench_ref"] = env.get("DRIFTBENCH_REF") or base["driftbench_ref"]
    return base


def build(out: Path, env: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    source = (HERE / "run.py").read_text()
    if "__PARAMS__" not in source:
        raise SystemExit("run.py lost its __PARAMS__ placeholder")
    (out / "run.py").write_text(source.replace("__PARAMS__", json.dumps(params(env)), 1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{env['KAGGLE_USERNAME']}/{SLUG}",
        "title": "Rigour Arena GPU",
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }, indent=1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kaggle GPU arena kernel")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = build(args.out, dict(os.environ))
    print(f"kernel -> {out}")


if __name__ == "__main__":
    main()
