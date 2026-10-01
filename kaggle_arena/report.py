"""Per-model pre-emption and Rigour's funnel from a Kaggle GPU arena run.

    python kaggle_arena/report.py kaggle-output/results

Expects <results>/<model>/<repo>/pre-pr/<tool>.json, as run.py writes them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from arena import preemption


def summarize(results: Path) -> list[dict]:
    rows = []
    for model in sorted(p for p in results.iterdir() if p.is_dir()):
        merged: dict = {"prs": {}}
        for path in sorted(model.glob("*/pre-pr/*.json")):
            data = json.loads(path.read_text())
            merged["prs"].update({f"{path.parent.parent.name}#{k}": v for k, v in data["prs"].items()})
        entries = list(merged["prs"].values())
        ok = [e for e in entries if e["status"] in ("PASS", "FAIL")]
        funnel = {k: sum((e.get("deep") or {}).get(k) or 0 for e in ok)
                  for k in ("findings_proposed", "findings_withdrawn", "findings_count", "chunks_failed")}
        score = preemption.score(merged) if ok else None
        rows.append({
            "model": model.name, "prs": len(entries), "errors": len(entries) - len(ok),
            "targets": score.targets if score else 0, "raised": score.preempted if score else 0,
            "minutes_per_pr": round(sum(e["seconds"] for e in ok) / 60 / len(ok), 1) if ok else None,
            **funnel,
        })
    return rows


def main() -> None:
    results = Path(sys.argv[1])
    if not results.exists():
        print(f"no results at {results}")
        return
    print(f"{'model':<18}{'PRs':>5}{'err':>5}{'targets':>9}{'raised':>8}{'min/PR':>8}   funnel: proposed -> withdrawn -> kept (failed calls)")
    for r in summarize(results):
        print(f"{r['model']:<18}{r['prs']:>5}{r['errors']:>5}{r['targets']:>9}{r['raised']:>8}{str(r['minutes_per_pr']):>8}   "
              f"{r['findings_proposed']} -> {r['findings_withdrawn']} -> {r['findings_count']} ({r['chunks_failed']})")


if __name__ == "__main__":
    main()
