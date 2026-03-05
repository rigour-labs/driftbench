#!/usr/bin/env python3
"""Print leaderboard summary. Used by leaderboard-pipeline.yml."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'results/leaderboard.json'
try:
    with open(path) as f:
        data = json.load(f)
    print(f'Generated: {data["generated_at"]}')
    print(f'Scenarios: {data.get("total_scenarios", "?")}')
    print()
    for m in data.get('models', []):
        acc = m.get('overall_accuracy', 0)
        oc = m.get('overconfident_errors', 0)
        ece = m.get('calibration_ece', 0)
        print(f'  #{m["rank"]} {m["display_name"]}: {acc:.1%} (overconfident: {oc}, ECE: {ece:.3f})')
        for cat, d in sorted(m.get('by_category', {}).items()):
            print(f'      {cat}: {d["accuracy"]:.1%} ({d["correct"]}/{d["total"]})')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
