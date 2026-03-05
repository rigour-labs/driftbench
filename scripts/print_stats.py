#!/usr/bin/env python3
"""Print training data category stats. Used by training-pipeline.yml."""
import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'rlaif/data/category_stats.json'
try:
    with open(path) as f:
        stats = json.load(f)
    print(f'Verified:   {stats["total_verified"]}')
    print(f'Dropped:    {stats["total_dropped"]}')
    retry_total = sum(s.get('retry_verified', 0) for s in stats.get('categories', {}).values())
    if retry_total > 0:
        print(f'Pass@2 recovered: {retry_total}')
    print()
    print('Per-category breakdown:')
    for cat, s in sorted(stats['categories'].items(), key=lambda x: x[1]['total'], reverse=True):
        print(f'  {cat:30s}  verified={s["verified"]:4d}  dropped={s["dropped"]:4d}  rate={s["verification_rate"]}%')
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
