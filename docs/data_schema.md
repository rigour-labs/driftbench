# DriftBench Leaderboard Data Schema

## `data.json` Structure

```json
{
  "generated_at": "2026-02-01T15:45:00Z",
  "version": "1.0",
  "total_tasks": 50,
  "repositories": {
    "lodash": { "language": "javascript", "tasks": 4 },
    "django": { "language": "python", "tasks": 6 },
    "fastapi": { "language": "python", "tasks": 4 },
    "flask": { "language": "python", "tasks": 5 },
    "pydantic": { "language": "python", "tasks": 5 },
    "react": { "language": "javascript", "tasks": 5 },
    "nextjs": { "language": "javascript", "tasks": 6 },
    "shadcn-ui": { "language": "javascript", "tasks": 6 },
    "tanstack-query": { "language": "javascript", "tasks": 4 },
    "openai-python": { "language": "python", "tasks": 4 }
  },
  "categories": {
    "staleness_drift": "Code uses deprecated/legacy patterns",
    "security_drift": "Code introduces security vulnerabilities",
    "architecture_drift": "Code violates architectural boundaries",
    "pattern_drift": "Code deviates from established patterns"
  },
  "leaderboard": [
    {
      "rank": 1,
      "model": "anthropic/claude-opus-4-5-20251101",
      "display_name": "Claude Opus 4.5",
      "provider": "Anthropic",
      "pass_rate": 95.2,
      "drift_detection_rate": 4.8,
      "accuracy": 94.0,
      "tasks_run": 50,
      "breakdown": {
        "by_repo": {
          "lodash": { "passed": 4, "failed": 0, "total": 4 },
          "django": { "passed": 6, "failed": 0, "total": 6 },
          "fastapi": { "passed": 3, "failed": 1, "total": 4 },
          "flask": { "passed": 5, "failed": 0, "total": 5 },
          "pydantic": { "passed": 5, "failed": 0, "total": 5 },
          "react": { "passed": 4, "failed": 1, "total": 5 },
          "nextjs": { "passed": 6, "failed": 0, "total": 6 },
          "shadcn-ui": { "passed": 5, "failed": 1, "total": 6 },
          "tanstack-query": { "passed": 4, "failed": 0, "total": 4 },
          "openai-python": { "passed": 4, "failed": 0, "total": 4 }
        },
        "by_language": {
          "python": { "passed": 23, "failed": 1, "total": 24 },
          "javascript": { "passed": 23, "failed": 2, "total": 26 }
        },
        "by_category": {
          "staleness_drift": { "passed": 12, "failed": 0, "total": 12 },
          "security_drift": { "passed": 15, "failed": 2, "total": 17 },
          "architecture_drift": { "passed": 10, "failed": 1, "total": 11 },
          "pattern_drift": { "passed": 9, "failed": 0, "total": 10 }
        }
      },
      "verified_at": "2026-02-01",
      "status": "verified"
    }
  ]
}
```

## Key Metrics

| Metric | Description |
|--------|-------------|
| `pass_rate` | % of tasks where LLM generated code with NO drift |
| `drift_detection_rate` | % of tasks where drift was detected (lower is better for LLM) |
| `accuracy` | % of tasks matching golden baseline behavior |

## Breakdown Dimensions

1. **by_repo**: Performance per repository (lodash, django, etc.)
2. **by_language**: Aggregated by language (python, javascript)
3. **by_category**: Aggregated by drift type (staleness, security, etc.)

## Usage in rigour-web

```typescript
// Fetch leaderboard data
const data = await fetch('/api/stats/data.json').then(r => r.json());

// Display overall leaderboard
data.leaderboard.map(model => (
  <Row key={model.model}>
    <Cell>{model.rank}</Cell>
    <Cell>{model.display_name}</Cell>
    <Cell>{model.pass_rate}%</Cell>
  </Row>
));

// Filter by repository
const djangoLeaders = data.leaderboard.map(m => ({
  ...m,
  pass_rate: (m.breakdown.by_repo.django.passed / m.breakdown.by_repo.django.total * 100).toFixed(1)
}));

// Filter by language
const pythonLeaders = data.leaderboard.map(m => ({
  ...m,
  pass_rate: (m.breakdown.by_language.python.passed / m.breakdown.by_language.python.total * 100).toFixed(1)
}));
```
