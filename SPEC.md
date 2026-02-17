# DriftBench Task Specification

Every task in the DriftBench dataset must follow this schema to ensure reproducibility and deterministic evaluation.

## task.schema.json

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "category": {
      "enum": ["logic_drift", "pattern_drift", "arch_drift", "stale_drift", "security_drift", "standard_drift"]
    },
    "name": { "type": "string" },
    "repository": { "type": "string" },
    "intent": { "type": "string" },
    "base_sha": { "type": "string" },
    "golden_patch": { "type": "string", "description": "Relative path to the correct patch" },
    "drift_candidates": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string" },
          "patch": { "type": "string" },
          "drift_type": { "type": "string" },
          "expected_result": { "enum": ["PASS", "FAIL"] },
          "fail_gate": { "type": "string", "description": "The Rigour gate id expected to catch this" }
        },
        "required": ["id", "patch", "expected_result"]
      }
    },
    "rigour_config": { "type": "string", "default": ".rigour/config.yaml" }
  },
  "required": ["id", "category", "name", "repository", "intent", "base_sha", "golden_patch", "drift_candidates"]
}
```

## Drift Categories Mapping

| Category | Primary Rigour Gate | Evaluation Goal |
| :--- | :--- | :--- |
| **Logic Drift** | `context-drift` | Detect bypass of business logic or variations in intent. |
| **Pattern Drift** | `pattern-index` | Detect reinventing existing code instead of reusing. |
| **Arch Drift** | `ast-analysis` | Detect boundary violations and circular dependencies. |
| **Stale Drift** | `staleness` | Detect use of deprecated/unsafe libraries or patterns. |
| **Security Drift** | `file-guard` | Detect PII leaks, direct DB access, or unsafe modifications. |
| **Standard Drift** | `ast-analysis` | Detect complexity drift or naming inconsistency. |
