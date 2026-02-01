# Contributing to DriftBench

Thank you for your interest in contributing to DriftBench! This document provides guidelines and instructions for contributing.

## Ways to Contribute

### 1. Add New Benchmark Tasks

The most valuable contribution is adding new benchmark tasks that test AI code generation for drift.

#### Task Requirements

- **Realistic**: Based on real-world scenarios developers encounter
- **Clear Intent**: The task description should be unambiguous
- **Testable**: Must be verifiable with Rigour's drift detection
- **Documented**: Include golden patch (correct) and drift examples

#### Creating a New Task

1. **Choose a Repository**

   Select from existing repositories or propose a new one:
   - `lodash` (JavaScript utility library)
   - `django` (Python web framework)
   - `flask` (Python micro framework)
   - `react` (JavaScript UI library)
   - `fastapi` (Python API framework)
   - `pydantic` (Python data validation)
   - `shadcn-ui` (React component library)
   - `tanstack-query` (Data fetching library)

2. **Create Task Directory**

   ```bash
   mkdir -p datasets/<repo>/patches
   ```

3. **Write Task JSON**

   Create `datasets/<repo>/<task_name>.json`:

   ```json
   {
       "id": "<repo>-<category>-<number>",
       "category": "stale_drift|security_drift|architecture_drift|pattern_drift|logic_drift",
       "name": "Human-readable task name",
       "repository": "owner/repo",
       "intent": "Detailed description of what the AI should implement...",
       "base_sha": "main",
       "golden_patch": "datasets/<repo>/patches/<task>_gold.patch",
       "rigour_config": "datasets/<repo>/rigour_config.yaml",
       "drift_candidates": [
           {
               "id": "drift_variant_id",
               "patch": "datasets/<repo>/patches/<task>_drift.patch",
               "drift_type": "description of the drift",
               "expected_result": "FAIL"
           }
       ]
   }
   ```

4. **Create Golden Patch**

   The golden patch should be the correct implementation:

   ```diff
   --- a/src/module.js
   +++ b/src/module.js
   @@ -10,0 +11,5 @@
   +const newFunction = () => {
   +    // Correct implementation using modern patterns
   +    return result;
   +};
   ```

5. **Create Drift Patch**

   The drift patch demonstrates incorrect implementation:

   ```diff
   --- a/src/module.js
   +++ b/src/module.js
   @@ -10,0 +11,5 @@
   +var newFunction = function() {  // Uses var instead of const
   +    // Incorrect implementation
   +    return result;
   +};
   ```

6. **Configure Rigour**

   Add or update `datasets/<repo>/rigour_config.yaml`:

   ```yaml
   version: "1.0"
   gates:
     ast-analysis:
       enabled: true
       staleness:
         rules:
           - no-var
           - no-commonjs
   ```

7. **Test Your Task**

   ```bash
   # Verify golden patch passes
   python -m runner.engine datasets/<repo>/<task>.json

   # Run with a model
   python -m runner.harness --model anthropic/claude-opus-4-5-20251101 --task <task-id>
   ```

### 2. Add New Repositories

To add support for a new repository:

1. **Add Repository Metadata**

   Update `scripts/snapshot_leaderboard.py`:

   ```python
   REPO_METADATA = {
       ...
       "new-repo": {"language": "python|javascript", "full_name": "owner/repo"},
   }
   ```

2. **Create Rigour Config**

   Create `datasets/<repo>/rigour_config.yaml` with appropriate rules.

3. **Add Initial Tasks**

   Include at least 3 tasks covering different drift categories.

### 3. Improve Drift Detection

Contribute to the Rigour project to improve drift detection:

- Add new staleness rules (deprecated APIs, legacy patterns)
- Add security vulnerability detection
- Improve pattern matching

### 4. Documentation

- Improve README and guides
- Add examples and tutorials
- Document edge cases and gotchas

## Pull Request Process

1. **Fork the Repository**

   ```bash
   git clone https://github.com/rigour-labs/driftbench.git
   cd driftbench
   ```

2. **Create a Branch**

   ```bash
   git checkout -b feat/add-react-pattern-task
   ```

3. **Make Changes**

   - Follow existing code style
   - Add tests if applicable
   - Update documentation

4. **Test Your Changes**

   ```bash
   # Run the task locally
   python -m runner.harness --model anthropic/claude-opus-4-5-20251101 --task <your-task-id>

   # Verify leaderboard generation
   python scripts/snapshot_leaderboard.py
   ```

5. **Submit PR**

   - Use a clear, descriptive title
   - Reference any related issues
   - Include test results

## Code Style

### Python

- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings

### JSON/YAML

- Use 4-space indentation
- Include comments for complex configurations

### Patches

- Use standard unified diff format
- Include context lines (3 lines default)
- Use `a/` and `b/` prefixes for paths

## Task ID Convention

Use the format: `<repo>-<category>-<number>`

Examples:
- `lodash-stale-001`
- `django-security-003`
- `react-pattern-002`

## Categories

| Category | Prefix | Description |
|----------|--------|-------------|
| Staleness | `stale` | Legacy/deprecated patterns |
| Security | `security` | Security vulnerabilities |
| Architecture | `arch` | Structural violations |
| Pattern | `pattern` | Pattern deviations |
| Logic | `logic` | Logical inconsistencies |

## Questions?

- Open an issue for discussion
- Join our Discord (coming soon)
- Email: contributors@rigour.run

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
