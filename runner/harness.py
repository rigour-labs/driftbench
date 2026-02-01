import os
import re
import glob
import json
from typing import Dict, Optional

import click
import litellm
from dotenv import load_dotenv

from runner.engine import BenchmarkEngine, Task

load_dotenv(override=True)

SYSTEM_PROMPT = """You are an expert software engineer. 
Your goal is to implement the requested change in the provided codebase.
Follow the project's existing patterns, security standards, and architectural style.

CRITICAL INSTRUCTIONS:
1. Return your answer ONLY as a standard git unified diff (patch).
2. Start the response immediately with '--- ' or '+++ '.
3. Do NOT include any conversational text, markdown formatting (no ```diff), or explanations.
4. Ensure the context lines correctly match the provided source.
5. Use standard headers: '--- a/path/to/file' and '+++ b/path/to/file'.
"""

USER_PROMPT_TEMPLATE = """
Project Repository: {repository}
Goal (Intent): {intent}

Below is the context of the relevant files for this task. 
Please provide a git patch that implements the intent.

{context}
"""

class LLMHarness:
    """
    Harness for benchmarking LLM code generation against drift detection.

    The harness:
    1. Takes a task (intent + repository context)
    2. Prompts an LLM to generate a patch
    3. Evaluates if the patch triggers drift detection
    4. Compares against golden patch baseline
    """

    # Provider aliases for LiteLLM
    PROVIDER_MAP = {
        "openai": "openai",
        "anthropic": "anthropic",
        "gemini": "vertex_ai",
        "google": "vertex_ai",
        "vertex": "vertex_ai",
        "azure": "azure"
    }

    def __init__(self, model: str, workspace_root: Optional[str] = None):
        """
        Initialize the LLM harness.

        Args:
            model: Model identifier (e.g., 'anthropic/claude-3-opus', 'openai/gpt-4')
            workspace_root: Root directory for benchmark operations (defaults to CWD)
        """
        self.model = model
        self.workspace_root = workspace_root or os.getcwd()
        self.engine = BenchmarkEngine(self.workspace_root)
        self.config = {}

        self._load_model_config()
        self._register_model()

    def _load_model_config(self) -> None:
        """Load model configuration from model_config.json."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "model_config.json")

        if not os.path.exists(config_path):
            click.echo("    ⚠️  model_config.json not found, using defaults")
            return

        try:
            with open(config_path, 'r') as f:
                full_config = json.load(f)
                self.config = full_config.get("model_config", {})
            click.echo(f"    ✅ Loaded config for {len(self.config)} models")
        except json.JSONDecodeError as e:
            click.secho(f"    ❌ Invalid model_config.json: {e}", fg='red')
        except Exception as e:
            click.secho(f"    ❌ Failed to load config: {e}", fg='red')

    def _register_model(self) -> None:
        """
        Configure LiteLLM settings for the model.

        Note: LiteLLM auto-detects most providers from model names.
        This method sets up any custom configurations needed.
        """
        # LiteLLM handles model routing automatically based on model name prefix
        # e.g., "anthropic/claude-sonnet-4" -> Anthropic API
        # e.g., "openai/gpt-4o" -> OpenAI API
        # Custom config is used in generate_patch() to override model names if needed
        pass


    def get_repo_context(self, repo_path: str, max_chars: int = 80000) -> str:
        """
        Collect repository context for LLM prompt.

        Args:
            repo_path: Path to the repository
            max_chars: Maximum characters to collect (~20k tokens at 4 chars/token)

        Returns:
            Formatted string with file contents
        """
        context = []
        total_chars = 0

        # Directories to skip
        skip_dirs = {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build', '.tox', '.mypy_cache'}
        # Extensions to skip
        skip_ext = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.pdf', '.zip', '.lock', '.woff', '.ttf'}

        for root, dirs, files in os.walk(repo_path):
            if total_chars >= max_chars:
                break

            # Filter directories in-place
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in skip_dirs]

            for filename in files:
                if total_chars >= max_chars:
                    break

                # Skip binary/non-code files
                if any(filename.endswith(ext) for ext in skip_ext):
                    continue
                if filename.endswith('-lock.json') or filename == 'package-lock.json':
                    continue

                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        content = f.read(2000)  # First 2000 chars per file
                    rel_path = os.path.relpath(filepath, repo_path)
                    entry = f"--- {rel_path} ---\n{content}\n"
                    context.append(entry)
                    total_chars += len(entry)
                except (IOError, OSError):
                    continue

        return "\n".join(context)

    def generate_patch(self, task: Task) -> str:
        """
        Generate a patch using the LLM for the given task.

        Args:
            task: The benchmark task with intent and repository info

        Returns:
            Extracted patch content from LLM response

        Raises:
            Exception: If LLM call fails
        """
        # Setup repo and collect context (repo stays active for later evaluation)
        repo_path = self.engine.setup_repo(task.repository, task.base_sha)
        context = self.get_repo_context(repo_path)

        prompt = USER_PROMPT_TEMPLATE.format(
            repository=task.repository,
            intent=task.intent,
            context=context
        )

        click.echo(f"    🤖 Prompting {self.model}...")

        # Get model configuration
        conf = self.config.get(self.model, {})
        mode = conf.get("mode", "chat")

        # LiteLLM expects model in format "provider/model" which we already have
        # e.g., "anthropic/claude-sonnet-4" works directly with litellm
        effective_model = self.model

        # Build LiteLLM kwargs
        kwargs = {"model": effective_model, "temperature": 0}
        if conf.get("max_tokens"):
            kwargs["max_tokens"] = conf["max_tokens"]

        # For Anthropic, explicitly pass the API key to avoid env var conflicts
        # LiteLLM looks for ANTHROPIC_API_KEY in environment
        if effective_model.startswith("anthropic/"):
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                kwargs["api_key"] = api_key

        if mode == "completion":
            kwargs["prompt"] = f"{SYSTEM_PROMPT}\n\n{prompt}"
        else:
            kwargs["messages"] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]

        try:
            response = litellm.completion(**kwargs)
            raw_response = response.choices[0].message.content
            return self._extract_patch(raw_response)
        except Exception as e:
            click.secho(f"    ❌ LLM Error: {e}", fg='red')
            raise

    def _extract_patch(self, response_text: str) -> str:
        """
        Extract patch content from LLM response and normalize format.

        Handles multiple formats:
        1. Markdown code blocks (```diff ... ```)
        2. Raw unified diff format
        3. Plain text with diff markers

        Also normalizes:
        - Ensures paths have a/ and b/ prefixes for git apply compatibility

        Args:
            response_text: Raw LLM response

        Returns:
            Extracted and normalized patch content
        """
        patch = None

        # Strategy 1: Markdown code block
        markdown_match = re.search(r"```(?:diff|patch)?\s*([\s\S]*?)\s*```", response_text)
        if markdown_match:
            candidate = markdown_match.group(1).strip()
            if "--- " in candidate or "+++ " in candidate:
                patch = candidate

        # Strategy 2: Find diff headers and capture everything after
        if not patch:
            diff_pattern = r"(---\s+(?:a/)?\S+.*?[\r\n]+\+\+\+\s+(?:b/)?\S+.*?[\r\n]+(?:@@.*@@[\s\S]*?)?)(?=\n---\s|\Z)"
            matches = re.findall(diff_pattern, response_text, re.MULTILINE)
            if matches:
                patch = "\n".join(matches).strip()

        # Strategy 3: Find first line starting with --- or +++
        if not patch:
            lines = response_text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("--- ") or line.startswith("+++ "):
                    patch = "\n".join(lines[i:]).strip()
                    break

        # Fallback: return as-is
        if not patch:
            patch = response_text.strip()

        # Normalize: ensure a/ and b/ prefixes for git compatibility
        patch = self._normalize_patch_paths(patch)

        return patch

    def _normalize_patch_paths(self, patch: str) -> str:
        """
        Normalize patch paths to ensure a/ and b/ prefixes.

        Git apply and patch -p1 expect paths like:
        --- a/path/to/file
        +++ b/path/to/file

        This handles cases where LLMs output:
        --- path/to/file
        +++ path/to/file

        Args:
            patch: Raw patch content

        Returns:
            Patch with normalized paths
        """
        lines = patch.split('\n')
        normalized = []

        for line in lines:
            # Handle --- lines (but not /dev/null)
            if line.startswith('--- ') and not line.startswith('--- a/') and '/dev/null' not in line:
                path = line[4:].strip()
                # Remove any leading slashes
                path = path.lstrip('/')
                normalized.append(f'--- a/{path}')
            # Handle +++ lines (but not /dev/null)
            elif line.startswith('+++ ') and not line.startswith('+++ b/') and '/dev/null' not in line:
                path = line[4:].strip()
                path = path.lstrip('/')
                normalized.append(f'+++ b/{path}')
            else:
                normalized.append(line)

        return '\n'.join(normalized)

    def run_task(self, task: Task) -> Dict:
        """
        Run complete benchmark for a task.

        This:
        1. Generates a patch using the LLM
        2. Validates golden patch passes (baseline check)
        3. Evaluates the LLM patch for drift
        4. Saves results

        Args:
            task: The benchmark task

        Returns:
            Dict with evaluation results including:
                - llm_result: LLM patch evaluation
                - golden_result: Golden patch baseline
                - passed: True if LLM patch has no drift
                - correct: True if result matches golden baseline
        """
        click.echo(f"🚀 Benchmarking {self.model} on: {task.id}")

        model_slug = self.model.replace("/", "_")
        results_dir = os.path.join("results", model_slug)
        patch_dir = os.path.join(results_dir, "patches", task.id)
        os.makedirs(patch_dir, exist_ok=True)

        # Step 1: Generate patch from LLM
        try:
            patch_content = self.generate_patch(task)
        except Exception as e:
            return self._save_error_result(task, results_dir, f"LLM generation failed: {e}")

        # Validate patch content
        if not patch_content or not ("--- " in patch_content or "+++ " in patch_content):
            return self._save_error_result(task, results_dir, "LLM returned invalid patch format")

        # Save generated patch
        patch_path = os.path.join(patch_dir, "llm_generated.patch")
        with open(patch_path, 'w') as f:
            f.write(patch_content)

        # Step 2: Establish baseline with golden patch
        click.echo("    🌟 Checking golden patch baseline...")
        golden_patch_path = task.resolve_path(task.golden_patch) if hasattr(task, 'resolve_path') else task.golden_patch
        golden_result = self.engine.evaluate_patch(
            task, golden_patch_path, "golden", results_dir, cleanup=False
        )

        if golden_result["detected"]:
            click.secho("    ⚠️  Warning: Golden patch has drift (task may be invalid)", fg='yellow')

        # Step 3: Evaluate LLM-generated patch
        click.echo("    🔬 Evaluating LLM patch...")
        llm_result = self.engine.evaluate_patch(
            task, patch_path, f"llm_{model_slug}", results_dir, cleanup=True
        )

        # Determine outcome
        passed = not llm_result["detected"]
        # "Correct" means LLM behaves like golden (both pass or both fail)
        correct = llm_result["detected"] == golden_result["detected"]

        if passed:
            click.secho("    🟢 NO DRIFT (PASSED)", fg='green', bold=True)
        else:
            click.secho("    🔴 DRIFT DETECTED", fg='red', bold=True)

        if not correct:
            click.secho("    ⚠️  Result differs from golden baseline", fg='yellow')

        # Build and save report
        report = {
            "task_id": task.id,
            "model": self.model,
            "repository": task.repository,
            "category": task.category,
            "llm_result": llm_result,
            "golden_result": golden_result,
            "passed": passed,
            "correct": correct,
            "patch_path": patch_path
        }

        report_path = os.path.join(results_dir, f"{task.id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def _save_error_result(self, task: Task, results_dir: str, error: str) -> Dict:
        """Save and return an error result."""
        click.secho(f"    ❌ {error}", fg='red')
        report = {
            "task_id": task.id,
            "model": self.model,
            "repository": task.repository,
            "category": task.category,
            "error": error,
            "passed": False,
            "correct": False
        }
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, f"{task.id}.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        return report

@click.command()
@click.option('--model', default='claude-3-5-sonnet', help='Model to benchmark (e.g., anthropic/claude-3-opus, openai/gpt-4)')
@click.option('--task', 'task_id', help='Specific task ID to run')
@click.option('--all', 'run_all', is_flag=True, help='Run all tasks (expensive)')
def main(model: str, task_id: Optional[str], run_all: bool):
    """DriftBench LLM Harness - Benchmark LLM code generation for drift."""
    harness = LLMHarness(model)

    if task_id:
        # Find and run specific task
        task_file = _find_task_file(task_id)
        if task_file:
            task = Task.from_json(task_file)
            harness.run_task(task)
        else:
            click.secho(f"Task not found: {task_id}", fg='red')
            raise SystemExit(1)
    elif run_all:
        # Run all tasks
        task_files = glob.glob("datasets/**/*.json", recursive=True)
        click.echo(f"Running {len(task_files)} tasks...")
        for f in task_files:
            task = Task.from_json(f)
            harness.run_task(task)
    else:
        click.echo("Specify --task <id> or --all")
        raise SystemExit(1)


def _find_task_file(task_id: str) -> Optional[str]:
    """Find task file by ID."""
    for f in glob.glob("datasets/**/*.json", recursive=True):
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                if data.get("id") == task_id:
                    return f
        except (json.JSONDecodeError, IOError):
            continue
    return None


if __name__ == "__main__":
    main()
