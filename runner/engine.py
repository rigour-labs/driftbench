import os
import json
import shutil
import subprocess
import click
from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class DriftCandidate(BaseModel):
    id: str
    patch: str
    drift_type: str
    expected_result: str
    fail_gate: Optional[str] = None


class Task(BaseModel):
    id: str
    category: str
    name: str
    repository: str
    intent: str
    base_sha: str
    golden_patch: str
    drift_candidates: List[DriftCandidate]
    rigour_config: str = ".rigour/config.yaml"

    @staticmethod
    def from_json(path: str) -> 'Task':
        """Load a Task from a JSON file. Stores the source path for resolving relative paths."""
        with open(path, 'r') as f:
            data = json.load(f)
        task = Task(**data)
        # Store the directory containing the task file for resolving relative paths
        task._source_dir = os.path.dirname(os.path.abspath(path))
        return task

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a relative path (patch, config) against the workspace root."""
        if os.path.isabs(relative_path):
            return relative_path
        # If we have a source directory from from_json(), compute workspace root
        # Task files are at: datasets/<category>/task.json
        # Relative paths are from workspace root: datasets/<category>/patches/foo.patch
        if hasattr(self, '_source_dir') and self._source_dir:
            # Go up from datasets/<category>/ to workspace root
            workspace_root = os.path.normpath(os.path.join(self._source_dir, '..', '..'))
            return os.path.normpath(os.path.join(workspace_root, relative_path))
        # Fallback: resolve from current working directory
        return os.path.abspath(relative_path)

class BenchmarkEngine:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.tmp_dir = os.path.join(workspace_root, ".drift_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)
        # Track active repo to avoid redundant setup/teardown cycles
        self._active_repo: Optional[str] = None
        self._active_sha: Optional[str] = None

    def load_task(self, task_path: str) -> Task:
        """Load a task from JSON file."""
        return Task.from_json(task_path)

    def setup_repo(self, repository: str, base_sha: str, force_clean: bool = False) -> str:
        """
        Clone or reuse a repository and checkout the specified SHA.

        Args:
            repository: GitHub repository in 'owner/repo' format
            base_sha: Git SHA or branch name to checkout
            force_clean: If True, always reset to clean state even if already at correct SHA

        Returns:
            Path to the repository directory
        """
        repo_name = repository.split("/")[-1]
        repo_path = os.path.join(self.tmp_dir, repo_name)

        # Optimize: if we're already at the right repo+SHA and don't need clean, return early
        if (not force_clean and
            self._active_repo == repo_path and
            self._active_sha == base_sha and
            os.path.exists(repo_path)):
            click.echo(f"    ⚡ Using already-active repo: {repo_name}@{base_sha[:8]}")
            return repo_path

        # Reuse existing repo if available
        if os.path.exists(repo_path):
            click.echo(f"    ♻️  Reusing cached repo: {repo_name}")
            try:
                subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, capture_output=True)
                subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
                subprocess.run(["git", "fetch", "--all"], cwd=repo_path, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                click.echo(f"    ⚠️  Git cleanup failed, re-cloning: {e}")
                shutil.rmtree(repo_path, ignore_errors=True)
                return self.setup_repo(repository, base_sha, force_clean)
        else:
            click.echo(f"    📥 Cloning {repository}...")
            url = f"https://github.com/{repository}.git"

            try:
                subprocess.run(["git", "clone", "--depth", "1", url, repo_path], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # Full clone as fallback for specific SHA access
                subprocess.run(["git", "clone", url, repo_path], check=True, capture_output=True)

        # Checkout the target SHA
        click.echo(f"    🔀 Checkout {base_sha[:8] if len(base_sha) > 8 else base_sha}")
        try:
            subprocess.run(["git", "fetch", "origin", base_sha], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "checkout", base_sha], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "checkout", base_sha], cwd=repo_path, check=True, capture_output=True)

        # Track active state
        self._active_repo = repo_path
        self._active_sha = base_sha

        return repo_path

    def reset_repo(self, repo_path: str) -> None:
        """Reset repository to clean state without full teardown."""
        try:
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            click.echo(f"    ⚠️  Reset failed: {e}")

    def cleanup_repo(self, repo_path: str) -> None:
        """Remove repository directory and clear active state."""
        try:
            shutil.rmtree(repo_path)
        except Exception as e:
            click.echo(f"    ⚠️  Cleanup failed: {e}")
        finally:
            if self._active_repo == repo_path:
                self._active_repo = None
                self._active_sha = None

    def apply_patch(self, repo_path: str, patch_path: str) -> List[str]:
        """
        Apply a patch file to the repository using multiple strategies.

        Tries in order:
        1. git apply (most reliable for git-style patches)
        2. git apply with --3way (handles missing context)
        3. patch utility with lenient options

        Args:
            repo_path: Path to the repository
            patch_path: Path to the patch file (can be relative to workspace_root)

        Returns:
            List of files modified by the patch

        Raises:
            subprocess.CalledProcessError: If all patch strategies fail
        """
        import re
        full_patch_path = os.path.abspath(patch_path)

        if not os.path.exists(full_patch_path):
            raise FileNotFoundError(f"Patch file not found: {full_patch_path}")

        # Extract files from patch before applying
        modified_files = []
        with open(full_patch_path, 'r') as f:
            patch_content = f.read()

        # Match '--- a/path', '+++ b/path', or '--- /dev/null' patterns
        for match in re.finditer(r'^(?:---|\+\+\+)\s+(?:[ab]/)?(.+?)(?:\s|$)', patch_content, re.MULTILINE):
            file_path = match.group(1)
            if file_path and file_path not in modified_files and file_path != '/dev/null':
                modified_files.append(file_path)

        click.echo(f"    🩹 Applying patch: {os.path.basename(patch_path)}")

        # Strategy 1: git apply (standard)
        try:
            result = subprocess.run(
                ["git", "apply", "--whitespace=nowarn", full_patch_path],
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                return modified_files
        except Exception:
            pass

        # Strategy 2: git apply with --3way (handles missing context by doing 3-way merge)
        try:
            result = subprocess.run(
                ["git", "apply", "--3way", "--whitespace=nowarn", full_patch_path],
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                click.echo(f"    ℹ️  Applied with 3-way merge")
                return modified_files
        except Exception:
            pass

        # Strategy 3: git apply with --reject (apply what we can, create .rej for rest)
        try:
            result = subprocess.run(
                ["git", "apply", "--reject", "--whitespace=nowarn", full_patch_path],
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                click.echo(f"    ℹ️  Applied with possible rejects")
                return modified_files
        except Exception:
            pass

        # Strategy 4: patch utility (fallback)
        try:
            result = subprocess.run(
                ["patch", "-p1", "--ignore-whitespace", "--fuzz=3", "--input", full_patch_path],
                cwd=repo_path, capture_output=True, text=True
            )
            if result.returncode == 0:
                click.echo(f"    ℹ️  Applied with patch utility")
                return modified_files
        except Exception:
            pass

        # Strategy 5: Direct file creation for new file patches
        if "--- /dev/null" in patch_content:
            try:
                return self._apply_new_file_patch(repo_path, patch_content)
            except Exception as e:
                click.secho(f"    ⚠️  New file patch failed: {e}", fg='yellow')

        # All strategies failed
        click.secho(f"    ❌ All patch strategies failed", fg='red')
        raise subprocess.CalledProcessError(1, "patch", b"", b"All patch strategies failed")

    def _apply_new_file_patch(self, repo_path: str, patch_content: str) -> List[str]:
        """
        Manually apply a patch that creates new files.

        This handles patches with '--- /dev/null' that create new files,
        even if they don't have proper context lines.
        """
        import re

        modified_files = []

        # Find all new file hunks
        # Pattern: +++ b/path followed by content
        pattern = r'\+\+\+ b/(.+?)(?:\s|$)[\s\S]*?(?=^---|\Z)'
        matches = list(re.finditer(pattern, patch_content, re.MULTILINE))

        for match in matches:
            file_path = match.group(1).strip()
            full_match = match.group(0)

            # Extract the added lines (lines starting with +, excluding +++)
            lines = []
            for line in full_match.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    lines.append(line[1:])  # Remove the leading +

            if lines and file_path:
                # Create parent directories
                full_path = os.path.join(repo_path, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)

                # Write the file
                with open(full_path, 'w') as f:
                    f.write('\n'.join(lines))

                modified_files.append(file_path)
                click.echo(f"    📝 Created new file: {file_path}")

        if not modified_files:
            raise ValueError("No new files found in patch")

        return modified_files

    def _get_rigour_command(self) -> List[str]:
        """
        Get the rigour command from environment configuration.

        Environment variables:
            RIGOUR_CLI_PATH: Full path to rigour CLI (e.g., /path/to/cli.js or /path/to/rigour)
            RIGOUR_USE_NODE: Set to "true" if RIGOUR_CLI_PATH requires node to execute

        Examples:
            # Local development (node script)
            RIGOUR_CLI_PATH=/path/to/rigour/packages/rigour-cli/dist/cli.js
            RIGOUR_USE_NODE=true

            # Global npm install
            RIGOUR_CLI_PATH=/usr/local/bin/rigour

            # Default: uses 'rigour' from PATH
        """
        cli_path = os.environ.get("RIGOUR_CLI_PATH")
        use_node = os.environ.get("RIGOUR_USE_NODE", "").lower() == "true"

        if cli_path:
            if use_node:
                return ["node", cli_path]
            return [cli_path]

        # Default: assume rigour is in PATH
        return ["rigour"]

    def run_rigour(
        self,
        repo_path: str,
        config_path: str,
        task: Optional[Task] = None,
        target_files: Optional[List[str]] = None
    ) -> Dict:
        """
        Run rigour check against the repository.

        Args:
            repo_path: Path to the repository
            config_path: Path to rigour config (relative paths resolved via task)
            task: Optional Task object for resolving relative config paths
            target_files: Optional list of files to check (for incremental analysis)

        Returns:
            Dict with rigour results including 'status', 'summary', etc.
        """
        # Resolve config path - this is the key fix
        if task and hasattr(task, 'resolve_path'):
            full_config_path = task.resolve_path(config_path)
        elif os.path.isabs(config_path):
            full_config_path = config_path
        else:
            # Fallback: resolve from workspace root (where datasets/ lives)
            full_config_path = os.path.join(self.workspace_root, config_path)

        if not os.path.exists(full_config_path):
            return {
                "status": "ERROR",
                "error": f"Rigour config not found: {full_config_path}"
            }

        click.echo(f"    🔍 Running rigour with config: {os.path.basename(full_config_path)}")

        try:
            rigour_cmd = self._get_rigour_command()
            cmd = rigour_cmd + ["check", "--config", full_config_path, "--json"]

            # If target files specified, only check those (incremental mode)
            if target_files:
                cmd.extend(target_files)
                click.echo(f"    📁 Checking {len(target_files)} modified file(s)")

            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)

            if not result.stdout:
                return {"status": "ERROR", "error": result.stderr or "No output from rigour"}

            return json.loads(result.stdout)
        except json.JSONDecodeError as e:
            return {"status": "ERROR", "error": f"Invalid JSON from rigour: {e}"}
        except FileNotFoundError:
            return {"status": "ERROR", "error": "rigour command not found - is it installed?"}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def evaluate_patch(
        self,
        task: Task,
        patch_path: str,
        candidate_id: str,
        results_dir: str = "results",
        cleanup: bool = False
    ) -> Dict:
        """
        Evaluate a single patch against rigour checks.

        Args:
            task: The benchmark task
            patch_path: Path to the patch file
            candidate_id: Identifier for this evaluation (e.g., 'golden', 'llm_gpt4')
            results_dir: Directory to store results and events
            cleanup: If True, delete repo after evaluation (use False for batch evaluations)

        Returns:
            Dict with evaluation results:
                - candidate_id: The provided identifier
                - detected: True if rigour detected drift/violations
                - report: Full rigour report
                - patch_applied: True if patch was successfully applied
        """
        repo_path = self.setup_repo(task.repository, task.base_sha, force_clean=True)

        # Apply patch and get list of modified files
        patch_applied = False
        modified_files = []
        try:
            modified_files = self.apply_patch(repo_path, patch_path)
            patch_applied = True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return {
                "candidate_id": candidate_id,
                "detected": True,  # Treat patch failure as drift
                "report": {"status": "ERROR", "error": f"Patch failed: {e}"},
                "patch_applied": False
            }

        # Run Rigour only on modified files (incremental analysis)
        # This avoids false positives from pre-existing issues in the repo
        report = self.run_rigour(
            repo_path, task.rigour_config, task=task,
            target_files=modified_files if modified_files else None
        )

        # Determine if drift was detected
        # FAIL = drift detected, PASS = no drift, ERROR = treat as inconclusive (log warning)
        status = report.get("status")
        if status == "ERROR":
            click.secho(f"    ⚠️  Rigour error: {report.get('error', 'Unknown')}", fg='yellow')
            # Check failures array for staleness violations even if status is ERROR
            failures = report.get("failures", [])
            staleness_failures = [f for f in failures if f.get("id", "").startswith("STALENESS_")]
            detected = len(staleness_failures) > 0
        else:
            detected = status == "FAIL"

        # Collect events for Studio Integration
        events_src = os.path.join(repo_path, ".rigour", "events.jsonl")
        if os.path.exists(events_src):
            studio_dir = os.path.join(results_dir, "studio", task.id)
            os.makedirs(studio_dir, exist_ok=True)
            shutil.copy2(events_src, os.path.join(studio_dir, f"{candidate_id}.jsonl"))

        # Only cleanup if explicitly requested (allows batch operations)
        if cleanup:
            self.cleanup_repo(repo_path)
        else:
            # Reset for next evaluation on same repo
            self.reset_repo(repo_path)

        return {
            "candidate_id": candidate_id,
            "detected": detected,
            "report": report,
            "patch_applied": patch_applied
        }

    def evaluate_task(self, task: Task, results_dir: str = "results") -> Dict:
        """
        Evaluate a complete task: golden patch + all drift candidates.

        This validates:
        1. Golden patch should PASS (no drift detected) - true negative
        2. Drift candidates should FAIL (drift detected) - true positive

        Args:
            task: The benchmark task to evaluate
            results_dir: Directory to store results

        Returns:
            Dict with:
                - status: 'VALID' or 'INVALID_TASK'
                - golden_result: Result of golden patch evaluation
                - candidates: List of candidate evaluation results
        """
        click.echo(f"🚀 Evaluating Task: {task.name} ({task.id})")

        os.makedirs(results_dir, exist_ok=True)

        # 1. Validate Golden Patch (True Negative Check)
        click.echo("  🌟 Validating Golden Patch...")
        golden_patch_path = task.resolve_path(task.golden_patch) if hasattr(task, 'resolve_path') else task.golden_patch
        gold_result = self.evaluate_patch(task, golden_patch_path, "golden", results_dir, cleanup=False)

        if gold_result["detected"]:
            click.secho("    ❌ Golden patch triggered false positive!", fg='red', bold=True)
            if self._active_repo:
                self.cleanup_repo(self._active_repo)
            return {
                "status": "INVALID_TASK",
                "golden_result": gold_result,
                "candidates": [],
                "error": "Golden patch should not trigger drift detection"
            }

        click.secho("    ✅ Golden patch passed correctly", fg='green')

        # 2. Evaluate Drift Candidates (True Positive Checks)
        results = []
        for i, candidate in enumerate(task.drift_candidates):
            is_last = (i == len(task.drift_candidates) - 1)
            click.echo(f"  🔍 Checking: {candidate.id} ({candidate.drift_type})")

            candidate_patch_path = task.resolve_path(candidate.patch) if hasattr(task, 'resolve_path') else candidate.patch
            res = self.evaluate_patch(
                task, candidate_patch_path, candidate.id, results_dir,
                cleanup=is_last  # Only cleanup on last candidate
            )

            # Determine if result matches expectation
            expected_fail = candidate.expected_result == "FAIL"
            actual_fail = res["detected"]

            if expected_fail and actual_fail:
                # Check specific gate if specified
                failed_gates = res["report"].get("summary", {})
                gate_caught = candidate.fail_gate is None or failed_gates.get(candidate.fail_gate) == "FAIL"
                if gate_caught:
                    click.secho(f"    ✅ Correctly detected {candidate.drift_type}", fg='green')
                    res["correct"] = True
                else:
                    click.secho(f"    ⚠️  Detected but wrong gate (expected {candidate.fail_gate})", fg='yellow')
                    res["correct"] = False
            elif not expected_fail and not actual_fail:
                click.secho(f"    ✅ Correctly allowed valid change", fg='green')
                res["correct"] = True
            else:
                if expected_fail:
                    click.secho(f"    ❌ Missed {candidate.drift_type} (expected FAIL)", fg='red')
                else:
                    click.secho(f"    ❌ False positive (expected PASS)", fg='red')
                res["correct"] = False

            results.append({
                **res,
                "expected": candidate.expected_result,
                "drift_type": candidate.drift_type
            })

        return {
            "status": "VALID",
            "golden_result": gold_result,
            "candidates": results
        }

@click.command()
@click.argument('task_path')
def main(task_path):
    engine = BenchmarkEngine(os.getcwd())
    if task_path:
        t = engine.load_task(task_path)
        engine.evaluate_task(t)

if __name__ == "__main__":
    main()
