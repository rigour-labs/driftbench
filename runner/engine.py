import os
import json
import subprocess
import click
from typing import List, Dict
from pydantic import BaseModel, Field

class DriftCandidate(BaseModel):
    id: str
    patch: str
    drift_type: str
    expected_result: str
    fail_gate: str = None

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

class BenchmarkEngine:
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.tmp_dir = os.path.join(workspace_root, ".drift_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def load_task(self, task_path: str) -> Task:
        with open(task_path, 'r') as f:
            data = json.load(f)
        return Task(**data)

    def setup_repo(self, repository: str, base_sha: str) -> str:
        repo_name = repository.split("/")[-1]
        repo_path = os.path.join(self.tmp_dir, repo_name)
        
        if not os.path.exists(repo_path):
            click.echo(f"    📥 Cloning {repository}...")
            url = f"https://github.com/{repository}.git"
            subprocess.run(["git", "clone", url, repo_path], check=True, capture_output=True)
        
        click.echo(f"    git checkout {base_sha}")
        try:
            subprocess.run(["git", "checkout", base_sha], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback for common branch names
            if base_sha == "main":
                subprocess.run(["git", "checkout", "master"], cwd=repo_path, check=True, capture_output=True)
            elif base_sha == "master":
                subprocess.run(["git", "checkout", "main"], cwd=repo_path, check=True, capture_output=True)
            else:
                raise
        # Ensure clean state
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
        return repo_path

    def apply_patch(self, repo_path: str, patch_path: str):
        full_patch_path = os.path.abspath(patch_path)
        click.echo(f"    🩹 Applying patch: {patch_path}")
        # Verbose output for debugging
        subprocess.run(["git", "apply", "--verbose", full_patch_path], cwd=repo_path, check=True)

    def run_rigour(self, repo_path: str, config_path: str) -> Dict:
        """Runs rigour check and returns the results."""
        try:
            full_config_path = os.path.abspath(config_path)
            
            # Use the new --config flag which we just implemented
            cmd = ["rigour", "check", "--config", full_config_path, "--json"]
            result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
            
            if not result.stdout:
                return {"status": "ERROR", "error": result.stderr or "No output from rigour"}
                
            return json.loads(result.stdout)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def evaluate_task(self, task: Task):
        click.echo(f"🚀 Evaluating Task: {task.name} ({task.id})")
        
        repo_path = self.setup_repo(task.repository, task.base_sha)
        results = []

        # 1. Validate Golden Patch (The "Null" Case)
        click.echo("  🌟 Validating Golden Patch (True Negative Check)...")
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
        self.apply_patch(repo_path, task.golden_patch)
        gold_report = self.run_rigour(repo_path, task.rigour_config)
        
        if gold_report.get("status") == "PASS":
            click.secho("    ✅ Golden patch passed correctly.", fg='green')
        else:
            click.secho("    ❌ FAILED: Golden patch triggered a false positive!", fg='red', bold=True)
            click.echo(f"    Full Report: {json.dumps(gold_report, indent=2)}")
            return {"status": "INVALID_TASK", "errors": gold_report.get("failures")}

        # 2. Evaluate Drift Candidates (The "Drift" Cases)
        for candidate in task.drift_candidates:
            click.echo(f"  🔍 Checking Candidate: {candidate.id} ({candidate.drift_type})")
            
            # Reset and apply candidate patch
            subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
            self.apply_patch(repo_path, candidate.patch)
            
            # Run Rigour
            report = self.run_rigour(repo_path, task.rigour_config)
            
            # Evaluation logic
            detected = report.get("status") == "FAIL"
            if detected:
                # Check if the specific gate failed if specified
                failed_gates = report.get("summary", {})
                gate_caught = candidate.fail_gate is None or failed_gates.get(candidate.fail_gate) == "FAIL"
                
                if gate_caught:
                    click.secho(f"    ✅ CORRECTly detected {candidate.drift_type}", fg='green')
                else:
                    click.secho(f"    ❌ FAILED (Wrong Gate): Expected {candidate.fail_gate} to catch this", fg='yellow')
            else:
                if candidate.expected_result == "FAIL":
                    click.secho(f"    ❌ FAILED to detect {candidate.drift_type} (Status: PASS)", fg='red')
                else:
                    click.secho(f"    ✅ CORRECTly allowed valid change", fg='green')
            
            results.append({
                "candidate_id": candidate.id,
                "detected": detected,
                "expected": candidate.expected_result,
                "report": report
            })
            
        return results

@click.command()
@click.argument('task_path')
def main(task_path):
    engine = BenchmarkEngine(os.getcwd())
    if task_path:
        t = engine.load_task(task_path)
        engine.evaluate_task(t)

if __name__ == "__main__":
    main()
