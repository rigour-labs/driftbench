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

    @staticmethod
    def from_json(path: str) -> 'Task':
        with open(path, 'r') as f:
            return Task(**json.load(f))

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
        
        # Aggressive cleanup if repo exists to save space
        if os.path.exists(repo_path):
            import shutil
            shutil.rmtree(repo_path)

        click.echo(f"    📥 Cloning {repository} (Shallow)...")
        url = f"https://github.com/{repository}.git"
        
        # Attempt shallow clone of the specific SHA if supported, or just the default branch
        try:
            # Note: depth 1 with a SHA is usually only supported if the server allows it
            # For robustness, we clone the default branch with depth 1 first
            subprocess.run(["git", "clone", "--depth", "1", url, repo_path], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback to full clone if shallow fails (rare)
            subprocess.run(["git", "clone", url, repo_path], check=True, capture_output=True)
        
        click.echo(f"    git checkout {base_sha}")
        try:
            # Try to fetch the specific SHA if it's not in the shallow clone
            subprocess.run(["git", "fetch", "--depth", "1", "origin", base_sha], cwd=repo_path, capture_output=True)
            subprocess.run(["git", "checkout", base_sha], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Last resort: just try checking out if it's a branch name
            subprocess.run(["git", "checkout", base_sha], cwd=repo_path, check=True, capture_output=True)

        return repo_path

    def apply_patch(self, repo_path: str, patch_path: str):
        full_patch_path = os.path.abspath(patch_path)
        click.echo(f"    🩹 Applying patch (Lenient): {patch_path}")
        # Use 'patch -p1' which is more lenient than 'git apply' regarding whitespace and offsets
        try:
            # We use --fuzz=3 to allow minor context mismatches often created by LLMs
            subprocess.run(["patch", "-p1", "--input", full_patch_path], cwd=repo_path, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            # If patch fails, log the specific error for debugging
            click.secho(f"    ❌ Patch failure details: {e.stderr.decode()}", fg='red')
            raise

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

    def evaluate_patch(self, task: Task, patch_path: str, candidate_id: str, results_dir: str = "results") -> Dict:
        """Evaluates a single patch for a task and collects events."""
        repo_path = self.setup_repo(task.repository, task.base_sha)
        
        # Reset and apply patch
        subprocess.run(["git", "clean", "-fd"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=repo_path, check=True, capture_output=True)
        self.apply_patch(repo_path, patch_path)
        
        # Run Rigour
        report = self.run_rigour(repo_path, task.rigour_config)
        detected = report.get("status") == "FAIL"
        
        # Collect events for Studio Integration
        events_src = os.path.join(repo_path, ".rigour", "events.jsonl")
        if os.path.exists(events_src):
            studio_dir = os.path.join(results_dir, "studio", task.id)
            os.makedirs(studio_dir, exist_ok=True)
            import shutil
            shutil.copy2(events_src, os.path.join(studio_dir, f"{candidate_id}.jsonl"))
        
        # CRITICAL: Aggressive cleanup of the repo to prevent OOM/Disk pressure
        try:
            import shutil
            shutil.rmtree(repo_path)
        except:
            pass
            
        return {
            "candidate_id": candidate_id,
            "detected": detected,
            "report": report
        }

    def evaluate_task(self, task: Task):
        click.echo(f"🚀 Evaluating Task: {task.name} ({task.id})")
        
        # Ensure results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # 1. Validate Golden Patch (The "Null" Case)
        click.echo("  🌟 Validating Golden Patch (True Negative Check)...")
        gold_result = self.evaluate_patch(task, task.golden_patch, "gold_baseline", results_dir)
        
        if not gold_result["detected"]:
            click.secho("    ✅ Golden patch passed correctly.", fg='green')
        else:
            click.secho("    ❌ FAILED: Golden patch triggered a false positive!", fg='red', bold=True)
            return {"status": "INVALID_TASK", "errors": gold_result["report"].get("failures")}

        # 2. Evaluate Drift Candidates (The "Drift" Cases)
        results = []
        for candidate in task.drift_candidates:
            click.echo(f"  🔍 Checking Candidate: {candidate.id} ({candidate.drift_type})")
            res = self.evaluate_patch(task, candidate.patch, candidate.id, results_dir)
            
            # Evaluation logic
            if res["detected"]:
                # Check if the specific gate failed if specified
                failed_gates = res["report"].get("summary", {})
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
                **res,
                "expected": candidate.expected_result,
                "drift_type": candidate.drift_type
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
