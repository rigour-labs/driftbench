#!/usr/bin/env python3
"""
Run full DriftBench benchmark across all models and tasks.

Supports parallel execution for faster benchmarking.

Usage:
    python scripts/run_full_benchmark.py --model anthropic/claude-opus-4-6-20260205
    python scripts/run_full_benchmark.py --model anthropic/claude-opus-4-6-20260205 --parallel 6
    python scripts/run_full_benchmark.py --task lodash-stale-001
    python scripts/run_full_benchmark.py --dry-run
"""
import os
import sys
import json
import glob
import click
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.harness import LLMHarness
from runner.engine import Task

# Thread-safe print lock
_print_lock = Lock()


def safe_echo(msg, **kwargs):
    """Thread-safe click.echo."""
    with _print_lock:
        click.echo(msg, **kwargs)


def safe_secho(msg, **kwargs):
    """Thread-safe click.secho."""
    with _print_lock:
        click.secho(msg, **kwargs)


def load_models() -> list:
    """Load models from model_config.json."""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model_config.json"
    )

    if not os.path.exists(config_path):
        click.secho("❌ model_config.json not found", fg='red')
        return []

    with open(config_path, 'r') as f:
        config = json.load(f)

    return list(config.get("model_config", {}).keys())


def load_tasks() -> list:
    """Load all task files."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_files = glob.glob(os.path.join(base_dir, "datasets/**/*.json"), recursive=True)

    tasks = []
    for f in task_files:
        try:
            task = Task.from_json(f)
            tasks.append(task)
        except Exception as e:
            click.secho(f"⚠️  Skipping invalid task {f}: {e}", fg='yellow')

    return tasks


def run_single_task(model: str, task: Task, worker_id: int, task_num: int, total: int) -> dict:
    """
    Run a single benchmark task in an isolated workspace.

    Each worker gets its own workspace directory to avoid repo conflicts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    worker_workspace = os.path.join(base_dir, f".drift_workers/worker_{worker_id}")
    os.makedirs(worker_workspace, exist_ok=True)

    safe_echo(f"\n[{task_num}/{total}] 🚀 {task.id} (worker {worker_id})")

    try:
        harness = LLMHarness(model, workspace_root=worker_workspace)
        result = harness.run_task(task)

        if result.get("passed"):
            safe_secho(f"    [{task.id}] ✅ PASSED", fg='green')
        elif result.get("error"):
            safe_secho(f"    [{task.id}] ❌ ERROR: {result['error'][:60]}", fg='red')
        else:
            safe_secho(f"    [{task.id}] 🔴 DRIFT DETECTED", fg='red')

        # Copy results back to main results dir
        model_slug = model.replace("/", "_")
        worker_result_path = os.path.join(worker_workspace, "results", model_slug, f"{task.id}.json")
        main_result_dir = os.path.join(base_dir, "results", model_slug)
        os.makedirs(main_result_dir, exist_ok=True)

        if os.path.exists(worker_result_path):
            shutil.copy2(worker_result_path, os.path.join(main_result_dir, f"{task.id}.json"))

        # Copy studio events too
        worker_studio_dir = os.path.join(worker_workspace, "results", model_slug, "studio", task.id)
        if os.path.exists(worker_studio_dir):
            main_studio_dir = os.path.join(main_result_dir, "studio", task.id)
            os.makedirs(main_studio_dir, exist_ok=True)
            for f in os.listdir(worker_studio_dir):
                shutil.copy2(os.path.join(worker_studio_dir, f), os.path.join(main_studio_dir, f))

        # Copy patches
        worker_patch_dir = os.path.join(worker_workspace, "results", model_slug, "patches", task.id)
        if os.path.exists(worker_patch_dir):
            main_patch_dir = os.path.join(main_result_dir, "patches", task.id)
            os.makedirs(main_patch_dir, exist_ok=True)
            for f in os.listdir(worker_patch_dir):
                shutil.copy2(os.path.join(worker_patch_dir, f), os.path.join(main_patch_dir, f))

        return {"task_id": task.id, "result": result, "error": None}

    except Exception as e:
        safe_secho(f"    [{task.id}] ❌ Exception: {e}", fg='red')
        return {"task_id": task.id, "result": None, "error": str(e)}


@click.command()
@click.option('--model', help='Run only this model (default: all models from config)')
@click.option('--task', 'task_id', help='Run only this task ID (default: all tasks)')
@click.option('--parallel', '-p', default=1, type=int, help='Number of parallel workers (default: 1, recommended: 4-6)')
@click.option('--dry-run', is_flag=True, help='Show what would be run without executing')
@click.option('--clean', is_flag=True, help='Clean worker directories before run')
def main(model: str, task_id: str, parallel: int, dry_run: bool, clean: bool):
    """Run full DriftBench benchmark with optional parallelism."""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load models
    if model:
        models = [model]
    else:
        models = load_models()

    if not models:
        click.secho("❌ No models configured", fg='red')
        return

    # Load tasks
    all_tasks = load_tasks()

    if task_id:
        tasks = [t for t in all_tasks if t.id == task_id]
        if not tasks:
            click.secho(f"❌ Task not found: {task_id}", fg='red')
            return
    else:
        tasks = all_tasks

    total_runs = len(models) * len(tasks)

    click.echo("=" * 60)
    click.echo(f"🚀 DriftBench Full Benchmark")
    click.echo(f"   Models: {len(models)}")
    click.echo(f"   Tasks: {len(tasks)}")
    click.echo(f"   Total Runs: {total_runs}")
    click.echo(f"   Parallel Workers: {parallel}")
    click.echo("=" * 60)

    if dry_run:
        click.echo("\n📋 Would run:")
        for m in models:
            click.echo(f"\n  Model: {m}")
            for t in tasks[:5]:
                click.echo(f"    - {t.id}")
            if len(tasks) > 5:
                click.echo(f"    ... and {len(tasks) - 5} more tasks")
        click.echo(f"\n  With {parallel} parallel worker(s)")
        return

    # Clean worker directories if requested
    workers_dir = os.path.join(base_dir, ".drift_workers")
    if clean and os.path.exists(workers_dir):
        click.echo("🧹 Cleaning worker directories...")
        shutil.rmtree(workers_dir, ignore_errors=True)

    # Run benchmarks
    start_time = datetime.now()
    completed = 0
    errors = []

    for m in models:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"🤖 Model: {m}")
        click.echo(f"{'=' * 60}")

        if parallel <= 1:
            # Sequential mode (original behavior)
            try:
                harness = LLMHarness(m)
            except Exception as e:
                click.secho(f"❌ Failed to initialize harness for {m}: {e}", fg='red')
                errors.append({"model": m, "task": None, "error": str(e)})
                continue

            for i, task in enumerate(tasks, 1):
                click.echo(f"\n[{i}/{len(tasks)}] Task: {task.id}")

                try:
                    result = harness.run_task(task)
                    completed += 1

                    if result.get("passed"):
                        click.secho(f"    ✅ PASSED", fg='green')
                    elif result.get("error"):
                        click.secho(f"    ❌ ERROR: {result['error']}", fg='red')
                        errors.append({"model": m, "task": task.id, "error": result["error"]})
                    else:
                        click.secho(f"    🔴 DRIFT DETECTED", fg='red')

                except Exception as e:
                    click.secho(f"    ❌ Exception: {e}", fg='red')
                    errors.append({"model": m, "task": task.id, "error": str(e)})
        else:
            # Parallel mode
            click.echo(f"⚡ Running {len(tasks)} tasks with {parallel} workers...")

            futures = {}
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                for i, task in enumerate(tasks, 1):
                    worker_id = (i - 1) % parallel
                    future = executor.submit(run_single_task, m, task, worker_id, i, len(tasks))
                    futures[future] = task

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        outcome = future.result()
                        completed += 1
                        if outcome["error"]:
                            errors.append({"model": m, "task": outcome["task_id"], "error": outcome["error"]})
                    except Exception as e:
                        errors.append({"model": m, "task": task.id, "error": str(e)})

    # Summary
    elapsed = datetime.now() - start_time
    click.echo("\n" + "=" * 60)
    click.echo("📊 Benchmark Complete")
    click.echo(f"   Completed: {completed}/{total_runs}")
    click.echo(f"   Errors: {len(errors)}")
    click.echo(f"   Duration: {elapsed}")
    click.echo(f"   Workers: {parallel}")
    click.echo("=" * 60)

    if errors:
        click.echo("\n❌ Errors encountered:")
        for err in errors[:10]:
            click.echo(f"   - {err['model']} / {err['task']}: {str(err['error'])[:60]}")
        if len(errors) > 10:
            click.echo(f"   ... and {len(errors) - 10} more errors")

    # Clean up worker directories
    if parallel > 1:
        click.echo("\n🧹 Cleaning worker directories...")
        shutil.rmtree(os.path.join(base_dir, ".drift_workers"), ignore_errors=True)

    # Update leaderboard
    click.echo("\n📸 Updating leaderboard...")
    os.system("python scripts/snapshot_leaderboard.py")


if __name__ == "__main__":
    main()
