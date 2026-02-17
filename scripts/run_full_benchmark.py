#!/usr/bin/env python3
"""
Run full DriftBench benchmark across all models and tasks.

Supports parallel execution for faster benchmarking.

Usage:
    python scripts/run_full_benchmark.py --model anthropic/claude-opus-4-6-20260205
    python scripts/run_full_benchmark.py --model anthropic/claude-opus-4-6-20260205 -p 4
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

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.harness import LLMHarness
from runner.engine import Task
from runner import log

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models() -> list:
    """Load models from model_config.json."""
    config_path = os.path.join(BASE_DIR, "model_config.json")
    if not os.path.exists(config_path):
        click.secho("❌ model_config.json not found", fg='red')
        return []
    with open(config_path, 'r') as f:
        config = json.load(f)
    return list(config.get("model_config", {}).keys())


def load_tasks() -> list:
    """Load all task files."""
    task_files = glob.glob(os.path.join(BASE_DIR, "datasets/**/*.json"), recursive=True)
    tasks = []
    for f in task_files:
        try:
            task = Task.from_json(f)
            tasks.append(task)
        except Exception as e:
            click.secho(f"⚠️  Skipping invalid task {f}: {e}", fg='yellow')
    return tasks


def run_single_task(model: str, task: Task, task_num: int, total: int) -> dict:
    """
    Run a single benchmark task in a workspace isolated by task ID.

    Each task gets its own directory so git clones never collide.
    All logging is prefixed with [task.id] via thread-local context.
    """
    # Set thread-local task context so ALL log messages (including from
    # engine.py and harness.py) are prefixed with this task's ID
    log.set_task_context(task.id)

    try:
        # Unique workspace per task — no sharing between concurrent tasks
        task_workspace = os.path.join(BASE_DIR, f".drift_workers/{task.id}")
        os.makedirs(task_workspace, exist_ok=True)

        log.echo(f"[{task_num}/{total}] 🚀 Starting")

        harness = LLMHarness(model, workspace_root=task_workspace)
        result = harness.run_task(task)

        if result.get("passed"):
            log.secho(f"✅ PASSED", fg='green')
        elif result.get("error"):
            log.secho(f"❌ ERROR: {result['error'][:80]}", fg='red')
        else:
            log.secho(f"🔴 DRIFT DETECTED", fg='red')

        # Copy results back to main results dir
        _copy_results_to_main(task_workspace, model, task.id)

        return {"task_id": task.id, "result": result, "error": None}

    except Exception as e:
        log.secho(f"❌ Exception: {e}", fg='red')
        return {"task_id": task.id, "result": None, "error": str(e)}

    finally:
        log.clear_task_context()


def _copy_results_to_main(task_workspace: str, model: str, task_id: str):
    """Copy results from task workspace to main results directory."""
    model_slug = model.replace("/", "_")
    main_result_dir = os.path.join(BASE_DIR, "results", model_slug)
    os.makedirs(main_result_dir, exist_ok=True)

    # Copy task result JSON
    worker_result = os.path.join(task_workspace, "results", model_slug, f"{task_id}.json")
    if os.path.exists(worker_result):
        shutil.copy2(worker_result, os.path.join(main_result_dir, f"{task_id}.json"))

    # Copy studio events
    worker_studio = os.path.join(task_workspace, "results", model_slug, "studio", task_id)
    if os.path.isdir(worker_studio):
        main_studio = os.path.join(main_result_dir, "studio", task_id)
        if os.path.exists(main_studio):
            shutil.rmtree(main_studio)
        shutil.copytree(worker_studio, main_studio)

    # Copy patches
    worker_patches = os.path.join(task_workspace, "results", model_slug, "patches", task_id)
    if os.path.isdir(worker_patches):
        main_patches = os.path.join(main_result_dir, "patches", task_id)
        if os.path.exists(main_patches):
            shutil.rmtree(main_patches)
        shutil.copytree(worker_patches, main_patches)


@click.command()
@click.option('--model', help='Run only this model (default: all models from config)')
@click.option('--task', 'task_id', help='Run only this task ID (default: all tasks)')
@click.option('--parallel', '-p', default=1, type=int, help='Number of parallel workers (default: 1, recommended: 4-6)')
@click.option('--dry-run', is_flag=True, help='Show what would be run without executing')
@click.option('--clean', is_flag=True, help='Clean worker directories before run')
def main(model: str, task_id: str, parallel: int, dry_run: bool, clean: bool):
    """Run full DriftBench benchmark with optional parallelism."""

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
    workers_dir = os.path.join(BASE_DIR, ".drift_workers")
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
            # Sequential mode (original behavior, single shared harness)
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
            # Parallel mode — each task gets its own isolated workspace
            click.echo(f"⚡ Running {len(tasks)} tasks across {parallel} threads...")

            futures = {}
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                for i, task in enumerate(tasks, 1):
                    future = executor.submit(run_single_task, m, task, i, len(tasks))
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
        shutil.rmtree(os.path.join(BASE_DIR, ".drift_workers"), ignore_errors=True)

    # Update leaderboard
    click.echo("\n📸 Updating leaderboard...")
    os.system("python scripts/snapshot_leaderboard.py")


if __name__ == "__main__":
    main()
