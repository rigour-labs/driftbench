#!/usr/bin/env python3
"""
Run full DriftBench benchmark across all models and tasks.

Usage:
    python scripts/run_full_benchmark.py
    python scripts/run_full_benchmark.py --model anthropic/claude-opus-4-5-20251101
    python scripts/run_full_benchmark.py --task lodash-stale-001
"""
import os
import sys
import json
import glob
import click
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from runner.harness import LLMHarness
from runner.engine import Task


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


@click.command()
@click.option('--model', help='Run only this model (default: all models from config)')
@click.option('--task', 'task_id', help='Run only this task ID (default: all tasks)')
@click.option('--dry-run', is_flag=True, help='Show what would be run without executing')
def main(model: str, task_id: str, dry_run: bool):
    """Run full DriftBench benchmark."""

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
    click.echo("=" * 60)

    if dry_run:
        click.echo("\n📋 Would run:")
        for m in models:
            click.echo(f"\n  Model: {m}")
            for t in tasks[:5]:
                click.echo(f"    - {t.id}")
            if len(tasks) > 5:
                click.echo(f"    ... and {len(tasks) - 5} more tasks")
        return

    # Run benchmarks
    start_time = datetime.now()
    completed = 0
    errors = []

    for m in models:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"🤖 Model: {m}")
        click.echo(f"{'=' * 60}")

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

    # Summary
    elapsed = datetime.now() - start_time
    click.echo("\n" + "=" * 60)
    click.echo("📊 Benchmark Complete")
    click.echo(f"   Completed: {completed}/{total_runs}")
    click.echo(f"   Errors: {len(errors)}")
    click.echo(f"   Duration: {elapsed}")
    click.echo("=" * 60)

    if errors:
        click.echo("\n❌ Errors encountered:")
        for err in errors[:10]:
            click.echo(f"   - {err['model']} / {err['task']}: {err['error'][:50]}")
        if len(errors) > 10:
            click.echo(f"   ... and {len(errors) - 10} more errors")

    # Update leaderboard
    click.echo("\n📸 Updating leaderboard...")
    os.system("python scripts/snapshot_leaderboard.py")


if __name__ == "__main__":
    main()
