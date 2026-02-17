"""
Thread-safe logging for DriftBench parallel execution.

All output from engine.py and harness.py goes through this module,
which ensures:
1. Messages from different threads don't interleave
2. Each message is prefixed with the task ID for traceability
3. Output is clean and unambiguous even with 16+ parallel workers
"""
import threading
import click

_lock = threading.Lock()
_task_context = threading.local()


def set_task_context(task_id: str):
    """Set the current task ID for this thread's log messages."""
    _task_context.task_id = task_id


def clear_task_context():
    """Clear the task context for this thread."""
    _task_context.task_id = None


def _prefix():
    """Get the current thread's task prefix, if any."""
    tid = getattr(_task_context, 'task_id', None)
    return f"[{tid}] " if tid else ""


def echo(msg, **kwargs):
    """Thread-safe click.echo with optional task prefix."""
    with _lock:
        click.echo(f"{_prefix()}{msg}", **kwargs)


def secho(msg, **kwargs):
    """Thread-safe click.secho with optional task prefix."""
    with _lock:
        click.secho(f"{_prefix()}{msg}", **kwargs)
