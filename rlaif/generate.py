"""RLAIF Synthetic Data Generator.

Orchestrates the full pipeline: clone repos -> extract facts ->
call teacher model -> verify findings -> store in SQLite.

Works with ANY OpenAI SDK-compatible provider:
    python -m rlaif.generate --provider deepseek --model-name deepseek-chat
    python -m rlaif.generate --provider ollama --model-name qwen2.5-coder:7b
    MODEL_PROVIDER=openai MODEL_NAME=gpt-4o python -m rlaif.generate
"""

import os
import json
import time
import hashlib
import sqlite3
import logging
import argparse
import subprocess
from typing import Dict, List
from datetime import datetime, timezone
from dataclasses import dataclass

from dotenv import load_dotenv

from .facts import FileFact, extract_repo_facts, facts_to_prompt
from .verifier import verify_finding
from .provider import (
    DEFAULT_PROVIDER, DEFAULT_MODEL_NAME, DEFAULT_API_KEY,
    DEFAULT_API_BASE, DEFAULT_TEACHER_MODEL,
    setup_provider, build_model_string, call_teacher,
)

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rlaif.generate")


@dataclass
class TrainingExample:
    """One training data point with verification result."""

    repo: str
    scan_id: str
    file_path: str
    finding: Dict
    verified: bool
    verification_notes: str
    teacher_model: str
    timestamp: str
    facts_hash: str


# ── Repo cloning ──


def clone_repo(
    repo: str, workspace: str, shallow: bool = True
) -> str:
    """Clone a GitHub repo into workspace. Returns repo path."""
    repo_name = repo.replace("/", "_")
    repo_path = os.path.join(workspace, repo_name)

    if os.path.exists(repo_path):
        logger.info(f"Reusing cached repo: {repo_path}")
        subprocess.run(
            ["git", "checkout", "."],
            cwd=repo_path, capture_output=True,
        )
        subprocess.run(
            ["git", "clean", "-fd"],
            cwd=repo_path, capture_output=True,
        )
        return repo_path

    url = f"https://github.com/{repo}.git"
    logger.info(f"Cloning {url} -> {repo_path}")

    cmd = ["git", "clone"]
    if shallow:
        cmd += ["--depth", "1"]
    cmd += [url, repo_path]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to clone {repo}: {result.stderr}")
    return repo_path


# ── SQLite storage ──


def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite database for training data."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL,
            repo TEXT NOT NULL,
            file_path TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT,
            confidence REAL,
            description TEXT,
            suggestion TEXT,
            verified INTEGER NOT NULL,
            verification_notes TEXT,
            teacher_model TEXT NOT NULL,
            facts_hash TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(scan_id, file_path, category, description)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scan_log (
            scan_id TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            teacher_model TEXT NOT NULL,
            total_findings INTEGER,
            verified_count INTEGER,
            dropped_count INTEGER,
            duration_ms INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_category "
        "ON training_data(category)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_verified "
        "ON training_data(verified)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_repo "
        "ON training_data(repo)"
    )
    conn.commit()
    return conn


def save_examples(
    conn: sqlite3.Connection, examples: List[TrainingExample]
):
    """Save training examples to SQLite."""
    for ex in examples:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO training_data
                (scan_id, repo, file_path, category, severity,
                 confidence, description, suggestion, verified,
                 verification_notes, teacher_model, facts_hash,
                 created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ex.scan_id, ex.repo, ex.file_path,
                ex.finding.get("category", ""),
                ex.finding.get("severity", ""),
                ex.finding.get("confidence", 0.0),
                ex.finding.get("description", ""),
                ex.finding.get("suggestion", ""),
                1 if ex.verified else 0,
                ex.verification_notes, ex.teacher_model,
                ex.facts_hash, ex.timestamp,
            ))
        except Exception as e:
            logger.warning(f"Failed to save example: {e}")
    conn.commit()


# ── Pipeline ──


def process_repo(
    repo: str,
    workspace: str,
    db_conn: sqlite3.Connection,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    batch_size: int = 15,
    api_base: str = "",
) -> Dict:
    """Full RLAIF pipeline for one repository."""
    scan_id = f"{repo.replace('/', '_')}_{int(time.time())}"
    start = time.time()

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing: {repo}")

    repo_path = clone_repo(repo, workspace)
    facts = extract_repo_facts(repo_path)
    if not facts:
        logger.warning(f"No facts extracted from {repo}")
        return {"repo": repo, "status": "empty", "findings": 0}

    facts_by_path = {f.path: f for f in facts}
    facts_json = json.dumps(
        [f.__dict__ for f in facts], sort_keys=True, default=str
    )
    facts_hash = hashlib.sha256(facts_json.encode()).hexdigest()[:16]

    all_findings = _call_teacher_batched(
        facts, teacher_model, batch_size, api_base
    )
    examples, verified_count, dropped_count = _verify_all(
        all_findings, facts_by_path, repo, scan_id,
        teacher_model, facts_hash
    )
    save_examples(db_conn, examples)

    duration_ms = int((time.time() - start) * 1000)
    _log_scan(
        db_conn, scan_id, repo, teacher_model,
        len(all_findings), verified_count, dropped_count, duration_ms,
    )
    logger.info(
        f"Results: {verified_count} verified, "
        f"{dropped_count} dropped ({duration_ms}ms)"
    )
    return {
        "repo": repo, "status": "ok",
        "total_findings": len(all_findings),
        "verified": verified_count, "dropped": dropped_count,
        "duration_ms": duration_ms,
    }


def _call_teacher_batched(
    facts: list, model: str, batch_size: int, api_base: str
) -> list:
    all_findings = []
    batch_count = max(1, len(facts) // batch_size)
    for i in range(batch_count):
        batch = facts[i * batch_size: (i + 1) * batch_size]
        prompt = facts_to_prompt(batch)
        findings = call_teacher(
            prompt, model=model, batch_index=i, api_base=api_base
        )
        all_findings.extend(findings)
        if i < batch_count - 1:
            time.sleep(1)
    logger.info(f"Total findings from teacher: {len(all_findings)}")
    return all_findings


def _verify_all(
    findings: list, facts_by_path: dict, repo: str,
    scan_id: str, teacher_model: str, facts_hash: str,
):
    examples = []
    verified_count, dropped_count = 0, 0
    for finding in findings:
        verified, notes = verify_finding(finding, facts_by_path)
        if verified:
            verified_count += 1
        else:
            dropped_count += 1
        examples.append(TrainingExample(
            repo=repo, scan_id=scan_id,
            file_path=finding.get("file", ""),
            finding=finding, verified=verified,
            verification_notes=notes,
            teacher_model=teacher_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            facts_hash=facts_hash,
        ))
    return examples, verified_count, dropped_count


def _log_scan(
    conn, scan_id, repo, model, total, verified, dropped, duration_ms
):
    conn.execute("""
        INSERT OR REPLACE INTO scan_log
        (scan_id, repo, teacher_model, total_findings,
         verified_count, dropped_count, duration_ms, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        scan_id, repo, model, total, verified, dropped,
        duration_ms, datetime.now(timezone.utc).isoformat(),
    ))
    conn.commit()


# ── CLI ──


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RLAIF Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --provider anthropic --model-name claude-sonnet-4-20250514
  --provider deepseek  --model-name deepseek-chat
  --provider ollama    --model-name qwen2.5-coder:7b
        """,
    )
    parser.add_argument("--repo", type=str, help="Single repo (owner/repo)")
    parser.add_argument("--repos", type=str, help="JSON file with repos")
    parser.add_argument(
        "--output", type=str, default="rlaif/data", help="Output dir"
    )
    parser.add_argument(
        "--provider", type=str, default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model-name", type=str, default=DEFAULT_MODEL_NAME,
        help=f"Model name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--api-key", type=str, default=DEFAULT_API_KEY,
        help="API key",
    )
    parser.add_argument(
        "--api-base", type=str, default=DEFAULT_API_BASE,
        help="Custom API base URL",
    )
    parser.add_argument(
        "--model", type=str, default="",
        help="(Legacy) Full litellm model string",
    )
    parser.add_argument(
        "--workspace", type=str, default=".rlaif_repos",
        help="Repo clone directory",
    )
    return parser


def _resolve_repos(args) -> list:
    if args.repo:
        return [args.repo]
    if args.repos:
        with open(args.repos) as f:
            data = json.load(f)
            return (data if isinstance(data, list)
                    else data.get("repos", []))
    repos_file = os.path.join(
        os.path.dirname(__file__), "..", "repos.json"
    )
    if os.path.exists(repos_file):
        with open(repos_file) as f:
            data = json.load(f)
            return [
                r["full_name"] if isinstance(r, dict) else r
                for r in data
            ]
    return []


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.model:
        teacher_model = args.model
        provider = (args.model.split("/")[0] if "/" in args.model
                    else args.provider)
    else:
        provider = args.provider
        teacher_model = build_model_string(provider, args.model_name)

    setup_provider(provider, args.api_key, args.api_base)

    repos = _resolve_repos(args)
    if not repos:
        parser.error("Provide --repo or --repos")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)

    db_path = os.path.join(args.output, "training_data.db")
    conn = init_db(db_path)

    logger.info(f"Processing {len(repos)} repos with {teacher_model}")

    results = []
    for repo in repos:
        try:
            result = process_repo(
                repo, args.workspace, conn,
                teacher_model=teacher_model,
                api_base=args.api_base,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {repo}: {e}")
            results.append({"repo": repo, "status": "error", "error": str(e)})

    total_v = sum(r.get("verified", 0) for r in results)
    total_d = sum(r.get("dropped", 0) for r in results)
    logger.info(
        f"SUMMARY: {total_v} verified, {total_d} dropped "
        f"across {len(repos)} repos"
    )

    summary_path = os.path.join(args.output, "generate_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "teacher_model": teacher_model,
            "provider": provider,
            "api_base": args.api_base or None,
            "repos_processed": len(repos),
            "total_verified": total_v,
            "total_dropped": total_d,
            "results": results,
        }, f, indent=2)

    conn.close()


if __name__ == "__main__":
    main()
