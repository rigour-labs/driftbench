"""RLAIF data generator — live (litellm) or batch (Anthropic Batch API, 50% cheaper)."""

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
    setup_provider, build_model_string, call_teacher, call_teacher_retry,
)

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rlaif.generate")

@dataclass
class TrainingExample:
    """Training data point with verification result."""
    repo: str
    scan_id: str
    file_path: str
    finding: Dict
    verified: bool
    verification_notes: str
    teacher_model: str
    timestamp: str
    facts_hash: str
    facts_prompt: str = ""  # AST facts prompt text used during teacher call

def clone_repo(
    repo: str, workspace: str, shallow: bool = True
) -> str:
    """Clone GitHub repo into workspace."""
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
            facts_prompt TEXT,
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
    # Migrate: add facts_prompt column if missing (existing DBs)
    try:
        conn.execute("SELECT facts_prompt FROM training_data LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("ALTER TABLE training_data ADD COLUMN facts_prompt TEXT")
        logger.info("Migrated DB: added facts_prompt column")

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
                 facts_prompt, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ex.scan_id, ex.repo, ex.file_path,
                ex.finding.get("category", ""),
                ex.finding.get("severity", ""),
                ex.finding.get("confidence", 0.0),
                ex.finding.get("description", ""),
                ex.finding.get("suggestion", ""),
                1 if ex.verified else 0,
                ex.verification_notes, ex.teacher_model,
                ex.facts_hash, ex.facts_prompt, ex.timestamp,
            ))
        except Exception as e:
            logger.warning(f"Failed to save example: {e}")
    conn.commit()

def process_repo(
    repo: str,
    workspace: str,
    db_conn: sqlite3.Connection,
    teacher_model: str = DEFAULT_TEACHER_MODEL,
    batch_size: int = 15,
    api_base: str = "",
    enable_retry: bool = True,
) -> Dict:
    """Full RLAIF pipeline for one repository (with Pass@2 retry)."""
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

    # Build facts prompt map for retry (path -> prompt with that file's batch)
    facts_prompt_map = _build_facts_prompt_map(facts, batch_size)

    all_findings = _call_teacher_batched(
        facts, teacher_model, batch_size, api_base
    )
    examples, verified_count, dropped_count = _verify_all(
        all_findings, facts_by_path, repo, scan_id,
        teacher_model, facts_hash, facts_prompt_map,
    )

    # Pass@2: retry rejected findings
    retry_verified = 0
    retry_dropped = 0
    if enable_retry:
        rejected = [ex for ex in examples if not ex.verified]
        retry_v, retry_d, retry_examples = _retry_rejected(
            rejected, facts_by_path, facts_prompt_map,
            repo, scan_id, teacher_model, facts_hash, api_base,
        )
        retry_verified = retry_v
        retry_dropped = retry_d
        examples.extend(retry_examples)
        logger.info(
            f"Pass@2 retry: {retry_verified} recovered, "
            f"{retry_dropped} permanently rejected"
        )

    save_examples(db_conn, examples)

    duration_ms = int((time.time() - start) * 1000)
    total_verified = verified_count + retry_verified
    total_dropped = dropped_count - retry_verified + retry_dropped
    _log_scan(
        db_conn, scan_id, repo, teacher_model,
        len(all_findings), total_verified, total_dropped, duration_ms,
    )
    logger.info(
        f"Results: {total_verified} verified "
        f"({verified_count} pass@1 + {retry_verified} pass@2), "
        f"{total_dropped} dropped ({duration_ms}ms)"
    )
    return {
        "repo": repo, "status": "ok",
        "total_findings": len(all_findings),
        "verified": total_verified, "dropped": total_dropped,
        "pass1_verified": verified_count,
        "pass2_recovered": retry_verified,
        "pass2_rejected": retry_dropped,
        "duration_ms": duration_ms,
    }


def _build_facts_prompt_map(
    facts: list, batch_size: int
) -> Dict[str, str]:
    """Build file_path -> facts_prompt map for retry.

    Each file maps to the prompt for the batch it was included in,
    so the teacher sees the same context during retry.
    """
    prompt_map: Dict[str, str] = {}
    batch_count = max(1, len(facts) // batch_size)
    for i in range(batch_count):
        batch = facts[i * batch_size: (i + 1) * batch_size]
        prompt = facts_to_prompt(batch)
        for f in batch:
            prompt_map[f.path] = prompt
    return prompt_map


def _call_teacher_batched(
    facts: list, model: str, batch_size: int, api_base: str
) -> list:
    all_findings = []
    failed_batches = 0
    batch_count = max(1, len(facts) // batch_size)
    for i in range(batch_count):
        batch = facts[i * batch_size: (i + 1) * batch_size]
        prompt = facts_to_prompt(batch)
        findings = call_teacher(
            prompt, model=model, batch_index=i, api_base=api_base
        )
        if findings:
            all_findings.extend(findings)
        else:
            failed_batches += 1
        if i < batch_count - 1:
            time.sleep(1)
    if failed_batches:
        logger.warning(
            f"{failed_batches}/{batch_count} batches returned no findings "
            f"(check logs for retry failures)"
        )
    logger.info(f"Total findings from teacher: {len(all_findings)}")
    return all_findings


def _verify_all(
    findings: list, facts_by_path: dict, repo: str,
    scan_id: str, teacher_model: str, facts_hash: str,
    facts_prompt_map: Dict[str, str] = None,
):
    examples = []
    verified_count, dropped_count = 0, 0
    for finding in findings:
        verified, notes = verify_finding(finding, facts_by_path)
        if verified:
            verified_count += 1
        else:
            dropped_count += 1
        # Resolve the facts prompt for this finding's file
        file_path = finding.get("file", "")
        fp = ""
        if facts_prompt_map:
            fp = facts_prompt_map.get(file_path, "")
            if not fp:
                # Fallback: partial match
                for path, prompt in facts_prompt_map.items():
                    if path.endswith(file_path) or file_path.endswith(path):
                        fp = prompt
                        break
        examples.append(TrainingExample(
            repo=repo, scan_id=scan_id,
            file_path=file_path,
            finding=finding, verified=verified,
            verification_notes=notes,
            teacher_model=teacher_model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            facts_hash=facts_hash,
            facts_prompt=fp,
        ))
    return examples, verified_count, dropped_count


def _retry_rejected(
    rejected_examples: List[TrainingExample],
    facts_by_path: Dict,
    facts_prompt_map: Dict[str, str],
    repo: str,
    scan_id: str,
    teacher_model: str,
    facts_hash: str,
    api_base: str = "",
    max_retries: int = 50,
) -> tuple:
    """Pass@2: retry rejected findings with rejection reason.

    Sends each rejected finding back to the teacher with:
    - The original AST facts
    - The rejected finding
    - The specific rejection reason

    Returns (verified_count, dropped_count, new_examples).
    """
    if not rejected_examples:
        return 0, 0, []

    # Only retry findings with actionable rejection notes
    # Skip generic "confidence floor" rejections — those aren't fixable
    retryable = [
        ex for ex in rejected_examples
        if ex.verification_notes
        and "confidence floor" not in ex.verification_notes.lower()
        and "batch_mode" not in ex.verification_notes.lower()
    ][:max_retries]

    if not retryable:
        return 0, 0, []

    logger.info(f"Pass@2: retrying {len(retryable)} rejected findings")

    verified_count = 0
    dropped_count = 0
    new_examples: List[TrainingExample] = []

    for ex in retryable:
        file_path = ex.finding.get("file", "")
        facts_prompt = facts_prompt_map.get(file_path, "")
        if not facts_prompt:
            # Fallback: try partial match
            for path, prompt in facts_prompt_map.items():
                if path.endswith(file_path) or file_path.endswith(path):
                    facts_prompt = prompt
                    break
        if not facts_prompt:
            dropped_count += 1
            continue

        corrected = call_teacher_retry(
            finding=ex.finding,
            rejection_reason=ex.verification_notes,
            facts_prompt=facts_prompt,
            model=teacher_model,
            api_base=api_base,
        )

        if not corrected:
            # Teacher gave up — permanent rejection (good negative example)
            new_examples.append(TrainingExample(
                repo=repo, scan_id=scan_id,
                file_path=file_path, finding=ex.finding,
                verified=False,
                verification_notes="pass2:teacher_gave_up",
                teacher_model=teacher_model,
                timestamp=datetime.now(timezone.utc).isoformat(),
                facts_hash=facts_hash,
                facts_prompt=facts_prompt,
            ))
            dropped_count += 1
            continue

        # Re-verify the corrected finding
        for corrected_finding in corrected[:1]:  # Take only first
            verified, notes = verify_finding(corrected_finding, facts_by_path)
            if verified:
                verified_count += 1
                new_examples.append(TrainingExample(
                    repo=repo, scan_id=scan_id,
                    file_path=corrected_finding.get("file", file_path),
                    finding=corrected_finding,
                    verified=True,
                    verification_notes=f"pass2:verified_retry|{notes}",
                    teacher_model=teacher_model,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    facts_hash=facts_hash,
                    facts_prompt=facts_prompt,
                ))
            else:
                dropped_count += 1
                new_examples.append(TrainingExample(
                    repo=repo, scan_id=scan_id,
                    file_path=corrected_finding.get("file", file_path),
                    finding=corrected_finding,
                    verified=False,
                    verification_notes=f"pass2:rejected_again|{notes}",
                    teacher_model=teacher_model,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    facts_hash=facts_hash,
                    facts_prompt=facts_prompt,
                ))

        # Small delay between retries to avoid rate limits
        time.sleep(0.5)

    return verified_count, dropped_count, new_examples


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

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLAIF Training Data Generator")
    parser.add_argument("--repo", type=str, help="Single repo (owner/repo)")
    parser.add_argument("--repos", type=str, help="JSON file with repos")
    parser.add_argument("--output", type=str, default="rlaif/data", help="Output dir")
    parser.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, help="LLM provider")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Model name")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API key")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE, help="Custom API base")
    parser.add_argument("--model", type=str, default="", help="Full litellm model string")
    parser.add_argument("--workspace", type=str, default=".rlaif_repos", help="Clone directory")
    parser.add_argument("--batch", action="store_true", help="Anthropic Batch API (50%% cheaper)")
    parser.add_argument("--batch-collect", type=str, metavar="ID", help="Collect batch results")
    parser.add_argument("--batch-collect-all", action="store_true", help="Collect all pending")
    parser.add_argument("--no-retry", action="store_true", help="Disable Pass@2 retry for rejected findings")
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


def _write_summary(output_dir, results, teacher_model, provider, api_base):
    """Write generate_summary.json with run stats."""
    total_v = sum(r.get("verified", 0) for r in results)
    total_d = sum(r.get("dropped", 0) for r in results)
    logger.info(
        f"SUMMARY: {total_v} verified, {total_d} dropped "
        f"across {len(results)} repos"
    )
    summary_path = os.path.join(output_dir, "generate_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "teacher_model": teacher_model,
            "provider": provider,
            "api_base": api_base or None,
            "repos_processed": len(results),
            "total_verified": total_v,
            "total_dropped": total_d,
            "results": results,
        }, f, indent=2)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.workspace, exist_ok=True)
    db_path = os.path.join(args.output, "training_data.db")
    conn = init_db(db_path)

    # --- Batch collect mode (no provider setup needed) ---
    if args.batch_collect:
        from .batch_orchestrator import collect_batch
        collect_batch(args.batch_collect, args.output, conn)
        conn.close()
        return
    if args.batch_collect_all:
        from .batch_provider import load_pending_batches
        from .batch_orchestrator import collect_batch
        for bid in list(load_pending_batches(args.output).keys()):
            collect_batch(bid, args.output, conn)
        conn.close()
        return

    # --- Resolve model + provider ---
    if args.model:
        teacher_model = args.model
        provider = (args.model.split("/")[0] if "/" in args.model
                    else args.provider)
    else:
        provider = args.provider
        teacher_model = build_model_string(provider, args.model_name)

    repos = _resolve_repos(args)
    if not repos:
        parser.error("Provide --repo or --repos")

    # --- Batch submit mode (Anthropic Batch API — 50% cheaper) ---
    if args.batch:
        from .batch_orchestrator import submit_batch
        submit_batch(
            repos, args.workspace, args.output,
            model_name=args.model_name, batch_size=15,
        )
        conn.close()
        return

    # --- Live mode (synchronous litellm) ---
    setup_provider(provider, args.api_key, args.api_base)
    logger.info(f"Processing {len(repos)} repos with {teacher_model}")

    results = []
    for repo in repos:
        try:
            result = process_repo(
                repo, args.workspace, conn,
                teacher_model=teacher_model,
                api_base=args.api_base,
                enable_retry=not args.no_retry,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {repo}: {e}")
            results.append({"repo": repo, "status": "error", "error": str(e)})

    _write_summary(args.output, results, teacher_model, provider, args.api_base)
    conn.close()


if __name__ == "__main__":
    main()
