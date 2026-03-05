"""Batch mode orchestration — submit repos to Anthropic Batch API, collect results.

Includes Pass@2 retry for batch-collected findings.
Separated from generate.py to keep both files under 400 lines.
"""

import re
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger("rlaif.batch_orchestrator")


def _make_custom_id(repo: str, batch_index: int) -> str:
    """Build a Batch API-safe custom_id from repo name + batch index.

    Anthropic requires: ^[a-zA-Z0-9_-]{1,64}$
    Strategy: sanitize the repo name, truncate if needed, append batch index.
    Also store a mapping so we can reverse-lookup the repo later.
    """
    # Replace / with -- (owner--repo), strip anything not alphanumeric/-/_
    safe = repo.replace("/", "--")
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", safe)
    suffix = f"_b{batch_index}"
    max_name = 64 - len(suffix)
    if len(safe) > max_name:
        # Truncate and append short hash for uniqueness
        short_hash = hashlib.md5(repo.encode()).hexdigest()[:6]
        safe = safe[: max_name - 7] + "_" + short_hash
    return safe + suffix


def _repo_from_custom_id(custom_id: str) -> str:
    """Best-effort reverse of _make_custom_id: extract repo from custom_id.

    Handles both old format (owner_repo_bN) and new format (owner--repo_bN).
    """
    # Strip batch suffix
    base = re.sub(r"_b\d+$", "", custom_id)
    # New format: owner--repo
    if "--" in base:
        return base.replace("--", "/", 1)
    # Old format fallback: owner_repo (ambiguous but best effort)
    return base.replace("_", "/", 1)


def submit_batch(
    repos: list,
    workspace: str,
    output_dir: str,
    model_name: str,
    batch_size: int = 15,
):
    """Clone repos, extract facts, submit all prompts to Anthropic Batch API."""
    from .batch_provider import create_batch, save_batch_state
    from .provider import TEACHER_SYSTEM_PROMPT
    from .generate import clone_repo
    from .facts import extract_repo_facts, facts_to_prompt

    all_prompts: list = []
    for repo in repos:
        repo_path = clone_repo(repo, workspace)
        facts = extract_repo_facts(repo_path)
        if not facts:
            logger.warning(f"No facts for {repo}, skipping")
            continue
        batch_count = max(1, len(facts) // batch_size)
        for i in range(batch_count):
            batch = facts[i * batch_size: (i + 1) * batch_size]
            prompt = facts_to_prompt(batch)
            cid = _make_custom_id(repo, i)
            all_prompts.append({"custom_id": cid, "content": prompt})

    if not all_prompts:
        logger.error("No prompts to submit")
        return

    logger.info(f"Submitting {len(all_prompts)} requests to Batch API")
    batch_id = create_batch(
        all_prompts,
        model=model_name,
        system_prompt=TEACHER_SYSTEM_PROMPT,
    )
    save_batch_state(
        output_dir, batch_id,
        repo=",".join(repos[:5]),
        model=model_name,
        prompt_count=len(all_prompts),
    )
    logger.info(f"Batch submitted: {batch_id}")
    logger.info(
        "Collect later: python -m rlaif.generate "
        f"--batch-collect {batch_id}"
    )


def collect_batch(
    batch_id: str, output_dir: str, db_conn,
    enable_retry: bool = True,
):
    """Poll + collect results from a submitted batch, store in DB.

    Verifies ALL findings against AST facts before storing.
    When enable_retry=True, runs Pass@2 retry on findings that fail
    structural verification (using live API calls for retries).
    """
    from .batch_provider import (
        poll_batch, collect_results, parse_batch_findings,
        remove_completed_batch,
    )
    from .generate import TrainingExample, save_examples
    from .verifier import verify_finding

    status = poll_batch(batch_id)
    if status == "timeout":
        logger.error(f"Batch {batch_id} did not complete in time")
        return

    results_map, ok, err = collect_results(batch_id)
    findings = parse_batch_findings(results_map)
    logger.info(
        f"Batch {batch_id}: {len(findings)} findings from {ok} responses"
    )

    # Verify ALL findings and build examples with correct verified status.
    # Previous bug: _build_examples marked ALL as verified=False, and
    # _batch_verify_and_retry skipped verified ones without updating them.
    examples = _verify_and_build_examples(
        findings, batch_id, output_dir, enable_retry,
    )

    save_examples(db_conn, examples)
    remove_completed_batch(output_dir, batch_id)
    verified = sum(1 for ex in examples if ex.verified)
    dropped = sum(1 for ex in examples if not ex.verified)
    logger.info(
        f"Stored {len(examples)} batch findings "
        f"({verified} verified, {dropped} dropped)"
    )


def _verify_and_build_examples(
    findings: List[Dict],
    batch_id: str,
    output_dir: str,
    enable_retry: bool = True,
    max_retries: int = 50,
) -> list:
    """Verify ALL batch findings against AST facts and build examples.

    This replaces the old _build_examples + _batch_verify_and_retry split
    which had a critical bug: _build_examples marked everything as
    verified=False, and _batch_verify_and_retry skipped verified findings
    assuming they were already correct.

    Now:
    1. Groups findings by repo, clones repos, extracts facts
    2. Verifies EVERY finding — marks verified=True/False correctly
    3. Optionally retries rejected findings (Pass@2) via live API
    4. Returns complete list of TrainingExample objects
    """
    from .generate import TrainingExample, clone_repo
    from .facts import extract_repo_facts, facts_to_prompt
    from .verifier import verify_finding
    from .provider import call_teacher_retry

    # Group findings by repo (extracted from custom_id)
    by_repo: Dict[str, list] = {}
    orphan_findings: list = []
    for f in findings:
        bid = f.get("_batch_id", "")
        repo = _repo_from_custom_id(bid) if bid else ""
        if repo:
            by_repo.setdefault(repo, []).append(f)
        else:
            orphan_findings.append(f)

    examples: list = []
    now = datetime.now(timezone.utc).isoformat()
    retry_count = 0
    total_verified = 0
    total_dropped = 0

    for repo, repo_findings in by_repo.items():
        try:
            repo_path = clone_repo(repo, ".rlaif_repos")
            facts = extract_repo_facts(repo_path)
            if not facts:
                # No facts — can't verify, mark all as unverified
                for f in repo_findings:
                    examples.append(TrainingExample(
                        repo=repo, scan_id=f"batch_{batch_id}",
                        file_path=f.get("file", ""), finding=f,
                        verified=False,
                        verification_notes="no_facts_available",
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))
                    total_dropped += 1
                continue

            facts_by_path = {f.path: f for f in facts}
            facts_prompt = facts_to_prompt(facts)

            for finding in repo_findings:
                verified, notes = verify_finding(finding, facts_by_path)

                if verified:
                    # Pass@1 verified — store with correct status
                    examples.append(TrainingExample(
                        repo=repo, scan_id=f"batch_{batch_id}",
                        file_path=finding.get("file", ""), finding=finding,
                        verified=True,
                        verification_notes=notes,
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))
                    total_verified += 1
                    continue

                # Not verified — store as dropped first
                examples.append(TrainingExample(
                    repo=repo, scan_id=f"batch_{batch_id}",
                    file_path=finding.get("file", ""), finding=finding,
                    verified=False,
                    verification_notes=notes,
                    teacher_model=f"batch:{batch_id}",
                    timestamp=now, facts_hash="",
                ))
                total_dropped += 1

                # Pass@2 retry (if enabled and retryable)
                if not enable_retry or retry_count >= max_retries:
                    continue
                if "confidence floor" in notes.lower():
                    continue

                corrected = call_teacher_retry(
                    finding=finding,
                    rejection_reason=notes,
                    facts_prompt=facts_prompt,
                )
                retry_count += 1

                if not corrected:
                    examples.append(TrainingExample(
                        repo=repo, scan_id=f"batch_{batch_id}",
                        file_path=finding.get("file", ""), finding=finding,
                        verified=False,
                        verification_notes="pass2:teacher_gave_up",
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))
                    continue

                for cf in corrected[:1]:
                    v2, n2 = verify_finding(cf, facts_by_path)
                    examples.append(TrainingExample(
                        repo=repo, scan_id=f"batch_{batch_id}",
                        file_path=cf.get("file", ""), finding=cf,
                        verified=v2,
                        verification_notes=(
                            f"pass2:verified_retry|{n2}" if v2
                            else f"pass2:rejected_again|{n2}"
                        ),
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))
                    if v2:
                        total_verified += 1
                    else:
                        total_dropped += 1

                time.sleep(0.5)  # Rate limit

        except Exception as e:
            logger.warning(f"Verification failed for {repo}: {e}")
            # Store unverified on error
            for f in repo_findings:
                examples.append(TrainingExample(
                    repo=repo, scan_id=f"batch_{batch_id}",
                    file_path=f.get("file", ""), finding=f,
                    verified=False,
                    verification_notes=f"verification_error:{e}",
                    teacher_model=f"batch:{batch_id}",
                    timestamp=now, facts_hash="",
                ))
                total_dropped += 1

    # Handle orphan findings (no repo identified)
    for f in orphan_findings:
        examples.append(TrainingExample(
            repo="", scan_id=f"batch_{batch_id}",
            file_path=f.get("file", ""), finding=f,
            verified=False,
            verification_notes="orphan:no_repo_identified",
            teacher_model=f"batch:{batch_id}",
            timestamp=now, facts_hash="",
        ))
        total_dropped += 1

    logger.info(
        f"Batch verification: {total_verified} verified, "
        f"{total_dropped} dropped ({retry_count} retries used)"
    )
    return examples
