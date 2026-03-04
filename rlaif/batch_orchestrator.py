"""Batch mode orchestration — submit repos to Anthropic Batch API, collect results.

Includes Pass@2 retry for batch-collected findings.
Separated from generate.py to keep both files under 400 lines.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List

logger = logging.getLogger("rlaif.batch_orchestrator")


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
            cid = f"{repo.replace('/', '_')}_b{i}"
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

    # Build basic examples (unverified by default in batch mode)
    examples = _build_examples(findings, batch_id)

    # If retry is enabled, verify + retry rejected findings
    if enable_retry and findings:
        retry_examples = _batch_verify_and_retry(
            findings, batch_id, output_dir,
        )
        examples.extend(retry_examples)

    save_examples(db_conn, examples)
    remove_completed_batch(output_dir, batch_id)
    logger.info(f"Stored {len(examples)} batch findings")


def _batch_verify_and_retry(
    findings: List[Dict],
    batch_id: str,
    output_dir: str,
    max_retries: int = 50,
    model: str = "",
) -> list:
    """Verify batch findings and retry rejected ones via live API.

    Batch mode skips verification by default. This function:
    1. Groups findings by repo, clones repos to get facts
    2. Runs structural verification on each finding
    3. Retries rejected findings (Pass@2) via live API
    4. Returns additional TrainingExample objects for retry results
    """
    from .generate import TrainingExample, clone_repo
    from .facts import extract_repo_facts, facts_to_prompt
    from .verifier import verify_finding
    from .provider import call_teacher_retry

    # Group findings by repo (extracted from custom_id)
    by_repo: Dict[str, list] = {}
    for f in findings:
        bid = f.get("_batch_id", "")
        repo = bid.rsplit("_b", 1)[0].replace("_", "/") if bid else ""
        if repo:
            by_repo.setdefault(repo, []).append(f)

    retry_examples: list = []
    now = datetime.now(timezone.utc).isoformat()
    retry_count = 0

    for repo, repo_findings in by_repo.items():
        try:
            repo_path = clone_repo(repo, ".rlaif_repos")
            facts = extract_repo_facts(repo_path)
            if not facts:
                continue
            facts_by_path = {f.path: f for f in facts}
            facts_prompt = facts_to_prompt(facts)

            for finding in repo_findings:
                if retry_count >= max_retries:
                    break
                verified, notes = verify_finding(finding, facts_by_path)
                if verified:
                    continue  # Already good, handled by _build_examples

                # Skip non-retryable rejections
                if "confidence floor" in notes.lower():
                    continue

                # Pass@2: retry via live API
                retry_kwargs = {
                    "finding": finding,
                    "rejection_reason": notes,
                    "facts_prompt": facts_prompt,
                }
                if model:
                    retry_kwargs["model"] = model
                corrected = call_teacher_retry(**retry_kwargs)
                retry_count += 1

                if not corrected:
                    retry_examples.append(TrainingExample(
                        repo=repo,
                        scan_id=f"batch_{batch_id}",
                        file_path=finding.get("file", ""),
                        finding=finding,
                        verified=False,
                        verification_notes="pass2:teacher_gave_up",
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))
                    continue

                for cf in corrected[:1]:
                    v2, n2 = verify_finding(cf, facts_by_path)
                    retry_examples.append(TrainingExample(
                        repo=repo,
                        scan_id=f"batch_{batch_id}",
                        file_path=cf.get("file", ""),
                        finding=cf,
                        verified=v2,
                        verification_notes=(
                            f"pass2:verified_retry|{n2}" if v2
                            else f"pass2:rejected_again|{n2}"
                        ),
                        teacher_model=f"batch:{batch_id}",
                        timestamp=now, facts_hash="",
                    ))

                time.sleep(0.5)  # Rate limit

        except Exception as e:
            logger.warning(f"Batch retry failed for {repo}: {e}")

    v = sum(1 for ex in retry_examples if ex.verified)
    d = sum(1 for ex in retry_examples if not ex.verified)
    logger.info(f"Batch Pass@2: {v} recovered, {d} permanently rejected")
    return retry_examples


def _build_examples(findings: List[Dict], batch_id: str) -> list:
    """Convert batch findings to TrainingExample objects."""
    from .generate import TrainingExample

    examples = []
    now = datetime.now(timezone.utc).isoformat()
    for f in findings:
        bid = f.get("_batch_id", "")
        repo = bid.rsplit("_b", 1)[0].replace("_", "/") if bid else ""
        examples.append(TrainingExample(
            repo=repo,
            scan_id=f"batch_{batch_id}",
            file_path=f.get("file", ""),
            finding=f,
            verified=False,
            verification_notes="batch_mode:unverified",
            teacher_model=f"batch:{batch_id}",
            timestamp=now,
            facts_hash="",
        ))
    return examples
