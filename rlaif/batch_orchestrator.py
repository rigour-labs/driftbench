"""Batch mode orchestration — submit repos to Anthropic Batch API, collect results.

Separated from generate.py to keep both files under 400 lines.
"""

import logging
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


def collect_batch(batch_id: str, output_dir: str, db_conn):
    """Poll + collect results from a submitted batch, store in DB."""
    from .batch_provider import (
        poll_batch, collect_results, parse_batch_findings,
        remove_completed_batch,
    )
    from .generate import TrainingExample, save_examples

    status = poll_batch(batch_id)
    if status == "timeout":
        logger.error(f"Batch {batch_id} did not complete in time")
        return

    results_map, ok, err = collect_results(batch_id)
    findings = parse_batch_findings(results_map)
    logger.info(
        f"Batch {batch_id}: {len(findings)} findings from {ok} responses"
    )

    examples = _build_examples(findings, batch_id)
    save_examples(db_conn, examples)
    remove_completed_batch(output_dir, batch_id)
    logger.info(f"Stored {len(examples)} batch findings")


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
