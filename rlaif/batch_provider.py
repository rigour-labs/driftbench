"""Anthropic Message Batches API provider for RLAIF pipeline.

50% cheaper than live API. Batches complete within 24h (usually <1h).
Use for training data generation where latency doesn't matter.

Usage:
    python -m rlaif.generate --batch --repo owner/repo
    python -m rlaif.generate --batch --repos repos.json
    python -m rlaif.generate --batch-collect <batch_id>
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rlaif.batch_provider")


def _sanitize_custom_id(custom_id: str) -> str:
    """Last-defence sanitization to guarantee Anthropic's custom_id pattern.

    Pattern: ^[a-zA-Z0-9_-]{1,64}$

    This is a safety-net inside create_batch() so that even if a caller
    builds custom_ids without going through _make_custom_id() in
    batch_orchestrator.py, the batch will never get a 400 from the API.
    Any mutation is logged as a WARNING so callers know to fix upstream.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", custom_id)[:64]
    if not safe:
        safe = "req"
    if safe != custom_id:
        logger.warning(
            "custom_id contained invalid chars and was sanitized: "
            "%r → %r (fix the caller to use _make_custom_id())",
            custom_id, safe,
        )
    return safe

# Batch API limits
MAX_REQUESTS_PER_BATCH = 100_000
POLL_INTERVAL_SECONDS = 30
MAX_POLL_HOURS = 24


def _get_client():
    """Lazy-load anthropic SDK (optional dependency)."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        raise ImportError(
            "anthropic SDK required for batch mode. "
            "Install: pip install anthropic"
        )


def create_batch(
    prompts: List[Dict[str, str]],
    model: str = "claude-sonnet-4-6",
    system_prompt: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> str:
    """Submit prompts to Anthropic Message Batches API.

    Args:
        prompts: list of {"custom_id": str, "content": str}
        model: Anthropic model name (no provider/ prefix)
        system_prompt: shared system prompt for all requests
        max_tokens: max tokens per response
        temperature: sampling temperature

    Returns:
        batch_id for polling/collection
    """
    client = _get_client()

    requests = []
    for p in prompts:
        params = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": p["content"]}],
        }
        if system_prompt:
            params["system"] = system_prompt
        requests.append({
            "custom_id": _sanitize_custom_id(p["custom_id"]),  # ← last-defence guard
            "params": params,
        })

    if len(requests) > MAX_REQUESTS_PER_BATCH:
        raise ValueError(
            f"Too many requests ({len(requests)}). "
            f"Max: {MAX_REQUESTS_PER_BATCH}"
        )

    batch = client.messages.batches.create(requests=requests)
    logger.info(
        f"Batch created: {batch.id} ({len(requests)} requests, "
        f"model={model})"
    )
    return batch.id


def poll_batch(batch_id: str, poll_interval: int = POLL_INTERVAL_SECONDS) -> str:
    """Poll until batch completes. Returns final status."""
    client = _get_client()
    deadline = time.time() + (MAX_POLL_HOURS * 3600)

    while time.time() < deadline:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        logger.info(
            f"Batch {batch_id}: {status} "
            f"(succeeded={counts.succeeded}, "
            f"errored={counts.errored}, "
            f"expired={counts.expired})"
        )

        if status == "ended":
            return status

        time.sleep(poll_interval)

    logger.error(f"Batch {batch_id} timed out after {MAX_POLL_HOURS}h")
    return "timeout"


def collect_results(batch_id: str) -> Tuple[Dict[str, str], int, int]:
    """Retrieve completed batch results.

    Returns:
        (results_map, succeeded_count, errored_count)
        results_map: {custom_id: response_text}
    """
    client = _get_client()
    results_map: Dict[str, str] = {}
    succeeded = 0
    errored = 0

    for entry in client.messages.batches.results(batch_id):
        cid = entry.custom_id
        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text
            results_map[cid] = text
            succeeded += 1
        else:
            logger.warning(f"Batch entry {cid} failed: {entry.result.type}")
            errored += 1

    logger.info(
        f"Batch {batch_id}: collected {succeeded} succeeded, "
        f"{errored} errored"
    )
    return results_map, succeeded, errored


def parse_batch_findings(results_map: Dict[str, str]) -> List[Dict]:
    """Parse JSON findings from batch results."""
    all_findings = []
    for custom_id, text in results_map.items():
        text = _strip_code_fences(text)
        try:
            data = json.loads(text)
            findings = data.get("findings", [])
            for f in findings:
                f["_batch_id"] = custom_id
            all_findings.extend(findings)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse batch result {custom_id}: {e}")
    return all_findings


def save_batch_state(
    output_dir: str, batch_id: str, repo: str, model: str, prompt_count: int
):
    """Save batch ID to disk so we can resume collection later."""
    state_path = os.path.join(output_dir, "pending_batches.json")
    pending = _load_pending(state_path)
    pending[batch_id] = {
        "repo": repo,
        "model": model,
        "prompt_count": prompt_count,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(state_path, "w") as f:
        json.dump(pending, f, indent=2)
    logger.info(f"Saved batch state: {batch_id} -> {state_path}")


def load_pending_batches(output_dir: str) -> Dict:
    """Load pending batch IDs from disk."""
    state_path = os.path.join(output_dir, "pending_batches.json")
    return _load_pending(state_path)


def remove_completed_batch(output_dir: str, batch_id: str):
    """Remove a completed batch from the pending state."""
    state_path = os.path.join(output_dir, "pending_batches.json")
    pending = _load_pending(state_path)
    pending.pop(batch_id, None)
    with open(state_path, "w") as f:
        json.dump(pending, f, indent=2)


def _load_pending(state_path: str) -> Dict:
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f)
    return {}


def _strip_code_fences(content: str) -> str:
    """Strip markdown code fences from model output."""
    text = content.strip()
    if not text.startswith("```"):
        return text
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
