"""
DPO (Direct Preference Optimization) Data Formatter

Reads the SQLite training database and outputs JSONL files for fine-tuning:
  - sft_data.jsonl:  Supervised fine-tuning pairs (facts → verified findings)
  - dpo_data.jsonl:  DPO preference pairs (chosen=verified, rejected=dropped)

CRITICAL: The prompts here MUST match the inference prompts in
  rigour/packages/rigour-core/src/deep/prompts.ts
exactly. If the model is trained on different prompts than it sees at inference
time, it won't generalize. Any changes to prompts.ts must be mirrored here.

Usage:
    python -m rlaif.format_dpo --db rlaif/data/training_data.db --output rlaif/data/
    python -m rlaif.format_dpo --db rlaif/data/training_data.db --min-examples 100
"""

import os
import json
import sqlite3
import random
import argparse
import logging
import sys
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rlaif.format_dpo")


# ─── System prompt: MUST match DEEP_SYSTEM_PROMPT in prompts.ts ──────────────
# This is the exact system prompt the fine-tuned model sees at inference time.
# If you change this, you MUST also update prompts.ts (and vice versa).

SYSTEM_PROMPT = """You are an expert code reviewer and software architect performing deep quality analysis. You receive AST-extracted facts about a codebase and must identify quality issues, anti-patterns, and best practice violations.

IMPORTANT RULES:
1. ONLY report issues you can verify from the provided facts. Do NOT hallucinate files, classes, or functions.
2. Every finding MUST reference a real file and entity from the facts.
3. Be specific: include file paths, struct/class names, function names, line counts.
4. Assign confidence scores honestly: 0.9+ only for certain issues, 0.5-0.7 for probable issues.
5. Respond ONLY with valid JSON matching the schema below. No explanation text outside JSON.
6. AIM for 5-15 findings per batch. Be thorough — report ALL issues you can identify, not just the most obvious ones.
7. For Go code: treat structs as classes, receiver methods as class methods. Check Go idioms specifically.

OUTPUT SCHEMA:
{
  "findings": [
    {
      "category": "string (see CATEGORIES below)",
      "severity": "string (critical|high|medium|low|info)",
      "file": "string (exact file path from facts)",
      "line": "number or null",
      "description": "string (what the issue is, referencing specific entities)",
      "suggestion": "string (actionable fix recommendation)",
      "confidence": "number 0.0-1.0"
    }
  ]
}

CATEGORIES:
  SOLID Principles:
    srp_violation, ocp_violation, lsp_violation, isp_violation, dip_violation
  Design Patterns & Anti-patterns:
    god_class, god_function, feature_envy, shotgun_surgery, long_params,
    data_clump, inappropriate_intimacy, primitive_obsession, lazy_class,
    speculative_generality, refused_bequest
  DRY & Duplication:
    dry_violation, copy_paste_code
  Error Handling:
    error_inconsistency, empty_catch, error_swallowing, missing_error_check, panic_in_library
  Concurrency:
    race_condition, goroutine_leak, missing_context, channel_misuse, mutex_scope
  Testing:
    test_quality, test_coupling, missing_test, test_duplication
  Architecture:
    architecture, circular_dependency, package_cohesion, api_design, missing_abstraction
  Language Idioms:
    language_idiom, naming_convention, dead_code, magic_number
  Performance & Security:
    performance, resource_leak, hardcoded_config
  Code Smells:
    code_smell, complex_conditional, long_file"""


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_training_data(
    db_path: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Load verified, dropped, and retry-verified findings from SQLite.

    Returns (verified_pass1, dropped, verified_retry).
    verified_retry = findings that failed pass@1 but passed pass@2.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Pass@1 verified (excludes retry-verified)
    verified = [dict(r) for r in conn.execute(
        "SELECT * FROM training_data WHERE verified = 1 "
        "AND verification_notes NOT LIKE 'pass2:verified_retry%' "
        "ORDER BY category, repo"
    ).fetchall()]

    # Pass@2 retry-verified (separate confidence tier)
    retry_verified = [dict(r) for r in conn.execute(
        "SELECT * FROM training_data WHERE verified = 1 "
        "AND verification_notes LIKE 'pass2:verified_retry%' "
        "ORDER BY category, repo"
    ).fetchall()]

    # All dropped (includes pass@2 permanently rejected)
    dropped = [dict(r) for r in conn.execute(
        "SELECT * FROM training_data WHERE verified = 0 ORDER BY category, repo"
    ).fetchall()]

    conn.close()

    logger.info(
        f"Loaded {len(verified)} pass@1 verified, "
        f"{len(retry_verified)} pass@2 recovered, "
        f"{len(dropped)} dropped findings"
    )
    return verified, dropped, retry_verified


# ─── Finding Formatting ─────────────────────────────────────────────────────

def format_finding_as_json(row: Dict) -> str:
    """Format a DB row as a finding JSON string (matches inference output schema)."""
    finding = {
        "category": row["category"],
        "severity": row["severity"],
        "file": row["file_path"],
        "line": None,
        "description": row["description"],
        "suggestion": row["suggestion"],
        "confidence": row["confidence"],
    }
    return json.dumps(finding)


def format_findings_array(rows: List[Dict]) -> str:
    """Format multiple findings as a JSON findings array (matches inference output).

    At inference time, the model produces {"findings": [...]}.
    SFT training must teach this same format — not single findings.
    """
    findings = []
    for row in rows:
        findings.append({
            "category": row["category"],
            "severity": row["severity"],
            "file": row["file_path"],
            "line": None,
            "description": row["description"],
            "suggestion": row["suggestion"],
            "confidence": row["confidence"],
        })
    return json.dumps({"findings": findings})


def _get_facts_prompt(row: Dict) -> str:
    """Get the AST facts prompt for a row. Falls back to minimal context."""
    facts_prompt = row.get("facts_prompt") or ""
    if facts_prompt:
        return facts_prompt
    # Fallback for older data without facts_prompt stored
    return f"FILE: {row['file_path']} (repo: {row['repo']})"


# ─── SFT Pair Building ──────────────────────────────────────────────────────

def _group_by_facts_batch(rows: List[Dict]) -> List[List[Dict]]:
    """Group findings by their facts_prompt (i.e., the batch they were in).

    Findings with the same facts_prompt were analyzed together by the teacher.
    Grouping them teaches the model to produce multi-finding output per batch.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        key = row.get("facts_prompt") or row.get("facts_hash") or row["file_path"]
        groups[key].append(row)
    return list(groups.values())


def build_sft_pairs(verified: List[Dict]) -> List[Dict]:
    """Build SFT pairs using the EXACT same prompt format as inference.

    Key differences from old format:
    1. System prompt matches DEEP_SYSTEM_PROMPT from prompts.ts
    2. User prompt includes AST facts (not just file path)
    3. Assistant response uses {"findings": [...]} array format
    4. Findings are grouped by batch (model learns multi-finding output)
    """
    # Group by the facts batch they came from
    batches = _group_by_facts_batch(verified)

    pairs = []
    for batch in batches:
        if not batch:
            continue

        # Use the facts prompt from the first finding (all share the same batch)
        facts_prompt = _get_facts_prompt(batch[0])

        # Build the user prompt matching buildAnalysisPrompt() in prompts.ts
        user_prompt = (
            f"ANALYSIS FOCUS:\n"
            f"- SOLID principle violations (SRP, OCP, LSP, ISP, DIP)\n"
            f"- Design pattern anti-patterns: god class/struct, god function, feature envy, shotgun surgery\n"
            f"- DRY violations — duplicated logic, copy-paste code across files\n"
            f"- Error handling: inconsistencies, empty catches, swallowed errors\n"
            f"- Language-specific anti-patterns, naming convention violations, dead code\n"
            f"- Architecture: layer violations, circular dependencies, package cohesion\n"
            f"- Code smells: complex conditionals, magic numbers, long files\n"
            f"\nAST-EXTRACTED FACTS:\n{facts_prompt}\n\n"
            f"Analyze the codebase facts above. Identify ALL quality issues "
            f"matching the analysis focus areas. Be thorough — check every file "
            f"for every category. Return findings as JSON."
        )

        pair = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": format_findings_array(batch)},
            ]
        }
        pairs.append(pair)

    logger.info(
        f"Built {len(pairs)} SFT pairs "
        f"(from {len(verified)} findings across {len(pairs)} batches, "
        f"avg {len(verified) / max(1, len(pairs)):.1f} findings/batch)"
    )
    return pairs


# ─── DPO Pair Building ──────────────────────────────────────────────────────

def _group_by_category(rows: List[Dict]) -> Dict[str, List[Dict]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    return grouped


def _make_dpo_pair(
    category: str, chosen_row: Dict, dropped_row: Dict,
    weight: float = 1.0,
) -> Dict:
    """Build a DPO pair with the real facts prompt as context.

    The prompt includes AST facts so the model learns to evaluate
    findings IN CONTEXT of the actual code structure — not in isolation.
    """
    facts_prompt = _get_facts_prompt(dropped_row)

    prompt = (
        f"Analyze this codebase for quality issues.\n\n"
        f"AST-EXTRACTED FACTS:\n{facts_prompt}\n\n"
        f"Return findings as JSON."
    )
    pair = {
        "prompt": prompt,
        "chosen": format_finding_as_json(chosen_row),
        "rejected": format_finding_as_json(dropped_row),
        "category": category,
        "chosen_notes": "Passed structural verification",
        "rejected_notes": dropped_row.get("verification_notes", ""),
    }
    if weight != 1.0:
        pair["weight"] = weight
    return pair


def build_dpo_pairs(
    verified: List[Dict],
    dropped: List[Dict],
    max_pairs: int = 10000,
    retry_verified: List[Dict] = None,
    retry_weight: float = 0.6,
) -> List[Dict]:
    """Build DPO preference pairs by matching verified/dropped per category.

    Pass@2 retry-verified findings are included with lower weight (0.6x default)
    since they required a correction step and may carry subtle biases.
    """
    verified_by_cat = _group_by_category(verified)
    dropped_by_cat = _group_by_category(dropped)
    all_cats = set(verified_by_cat) | set(dropped_by_cat)

    pairs = []
    categories_used = set()

    # Pass@1 DPO pairs (full weight)
    for category in all_cats:
        v_list = verified_by_cat.get(category, [])
        d_list = dropped_by_cat.get(category, [])
        if not v_list or not d_list:
            continue
        categories_used.add(category)
        for dropped_row in d_list:
            chosen_row = random.choice(v_list)
            pairs.append(_make_dpo_pair(category, chosen_row, dropped_row))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break

    # Pass@2 retry-verified DPO pairs (lower weight)
    if retry_verified and len(pairs) < max_pairs:
        retry_by_cat = _group_by_category(retry_verified)
        retry_pairs = 0
        for category in retry_by_cat:
            rv_list = retry_by_cat[category]
            d_list = dropped_by_cat.get(category, [])
            if not d_list:
                continue
            categories_used.add(category)
            for dropped_row in d_list:
                chosen_row = random.choice(rv_list)
                pairs.append(_make_dpo_pair(
                    category, chosen_row, dropped_row,
                    weight=retry_weight,
                ))
                retry_pairs += 1
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break
        if retry_pairs > 0:
            logger.info(
                f"Added {retry_pairs} Pass@2 DPO pairs (weight={retry_weight})"
            )

    random.shuffle(pairs)
    logger.info(f"Built {len(pairs)} DPO pairs across {len(categories_used)} categories")
    return pairs


# ─── Stats & Export ──────────────────────────────────────────────────────────

def build_category_stats(
    verified: List[Dict],
    dropped: List[Dict],
    retry_verified: List[Dict] = None,
) -> Dict:
    """Build per-category statistics for reporting."""
    stats = defaultdict(lambda: {
        "verified": 0, "dropped": 0, "total": 0, "retry_verified": 0,
    })

    for row in verified:
        cat = row["category"]
        stats[cat]["verified"] += 1
        stats[cat]["total"] += 1

    for row in (retry_verified or []):
        cat = row["category"]
        stats[cat]["retry_verified"] += 1
        stats[cat]["verified"] += 1  # Also counts as verified
        stats[cat]["total"] += 1

    for row in dropped:
        cat = row["category"]
        stats[cat]["dropped"] += 1
        stats[cat]["total"] += 1

    for cat, s in stats.items():
        s["verification_rate"] = (
            round(s["verified"] / s["total"] * 100, 1) if s["total"] > 0 else 0
        )

    return dict(sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True))


def _write_jsonl(path: str, rows: List[Dict], label: str):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info(f"Wrote {len(rows)} {label} → {path}")


def _write_dataset_card(
    path: str, sft_count: int, dpo_count: int,
    total_verified: int, total_dropped: int,
    top_cats: list,
):
    rate = round(
        total_verified / max(1, total_verified + total_dropped) * 100, 1
    )
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cat_rows = "\n".join(
        f"| {cat} | {s['verified']} | {s['dropped']} "
        f"| {s['verification_rate']}% |"
        for cat, s in top_cats
    )

    card = f"""---
language: en
license: apache-2.0
task_categories:
  - text-generation
tags:
  - code-quality
  - rigour
  - rlaif
  - dpo
size_categories:
  - 1K<n<10K
---

# Rigour RLAIF Training Data

Training data for fine-tuning code quality analysis models.

## Stats

- **SFT pairs**: {sft_count} (batched multi-finding format)
- **DPO pairs**: {dpo_count}
- **Verified**: {total_verified}, **Dropped**: {total_dropped}
- **Rate**: {rate}%

## Top Categories

| Category | Verified | Dropped | Rate |
|----------|----------|---------|------|
{cat_rows}

## Format

SFT data uses the same prompt format as inference (DEEP_SYSTEM_PROMPT +
AST facts + multi-finding JSON output). DPO pairs include AST facts context.

## Usage with trl

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="dpo_data.jsonl")
```

Generated: {date_str} | Pipeline: rigour-labs/driftbench RLAIF
"""
    with open(path, "w") as f:
        f.write(card)
    logger.info(f"Wrote dataset card → {path}")


def export_for_huggingface(
    sft_pairs: List[Dict],
    dpo_pairs: List[Dict],
    output_dir: str,
    stats: Dict,
):
    """Export training data in HuggingFace-compatible format."""
    os.makedirs(output_dir, exist_ok=True)

    _write_jsonl(
        os.path.join(output_dir, "sft_data.jsonl"), sft_pairs, "SFT pairs"
    )
    _write_jsonl(
        os.path.join(output_dir, "dpo_data.jsonl"), dpo_pairs, "DPO pairs"
    )

    total_v = sum(s["verified"] for s in stats.values())
    total_d = sum(s["dropped"] for s in stats.values())
    top_cats = sorted(
        stats.items(), key=lambda x: x[1]["total"], reverse=True
    )[:10]

    _write_dataset_card(
        os.path.join(output_dir, "README.md"),
        len(sft_pairs), len(dpo_pairs), total_v, total_d, top_cats,
    )

    stats_path = os.path.join(output_dir, "category_stats.json")
    with open(stats_path, "w") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sft_pairs": len(sft_pairs),
            "dpo_pairs": len(dpo_pairs),
            "total_verified": total_v,
            "total_dropped": total_d,
            "categories": stats,
        }, f, indent=2)
    logger.info(f"Wrote stats → {stats_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

MIN_TRAINING_EXAMPLES = 50  # Fail hard if fewer verified findings than this


def main():
    parser = argparse.ArgumentParser(description="Format RLAIF data for DPO training")
    parser.add_argument("--db", type=str, default="rlaif/data/training_data.db",
                        help="Path to training data SQLite DB")
    parser.add_argument("--output", type=str, default="rlaif/data",
                        help="Output directory for JSONL files")
    parser.add_argument("--max-dpo-pairs", type=int, default=10000,
                        help="Maximum DPO pairs to generate")
    parser.add_argument("--min-examples", type=int, default=MIN_TRAINING_EXAMPLES,
                        help="Minimum verified findings required (fail if below)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        logger.error("Run 'python -m rlaif.generate' first to create training data")
        sys.exit(1)

    # Load data (3-way split: pass@1 verified, dropped, pass@2 recovered)
    verified, dropped, retry_verified = load_training_data(args.db)

    all_verified = verified + retry_verified
    total_findings = len(all_verified) + len(dropped)
    verification_rate = len(all_verified) / max(1, total_findings) * 100

    logger.info(
        f"Verification rate: {verification_rate:.1f}% "
        f"({len(all_verified)} verified / {total_findings} total)"
    )

    # Guard: fail hard if not enough training data
    if len(all_verified) < args.min_examples:
        logger.error(
            f"INSUFFICIENT TRAINING DATA: {len(all_verified)} verified findings "
            f"(minimum: {args.min_examples}). "
            f"Verification rate: {verification_rate:.1f}%. "
            f"Check: 1) teacher model producing findings? "
            f"2) verifier thresholds too strict? "
            f"3) training repos have enough code?"
        )
        sys.exit(1)

    if verification_rate < 10:
        logger.warning(
            f"LOW VERIFICATION RATE: {verification_rate:.1f}%. "
            f"Most teacher findings are being rejected. "
            f"Consider relaxing verifier thresholds or improving teacher prompts."
        )

    # Build pairs — SFT uses batched multi-finding format
    sft_pairs = build_sft_pairs(all_verified)
    dpo_pairs = build_dpo_pairs(
        verified, dropped,
        max_pairs=args.max_dpo_pairs,
        retry_verified=retry_verified,
    )
    stats = build_category_stats(verified, dropped, retry_verified)

    # Export
    export_for_huggingface(sft_pairs, dpo_pairs, args.output, stats)

    logger.info(f"\nDone! Files ready for upload to HuggingFace.")


if __name__ == "__main__":
    main()
