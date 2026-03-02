"""Structural verifier for RLAIF training data.

Mirrors the TypeScript verifier logic from rigour-core/src/deep/verifier.ts.
Validates teacher model findings against AST facts to filter false positives.
"""

import logging
from typing import Dict, Optional, Tuple

from .facts import FileFact

logger = logging.getLogger("rlaif.verifier")


def verify_finding(
    finding: Dict, facts_by_path: Dict[str, FileFact]
) -> Tuple[bool, str]:
    """Verify a single finding against AST facts.

    Returns (verified: bool, notes: str).
    """
    file_path = finding.get("file", "")
    category = finding.get("category", "")
    confidence = finding.get("confidence", 0.0)
    description = finding.get("description", "")

    facts = _resolve_file(file_path, facts_by_path)
    if not facts:
        return False, f"File not found: {file_path}"

    all_names = _collect_entity_names(facts)

    # Route to category-specific verification
    if category in _CLASS_CATEGORIES:
        return _verify_class_category(
            finding, facts, all_names, description, confidence
        )
    if category in _FUNCTION_CATEGORIES:
        return _verify_function_category(
            finding, facts, all_names, description, confidence
        )
    if category in _ENTITY_NAME_REQUIRED:
        return _verify_entity_name_required(
            all_names, description, confidence
        )
    if category in _STRUCTURAL_VERIFIERS:
        return _STRUCTURAL_VERIFIERS[category](
            finding, facts, facts_by_path, all_names, description, confidence
        )
    if category in _CONCURRENCY_CATEGORIES:
        return _verify_concurrency(facts, confidence)
    if category in _ERROR_CATEGORIES:
        return confidence >= 0.4, "Error handling check"
    if category in _TEST_CATEGORIES:
        return confidence >= 0.3 and facts.has_tests, "Test file check"
    if category == "missing_test":
        return _verify_missing_test(facts)
    if category == "long_file":
        return facts.line_count > 300, f"File is {facts.line_count} lines"
    if category == "magic_number":
        return facts.magic_numbers > 3, f"{facts.magic_numbers} magic numbers"
    if category in _CONFIDENCE_FLOOR_CATEGORIES:
        return confidence >= 0.5, f"Confidence floor: {confidence}"
    return confidence >= 0.5, f"Unknown category '{category}'"


# ── Category sets ──

_CLASS_CATEGORIES = {
    "god_class", "srp_violation", "ocp_violation",
    "lsp_violation", "isp_violation", "dip_violation",
}
_FUNCTION_CATEGORIES = {
    "god_function", "long_params", "complex_conditional",
}
_ENTITY_NAME_REQUIRED = {
    "lazy_class", "feature_envy", "primitive_obsession",
    "speculative_generality", "refused_bequest",
    "missing_abstraction", "api_design",
}
_CONCURRENCY_CATEGORIES = {
    "race_condition", "goroutine_leak", "missing_context",
    "channel_misuse", "mutex_scope",
}
_ERROR_CATEGORIES = {
    "empty_catch", "error_inconsistency", "error_swallowing",
    "missing_error_check", "panic_in_library",
}
_TEST_CATEGORIES = {
    "test_quality", "test_coupling", "test_duplication",
}
_CONFIDENCE_FLOOR_CATEGORIES = {
    "architecture", "package_cohesion", "code_smell", "language_idiom",
}


# ── Helpers ──

def _resolve_file(
    file_path: str, facts_by_path: Dict[str, FileFact]
) -> Optional[FileFact]:
    facts = facts_by_path.get(file_path)
    if facts:
        return facts
    for path, f in facts_by_path.items():
        if path.endswith(file_path) or file_path.endswith(path):
            return f
    return None


def _collect_entity_names(facts: FileFact) -> list:
    return (
        [c["name"] for c in facts.classes]
        + [s["name"] for s in (facts.structs or [])]
        + [fn["name"] for fn in facts.functions]
    )


def _find_entity(all_names: list, desc: str) -> Optional[str]:
    for name in sorted(all_names, key=len, reverse=True):
        if name in desc:
            return name
    return None


# ── Category verifiers ──

def _verify_class_category(
    finding: Dict, facts: FileFact, all_names: list,
    description: str, confidence: float,
) -> Tuple[bool, str]:
    category = finding["category"]
    entities = facts.classes + (facts.structs or [])
    if not entities:
        return False, "No classes or structs found"
    entity_name = _find_entity(all_names, description)
    if entity_name:
        entity = next((e for e in entities if e["name"] == entity_name), None)
        if not entity:
            return False, f"Entity '{entity_name}' not found"
        if category in ("god_class", "srp_violation"):
            mc = entity.get("methodCount", 0)
            lc = entity.get("lineCount", 0)
            if mc < 5 and lc < 200:
                return False, f"'{entity_name}' too small for god class"
        return True, f"'{entity_name}' verified"
    return confidence >= 0.4, "No entity name found"


def _verify_function_category(
    finding: Dict, facts: FileFact, all_names: list,
    description: str, confidence: float,
) -> Tuple[bool, str]:
    category = finding["category"]
    entity_name = _find_entity(all_names, description)
    if entity_name:
        func = next(
            (fn for fn in facts.functions if fn["name"] == entity_name), None
        )
        if not func:
            return False, f"Function '{entity_name}' not found"
        if category == "god_function" and func.get("lineCount", 0) < 30:
            return False, f"'{entity_name}' only {func['lineCount']} lines"
        if category == "long_params" and func.get("paramCount", 0) < 4:
            return False, f"'{entity_name}' only {func['paramCount']} params"
        return True, f"Function '{entity_name}' verified"
    return confidence >= 0.4, "No function name found"


def _verify_entity_name_required(
    all_names: list, description: str, confidence: float,
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if entity_name:
        return confidence >= 0.3, f"Entity '{entity_name}' exists"
    return confidence >= 0.6, "No entity name — requires high confidence"


def _verify_dead_code(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if not entity_name:
        return confidence >= 0.6, "No entity name for dead code"
    for path, other in facts_by_path.items():
        if path == facts.path:
            continue
        if any(entity_name in imp for imp in other.imports):
            return False, f"'{entity_name}' imported by {path}"
    return True, f"'{entity_name}' unreferenced — dead code confirmed"


def _verify_naming_convention(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if not entity_name:
        return confidence >= 0.6, "No entity name for naming check"
    if (facts.language == "go" and "_" in entity_name
            and entity_name != entity_name.upper()):
        return True, f"Go: '{entity_name}' uses snake_case"
    if facts.language == "python":
        is_func = any(fn["name"] == entity_name for fn in facts.functions)
        if (is_func and any(c.isupper() for c in entity_name)
                and not entity_name.startswith("_")):
            return True, f"Python function '{entity_name}' not snake_case"
    return False, f"'{entity_name}' follows conventions"


def _verify_hardcoded_config(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    if facts.has_tests:
        return False, "Test file — hardcoded values expected"
    if facts.magic_numbers > 2:
        return True, f"{facts.magic_numbers} magic numbers found"
    return confidence >= 0.6, "No magic numbers detected"


def _verify_data_clump(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    if len(facts.functions) < 2:
        return False, "Need 2+ functions for data clump"
    max_shared = 0
    for i, fn_a in enumerate(facts.functions):
        for fn_b in facts.functions[i + 1:]:
            params_a = {
                p.split(":")[0].strip().lower()
                for p in fn_a.get("params", [])
            }
            params_b = {
                p.split(":")[0].strip().lower()
                for p in fn_b.get("params", [])
            }
            max_shared = max(max_shared, len(params_a & params_b))
    if max_shared >= 3:
        return True, f"Functions share {max_shared} params"
    return False, f"Max shared params: {max_shared} (need 3+)"


def _verify_performance(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if entity_name:
        func = next(
            (fn for fn in facts.functions if fn["name"] == entity_name), None
        )
        if func and func.get("lineCount", 0) < 10:
            return False, f"'{entity_name}' only {func['lineCount']} lines"
        if func:
            return True, f"'{entity_name}' verified"
    return confidence >= 0.5 and facts.line_count > 50, "File-level check"


def _verify_circular_dependency(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    for path, other in facts_by_path.items():
        if path == facts.path:
            continue
        this_imp = any(
            path.replace("/", ".").replace("\\", ".") in imp or imp in path
            for imp in facts.imports
        )
        other_imp = any(
            facts.path.replace("/", ".").replace("\\", ".") in imp
            or imp in facts.path
            for imp in other.imports
        )
        if this_imp and other_imp:
            return True, f"Cycle: {facts.path} <-> {path}"
    return False, "No circular dependency found"


def _verify_dry_violation(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if not entity_name:
        return confidence >= 0.6, "No entity name for DRY check"
    func = next(
        (fn for fn in facts.functions if fn["name"] == entity_name), None
    )
    if not func:
        return False, f"No similar function for '{entity_name}'"
    for path, other in facts_by_path.items():
        if path == facts.path:
            continue
        for ofn in other.functions:
            if _functions_similar(func, ofn):
                return True, f"Similar to {ofn['name']} in {path}"
    return False, f"No similar function for '{entity_name}'"


def _functions_similar(func_a: Dict, func_b: Dict) -> bool:
    if func_a["name"] == func_b["name"]:
        return True
    param_diff = abs(
        func_a.get("paramCount", 0) - func_b.get("paramCount", 0)
    )
    line_diff = abs(
        func_a.get("lineCount", 0) - func_b.get("lineCount", 0)
    )
    return (param_diff <= 1 and line_diff <= 5
            and func_a.get("lineCount", 0) > 5)


def _verify_shotgun_surgery(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    entity_name = _find_entity(all_names, description)
    if not entity_name:
        return confidence >= 0.6, "No entity name"
    count = sum(
        1 for p, o in facts_by_path.items()
        if p != facts.path and any(entity_name in imp for imp in o.imports)
    )
    if count >= 4:
        return True, f"'{entity_name}' imported by {count} files"
    return False, f"Only imported by {count} files (need 4+)"


def _verify_inappropriate_intimacy(
    finding, facts, facts_by_path, all_names, description, confidence
) -> Tuple[bool, str]:
    for path, other in facts_by_path.items():
        if path == facts.path:
            continue
        this_imp = any(imp in path for imp in facts.imports)
        other_imp = any(imp in facts.path for imp in other.imports)
        if this_imp and other_imp:
            return True, f"Bidirectional import: {facts.path} <-> {path}"
    return False, "No bidirectional imports"


def _verify_concurrency(
    facts: FileFact, confidence: float
) -> Tuple[bool, str]:
    has_concurrency = (
        facts.goroutines > 0 or facts.channels > 0
        or facts.mutexes > 0
        or any(fn.get("isAsync") for fn in facts.functions)
    )
    if not has_concurrency:
        return False, "No concurrency constructs"
    return confidence >= 0.4, "Concurrency present"


def _verify_missing_test(facts: FileFact) -> Tuple[bool, str]:
    return (
        not facts.has_tests
        and facts.line_count > 50
        and len(facts.functions) > 1
    ), "Needs tests"


# ── Dispatch table for structural verifiers ──

_STRUCTURAL_VERIFIERS = {
    "dead_code": _verify_dead_code,
    "naming_convention": _verify_naming_convention,
    "hardcoded_config": _verify_hardcoded_config,
    "data_clump": _verify_data_clump,
    "performance": _verify_performance,
    "circular_dependency": _verify_circular_dependency,
    "dry_violation": _verify_dry_violation,
    "copy_paste_code": _verify_dry_violation,
    "shotgun_surgery": _verify_shotgun_surgery,
    "inappropriate_intimacy": _verify_inappropriate_intimacy,
}
