"""AST fact extraction for RLAIF pipeline.

Extracts structured facts (classes, functions, imports, etc.) from source files.
Python reimplementation of the TypeScript fact-extractor in rigour-core.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("rlaif.facts")

MAX_FILES_PER_REPO = 200
MAX_FILE_SIZE = 50_000  # chars
SUPPORTED_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx", ".py", ".go", ".rs",
    ".cs", ".java", ".rb", ".kt",
}
SKIP_DIRS = {
    "node_modules", "dist", "build", ".git", "vendor",
    "__pycache__", ".tox", "venv", ".venv", "env",
    ".next", ".nuxt", "coverage", ".nyc_output",
}


@dataclass
class FileFact:
    """Mirrors the TypeScript FileFacts interface."""

    path: str
    language: str
    line_count: int
    classes: List[Dict]
    functions: List[Dict]
    imports: List[str]
    exports: List[str]
    error_handling: List[Dict]
    test_assertions: int
    has_tests: bool
    structs: Optional[List[Dict]] = None
    interfaces: Optional[List[Dict]] = None
    goroutines: int = 0
    channels: int = 0
    defers: int = 0
    mutexes: int = 0
    magic_numbers: int = 0
    comment_ratio: float = 0.0
    todo_count: int = 0


def detect_language(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    lang_map = {
        ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".jsx": "javascript",
        ".py": "python", ".go": "go", ".rs": "rust",
        ".cs": "csharp", ".java": "java", ".rb": "ruby", ".kt": "kotlin",
    }
    return lang_map.get(ext, "unknown")


def _extract_classes(content: str, lang: str) -> List[Dict]:
    classes = []
    lines = content.split("\n")
    pattern = (
        re.compile(r'^\s*class\s+(\w+)') if lang == "python"
        else re.compile(r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)')
    )
    for i, line in enumerate(lines):
        m = pattern.match(line) if lang == "python" else pattern.search(line)
        if not m:
            continue
        name = m.group(1)
        end = _find_block_end(lines, i, lang)
        class_content = "\n".join(lines[i:end])
        methods = _find_methods_in_class(class_content, name, lang)
        classes.append({
            "name": name, "lineStart": i + 1, "lineEnd": end,
            "methodCount": len(methods), "methods": methods[:20],
            "lineCount": end - i,
        })
    return classes


def _find_block_end(lines: List[str], start: int, lang: str) -> int:
    """Find where a class/function block ends."""
    end = min(start + 500, len(lines))
    if lang == "python":
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        for j in range(start + 1, end):
            indent = len(lines[j]) - len(lines[j].lstrip())
            if lines[j].strip() and indent <= base_indent:
                return j
    else:
        braces, started = 0, False
        for j in range(start, end):
            braces += lines[j].count("{") - lines[j].count("}")
            if lines[j].count("{") > 0:
                started = True
            if started and braces <= 0:
                return j + 1
    return end


def _find_methods_in_class(
    class_content: str, class_name: str, lang: str
) -> List[str]:
    pattern = (
        re.compile(r'^\s+def\s+(\w+)', re.MULTILINE) if lang == "python"
        else re.compile(
            r'(?:public|private|protected|static|async)?\s*'
            r'(?:async\s+)?(\w+)\s*\(', re.MULTILINE
        )
    )
    skip = (class_name, "constructor", "if", "for", "while")
    return [m.group(1) for m in pattern.finditer(class_content)
            if m.group(1) not in skip]


def _extract_functions(content: str, lang: str) -> List[Dict]:
    functions = []
    lines = content.split("\n")
    patterns = _get_function_patterns(lang)

    for i, line in enumerate(lines):
        for pattern in patterns:
            m = (pattern.match(line) if lang in ("python", "go")
                 else pattern.search(line))
            if not m:
                continue
            parsed = _parse_function_match(m, line, lang)
            if not parsed:
                continue
            name, params = parsed
            end = _find_func_end(lines, i, lang)
            line_count = end - i
            functions.append({
                "name": name, "lineStart": i + 1, "lineEnd": end,
                "lineCount": line_count, "paramCount": len(params),
                "params": params[:10], "isAsync": "async" in line,
                "isExported": (name[0].isupper() if lang == "go"
                               else "export" in line),
            })
            break
    return functions


def _get_function_patterns(lang: str) -> list:
    if lang == "python":
        return [re.compile(r'^(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)')]
    if lang == "go":
        return [re.compile(
            r'^func\s+(?:\(\s*\w+\s+\*?(\w+)\s*\)\s+)?(\w+)\s*\(([^)]*)\)'
        )]
    return [
        re.compile(
            r'(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)'
        ),
        re.compile(
            r'(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*'
            r'(?:async\s+)?\([^)]*\)\s*=>'
        ),
    ]


def _parse_function_match(m, line: str, lang: str):
    """Extract name and params from a regex match. Returns None to skip."""
    if lang == "go":
        receiver = m.group(1) or ""
        name = f"{receiver}.{m.group(2)}" if receiver else m.group(2)
        param_str = m.group(3) or ""
    else:
        name = m.group(1)
        param_str = m.group(2) if m.lastindex >= 2 else ""
    if name in ("if", "for", "while", "switch"):
        return None
    params = ([p.strip() for p in param_str.split(",") if p.strip()]
              if param_str else [])
    return name, params


def _find_func_end(lines: List[str], start: int, lang: str) -> int:
    if lang == "python":
        base_indent = len(lines[start]) - len(lines[start].lstrip())
        end = start + 1
        for j in range(start + 1, min(len(lines), start + 300)):
            if (lines[j].strip() and
                    (len(lines[j]) - len(lines[j].lstrip())) <= base_indent):
                break
            end = j + 1
        return end
    braces, started = 0, False
    for j in range(start, min(len(lines), start + 300)):
        braces += lines[j].count("{") - lines[j].count("}")
        if lines[j].count("{") > 0:
            started = True
        if started and braces <= 0:
            return j + 1
    return start + 1


def _extract_imports(content: str, lang: str) -> List[str]:
    imports = []
    if lang == "python":
        for m in re.finditer(
            r'^(?:from\s+(\S+)\s+)?import\s+(.+)$', content, re.MULTILINE
        ):
            imports.append(m.group(1) or m.group(2).strip())
    else:
        for m in re.finditer(
            r"import\s+.+?from\s+['\"](.+?)['\"]", content
        ):
            imports.append(m.group(1))
        for m in re.finditer(
            r"require\s*\(\s*['\"](.+?)['\"]\s*\)", content
        ):
            imports.append(m.group(1))
    return imports


def _extract_exports(content: str, lang: str) -> List[str]:
    if lang in ("typescript", "javascript"):
        return [m.group(1) for m in re.finditer(
            r'export\s+(?:default\s+)?'
            r'(?:class|function|const|let|var|interface|type|enum)\s+(\w+)',
            content,
        )]
    return []


def _extract_go_structs(content: str) -> List[Dict]:
    structs = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r'^type\s+(\w+)\s+struct\s*\{', line)
        if not m:
            continue
        name = m.group(1)
        end = _find_block_end(lines, i, "go")
        method_pat = re.compile(
            rf'^func\s*\(\s*\w+\s+\*?{re.escape(name)}\s*\)\s+(\w+)\s*\(',
            re.MULTILINE,
        )
        methods = [mm.group(1) for mm in method_pat.finditer(content)]
        structs.append({
            "name": name, "lineStart": i + 1, "lineEnd": end,
            "methodCount": len(methods), "methods": methods[:20],
            "lineCount": end - i,
        })
    return structs


def _extract_go_interfaces(content: str) -> List[Dict]:
    interfaces = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r'^type\s+(\w+)\s+interface\s*\{', line)
        if not m:
            continue
        name = m.group(1)
        methods, braces, started = [], 0, False
        for j in range(i, min(len(lines), i + 100)):
            braces += lines[j].count("{") - lines[j].count("}")
            if lines[j].count("{") > 0:
                started = True
            if j > i and braces > 0:
                mm = re.match(r'^\s+(\w+)\s*\(', lines[j])
                if mm:
                    methods.append(mm.group(1))
            if started and braces <= 0:
                break
        interfaces.append({
            "name": name, "lineStart": i + 1,
            "methodCount": len(methods), "methods": methods,
        })
    return interfaces


def _is_test_file(filepath: str, content: str) -> bool:
    if re.search(r'\.(test|spec|_test)\.', filepath):
        return True
    if any(d in filepath for d in ("__tests__", "test/", "tests/")):
        return True
    if "describe(" in content or "it(" in content or "test(" in content:
        return True
    if "def test_" in content or "@pytest" in content:
        return True
    return False


def extract_facts_from_file(
    filepath: str, rel_path: str
) -> Optional[FileFact]:
    """Extract structured facts from a single source file."""
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    if len(content) > MAX_FILE_SIZE or len(content.split("\n")) < 5:
        return None

    lang = detect_language(filepath)
    if lang == "unknown":
        return None

    lines = content.split("\n")
    assertions = len(re.findall(
        r'\b(?:expect|assert\w*|should\.)\s*[.(]', content
    ))
    allowed_nums = {
        "0", "1", "-1", "2", "100", "200", "201", "204",
        "301", "302", "400", "401", "403", "404", "500",
    }
    magic = re.findall(r'(?<![.\w])\d{2,}(?![.\w])', content)
    magic_count = len([m for m in magic if m not in allowed_nums])

    fact = FileFact(
        path=rel_path, language=lang, line_count=len(lines),
        classes=_extract_classes(content, lang),
        functions=_extract_functions(content, lang),
        imports=_extract_imports(content, lang),
        exports=_extract_exports(content, lang),
        error_handling=[], test_assertions=assertions,
        has_tests=_is_test_file(rel_path, content),
        magic_numbers=magic_count,
    )

    if lang == "go":
        fact.structs = _extract_go_structs(content)
        fact.interfaces = _extract_go_interfaces(content)
        fact.goroutines = len(re.findall(r'\bgo\s+\w+', content))
        fact.channels = len(re.findall(
            r'\bchan\b|make\s*\(\s*chan\b|<-', content
        ))
        fact.defers = len(re.findall(r'\bdefer\b', content))
        fact.mutexes = len(re.findall(
            r'sync\.(?:Mutex|RWMutex|WaitGroup)', content
        ))
    return fact


def extract_repo_facts(repo_path: str) -> List[FileFact]:
    """Walk a repository and extract facts from all supported files."""
    facts = []
    count = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if count >= MAX_FILES_PER_REPO:
                break
            if Path(fname).suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_path)
            fact = extract_facts_from_file(full_path, rel_path)
            if fact:
                facts.append(fact)
                count += 1
        if count >= MAX_FILES_PER_REPO:
            break
    logger.info(f"Extracted facts from {len(facts)} files in {repo_path}")
    return facts


def facts_to_prompt(facts: List[FileFact], max_chars: int = 12000) -> str:
    """Convert facts to compact prompt string."""
    parts = []
    for f in facts:
        lines = [f"FILE: {f.path} ({f.language}, {f.line_count} lines)"]
        for cls in f.classes:
            methods_str = ", ".join(cls["methods"][:10])
            lines.append(
                f"  CLASS {cls['name']} "
                f"({cls['lineCount']} lines, "
                f"{cls['methodCount']} methods: {methods_str})"
            )
        if f.structs:
            for s in f.structs:
                methods_str = ", ".join(s["methods"][:10])
                lines.append(
                    f"  STRUCT {s['name']} "
                    f"({s['lineCount']} lines, "
                    f"{s['methodCount']} methods: {methods_str})"
                )
        if f.interfaces:
            for iface in f.interfaces:
                methods_str = ", ".join(iface["methods"][:10])
                lines.append(
                    f"  INTERFACE {iface['name']} "
                    f"({iface['methodCount']} methods: {methods_str})"
                )
        for fn in f.functions:
            if fn["lineCount"] < 8:
                continue
            flags = []
            if fn.get("isAsync"):
                flags.append("async")
            if fn.get("isExported"):
                flags.append("exported")
            params_str = ", ".join(fn["params"][:5])
            flag_str = f", {', '.join(flags)}" if flags else ""
            lines.append(
                f"  FN {fn['name']}({params_str}) "
                f"[{fn['lineCount']} lines{flag_str}]"
            )
        if f.imports:
            imp_str = ", ".join(f.imports[:8])
            ellipsis = "..." if len(f.imports) > 8 else ""
            lines.append(
                f"  IMPORTS: {len(f.imports)} ({imp_str}{ellipsis})"
            )
        parts.append("\n".join(lines))
        if sum(len(p) for p in parts) > max_chars:
            break
    return "\n\n".join(parts)
