"""Judged verdicts: which SZZ bugs are real, and which findings describe them.

SZZ credits a PR with every line a later fix changed, and proximity credits a
finding with any bug it lands near. Both over-count: on TanStack/router 28 of
65 fixes repaired nothing the blamed PR introduced, and most comments near a
real bug were about something else. A verdict file records one judgment per
fix and per (tool, PR, finding, fix) pair; judged scores count only real bugs
and findings that describe them.

    arena/verdicts/<owner>__<name>.json

Verdicts hold labels and our own short reasons, never third-party comment
text. `pack` lists what still needs judging, for agents or a person; `merge`
validates their answers into the file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from arena.score import Finding, hits, in_scope

REAL = {"yes", "partial"}
KINDS = {"yes", "partial", "no"}
NEEDS = {"local", "cross_file", "external_semantics", "intent", "execution", None}

INSTRUCTIONS = """\
Judge each fix in fixes.json. Read-only; no clone needed:
  gh api repos/<repo>/commits/<fix>                         the fix's diff
  gh api "repos/<repo>/contents/<path>?ref=<merge_sha>"     a blamed file as the PR merged it
  gh api repos/<repo>/pulls/<pr>                            the PR's description

Per fix, answer {"fix", "real", "needs", "reason"}:
- real: "yes" if a blamed PR introduced the defect the fix repairs; "no" if the fix
  only touched lines the PR refactored, renamed or moved, repaired tooling/config,
  or the defect predates the PR; "partial" if the PR made it worse or copied it.
- needs (for yes/partial): the least a reviewer needed at PR time: "local"
  (the changed function), "cross_file", "external_semantics" (library, platform,
  spec behaviour), "intent" (product intent), or "execution" (only running finds it).
- reason: one sentence, your words.

Per candidate, answer {"tool", "pr", "finding", "fix", "describes", "reason"}:
- describes: true only if the finding states this defect (not merely a nearby
  issue). CodeRabbit bodies: gh api repos/<repo>/pulls/comments/<finding>. Rigour
  findings carry their message. Do not copy comment text into reason.

Write one JSON array of answers per file, then: python -m arena judge-merge --repo <repo> <files>
"""


def pair_key(tool: str, pr: str, finding: str, fix: str) -> str:
    return f"{tool}|{pr}|{finding}|{fix}"


@dataclass
class Verdicts:
    fixes: dict[str, dict] = field(default_factory=dict)
    pairs: dict[str, dict] = field(default_factory=dict)
    #: Hand spot-check of the judge: {"fixes": {"checked", "agreed"}, "pairs": {...}}.
    spot_check: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Verdicts:
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(data.get("fixes", {}), data.get("pairs", {}), data.get("spot_check", {}))

    def save(self, path: Path, repo: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"repo": repo, "fixes": dict(sorted(self.fixes.items())), "pairs": dict(sorted(self.pairs.items())),
                "spot_check": self.spot_check}
        path.write_text(json.dumps(data, indent=1) + "\n")

    def is_real(self, fix: str) -> bool | None:
        verdict = self.fixes.get(fix)
        return None if verdict is None else verdict["real"] in REAL

    def describes(self, tool: str, pr: str, finding: str, fix: str) -> bool | None:
        verdict = self.pairs.get(pair_key(tool, pr, finding, fix))
        return None if verdict is None else verdict["describes"]


def merge(verdicts: Verdicts, answers: list[dict]) -> int:
    """Validate answers and add them; returns how many were added. Raises on a malformed answer."""
    for answer in answers:
        if "describes" in answer:
            missing = {"tool", "pr", "finding", "fix", "reason"} - answer.keys()
            if missing or not isinstance(answer["describes"], bool):
                raise ValueError(f"bad pair verdict {answer}: missing {sorted(missing)} or non-boolean describes")
            key = pair_key(answer["tool"], str(answer["pr"]), str(answer["finding"]), answer["fix"])
            verdicts.pairs[key] = {"describes": answer["describes"], "reason": answer["reason"]}
        else:
            if answer.get("real") not in KINDS or answer.get("needs") not in NEEDS or not answer.get("fix") or not answer.get("reason"):
                raise ValueError(f"bad fix verdict {answer}")
            verdicts.fixes[answer["fix"]] = {"real": answer["real"], "needs": answer["needs"], "reason": answer["reason"]}
    return len(answers)


def pack(labels: dict, results: dict[str, dict], verdicts: Verdicts, scope: str = "code",
         merges: dict[str, str] | None = None) -> list[dict]:
    """Fixes still to judge, and each tool's findings near a bug whose pair is unjudged.

    A fix judged not real needs no pair verdicts: it never counts. `merges` maps PR
    number to merge commit, so a judge can read the blamed file without a clone.
    """
    fixes: dict[str, dict] = {}
    for pr, bugs in labels["prs"].items():
        for bug in bugs:
            if not in_scope(bug["path"], scope) or verdicts.is_real(bug["fix_sha"]) is False:
                continue
            entry = fixes.setdefault(bug["fix_sha"], {"fix": bug["fix_sha"], "subject": bug["fix_subject"],
                                                      "judged": verdicts.is_real(bug["fix_sha"]) is not None,
                                                      "blamed": [], "candidates": []})
            entry["blamed"].append({"pr": pr, "merge_sha": (merges or {}).get(pr, ""), "path": bug["path"], "lines": bug["lines"]})
            for tool, result in results.items():
                for f in result["prs"].get(pr, {}).get("findings", []):
                    finding = Finding(f["path"], f["line"], id=str(f.get("id", "")), message=f.get("message", ""))
                    if (hits(finding, (bug["path"], bug["lines"]))
                            and verdicts.describes(tool, pr, finding.id, bug["fix_sha"]) is None):
                        entry["candidates"].append({"tool": tool, "pr": pr, "finding": finding.id, "path": finding.path,
                                                    "line": finding.line, "message": finding.message})
    return [f for f in fixes.values() if not f["judged"] or f["candidates"]]


def unjudged(labels: dict, results: dict[str, dict], verdicts: Verdicts, scope: str = "code") -> tuple[int, int]:
    """(fixes, pairs) still unjudged: a claim needs both at zero."""
    todo = pack(labels, results, verdicts, scope)
    return sum(1 for f in todo if not f["judged"]), sum(len(f["candidates"]) for f in todo)
