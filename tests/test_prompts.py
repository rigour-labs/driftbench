"""Which instructions each paid entrant ran, recorded by hash only: never Claude Code's text."""
from __future__ import annotations

import hashlib
import json

import pytest

from bench.harness import prompts


def fake_npm(calls: list):
    def run(args):
        calls.append(args)
        if args[:2] == ["npm", "view"]:
            return f"sha512-{args[2]}"
        return "rigour-prompt-sha256"                     # node printing PROMPT_VERSION
    return run


def test_both_entrants_are_pinned_by_package_integrity_and_neither_text_is_kept(tmp_path):
    calls = []
    record = prompts.prompt_record(["claude-code-review", "rigour-reviewer"], tmp_path, fake_npm(calls))
    cc = record["claude-code-review"]
    assert cc["command"] == "/code-review" and cc["package"].endswith("@2.1.285")
    assert cc["package"].startswith("@anthropic-ai/claude-code-") and cc["integrity"] == f"sha512-{cc['package']}"
    assert record["rigour-reviewer"] == {"package": "@rigour-labs/core@6.12.1", "integrity": "sha512-@rigour-labs/core@6.12.1",
                                         "prompt_version": "rigour-prompt-sha256"}
    assert "never extracted" in cc["text"] and len(json.dumps(record)) < 1000
    assert calls[-1][:2] == ["node", "--input-type=module"] and "prompt.js" in calls[-1][-1]


def test_rigour_needs_its_installed_core_and_free_entrants_record_nothing():
    with pytest.raises(ValueError, match="--rigour-core"):
        prompts.prompt_record(["rigour-reviewer"], None, fake_npm([]))
    assert prompts.prompt_record(["rigour"], None, fake_npm([])) == {}
    with pytest.raises(ValueError, match="no npm integrity"):
        prompts.integrity("x@1", lambda args: "")


def test_the_native_package_matches_the_runner_platform():
    assert prompts.native_package("Linux", "x86_64") == "@anthropic-ai/claude-code-linux-x64"
    assert prompts.native_package("Darwin", "arm64") == "@anthropic-ai/claude-code-darwin-arm64"


def test_installed_files_are_hashed_from_the_global_install_and_the_npx_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "native_package", lambda: "@anthropic-ai/claude-code-linux-x64")
    native = tmp_path / "g" / "@anthropic-ai" / "claude-code" / "node_modules" / "@anthropic-ai" / "claude-code-linux-x64"
    native.mkdir(parents=True)
    (native / "claude").write_bytes(b"binary")
    for version in ("6.12.0", "6.12.1"):
        core = tmp_path / "npx" / version / "node_modules" / "@rigour-labs" / "core"
        (core / prompts.PROMPT_JS).parent.mkdir(parents=True)
        (core / "package.json").write_text(json.dumps({"version": version}))
        (core / prompts.PROMPT_JS).write_text(f"prompt {version}")
    found = prompts.installed(tmp_path / "g", tmp_path / "npx")
    assert found["claude-code-review"] == {"claude": hashlib.sha256(b"binary").hexdigest()}
    assert found["rigour-reviewer"] == {str(prompts.PROMPT_JS): hashlib.sha256(b"prompt 6.12.1").hexdigest()}
