"""The free Rigour entrant on another version or config: diagnostic runs only, fixed in run.json."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from bench.adapters.rigour import VERSION, RigourDeterministic, command
from bench.harness import override

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "rigour-security-block.yml"


def args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"rigour_version": None, "rigour_config": None, "diagnostic": None, **kw})


class Request:
    base_sha = "b" * 40


def test_the_override_needs_a_diagnostic_run_and_a_real_config():
    with pytest.raises(ValueError, match="diagnostic runs only"):
        override.check(args(rigour_version="6.14.0-rc.1"))
    with pytest.raises(ValueError, match="no such file"):
        override.check(args(rigour_config=Path("/none.yml"), diagnostic="x"))
    override.check(args(rigour_version="6.14.0-rc.1", rigour_config=CONFIG, diagnostic="x"))


def test_only_the_free_rigour_entrant_changes_and_run_json_records_it():
    other = type("Other", (), {"name": "no-tool"})()
    chosen = override.apply(args(rigour_version="6.14.0-rc.1", rigour_config=CONFIG, diagnostic="x"),
                            [RigourDeterministic(), other])
    assert chosen[0].version == "6.14.0-rc.1" and chosen[0].config == CONFIG.resolve() and chosen[1] is other
    assert command(Request(), chosen[0].version, chosen[0].config)[-2:] == ["-c", str(CONFIG.resolve())]
    assert command(Request())[2] == f"@rigour-labs/cli@{VERSION}" and "-c" not in command(Request())
    recorded = override.record(args(rigour_config=CONFIG, diagnostic="x"))
    assert recorded["version"] == VERSION and len(recorded["config_sha256"]) == 64
    assert override.record(args()) is None and override.apply(args(), [other]) == [other]
