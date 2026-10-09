import re
from pathlib import Path

import pytest

from bench.labels.rules import CLASSES, suggest

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "LABELLING.md"
ROW_RE = re.compile(r"^\| \*\*(?P<label>[^*]+)\*\* \|[^|]*\| (?P<examples>.+) \|$", re.MULTILINE)


def guide_examples() -> list[tuple[str, str]]:
    return [(text, row["label"]) for row in ROW_RE.finditer(GUIDE.read_text())
            for text in re.findall(r'"([^"]+)"', row["examples"])]


@pytest.mark.parametrize("text, expected", [
    ("nit: this leaks nothing, just rename it", "mechanical"),
    ("This leaks the file handle on error", "claim/contract"),
    ("The error toast shows the raw exception", "user journey"),
    ("This allocates on every packet, which is expensive", "performance"),
    ("typo in the comment", "mechanical"),
    ("I'd keep this in the handler instead", "judgment"),
    ("The docs say inclusive but users see exclusive", "claim/contract"),  # decision order
    ("Can we talk about this tomorrow?", None),
    ("none of these matter", None),
    ("Imports aren't sorted:\n```suggestion\nimport \"example.com/types\"\n```", "mechanical"),
    ("A `nil` check here would be cleaner", "judgment"),
    ("unclosed fence ```go\nvar bug = types.X", None),
    ("This leaks the session token into the log", "claim/contract"),  # decision order: contract first
    ("The user's password is shown in the debug page", "security/privacy"),  # over user journey
    ("Unsanitized input reaches the query builder", "security/privacy"),
    ("O(n²) over every row", "performance"),
    ("dead code: only the tests call this", "mechanical"),
])
def test_suggestions_follow_the_decision_order(text, expected):
    assert suggest(text) == expected


def test_every_suggestion_is_a_known_class():
    assert {suggest(t) for t in ("nit", "bug", "ui", "cpu", "lint", "maybe", "secret")} == set(CLASSES)


def test_the_guide_has_examples_for_every_class():
    assert {label for _, label in guide_examples()} == set(CLASSES)


@pytest.mark.parametrize("text, label", guide_examples())
def test_every_guide_example_is_suggested_as_documented(text, label):
    assert suggest(text) == label
