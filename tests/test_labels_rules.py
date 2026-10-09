import pytest

from bench.labels.rules import CLASSES, suggest


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
])
def test_suggestions_follow_the_decision_order(text, expected):
    assert suggest(text) == expected


def test_every_suggestion_is_a_known_class():
    assert {suggest(t) for t in ("nit", "bug", "ui", "cpu", "lint", "maybe")} == set(CLASSES)
