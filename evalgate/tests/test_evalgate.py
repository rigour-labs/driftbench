import re

from evalgate.metrics import Prediction, gate, report
from evalgate.prompt import build_prompt, parse_answer
from evalgate.sanitize import sanitize_patch
from evalgate.scenarios import load_items

PATCH = """--- /dev/null
+++ b/app/db.py
@@ -0,0 +1,6 @@
+import sqlite3
+# SECURITY DRIFT: SQL Injection vulnerability!
+def find(name):
+    url = "http://example.com/#anchor"  # LOGIC DRIFT: hardcoded
+    return db.execute(f"SELECT * FROM t WHERE n = '{name}'")  // injection
+    tag = "#not-a-comment"
"""


def test_sanitize_removes_comment_lines_and_trailing_comments_only():
    out = sanitize_patch(PATCH)
    assert "DRIFT" not in out and "injection" not in out
    assert '+    url = "http://example.com/#anchor"\n' in out
    assert '+    tag = "#not-a-comment"\n' in out
    assert "+++ b/app/db.py" in out and "+def find(name):" in out


def test_sanitize_drops_docstrings_on_either_side():
    patch = '+def find(name):\n+    """Uses the ORM to prevent SQL injection."""\n+    q = 1\n+    """\n+    Safe: parameterised.\n+    """\n+    return q\n'
    assert sanitize_patch(patch) == "+def find(name):\n+    q = 1\n+    return q\n"


def test_real_dataset_is_balanced_and_no_patch_names_its_answer():
    items = load_items()
    assert sum(i.has_drift for i in items) == sum(not i.has_drift for i in items) == 27
    tell = re.compile(r"drift|vulnerab|injection", re.IGNORECASE)
    leaks = [i.id for i in items if any(tell.search(l) for l in i.patch.splitlines() if l.startswith("+") and not l.startswith("+++"))]
    assert leaks == [], f"patches that still name their answer: {leaks}"


def test_prompt_never_states_the_drift_type():
    item = next(i for i in load_items() if i.has_drift)
    prompt = build_prompt(item)
    assert "drift_type" not in prompt and "identified" not in prompt
    assert item.intent in prompt


def test_parse_is_strict_and_never_defaults():
    assert parse_answer('```json\n{"is_drift": true, "confidence": 0.8}\n```') is True
    assert parse_answer('{"is_drift": false}') is False
    assert parse_answer('{"is_drift": "yes"}') is None
    assert parse_answer("I think this is drift.") is None


def _preds(answers):
    expected = [True, True, False, False]
    return [Prediction(str(i), e, a) for i, (e, a) in enumerate(zip(expected, answers))]


def test_the_gate_rejects_constant_and_unparseable_models_and_passes_a_good_one():
    assert "gives the same answer to every item" in gate(report(_preds([True, True, True, True])))
    assert "gives the same answer to every item" in gate(report(_preds([False, False, False, False])))
    assert any("unparseable" in f for f in gate(report(_preds([True, None, False, None]))))
    assert gate(report(_preds([True, True, False, False]))) == []


def test_unparseable_replies_count_as_wrong():
    r = report(_preds([None, True, None, False]))
    assert (r.tp, r.fn, r.tn, r.fp, r.unparseable) == (1, 1, 1, 1, 2)
