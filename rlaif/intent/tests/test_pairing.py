from rlaif.intent.pairing import PROPAGATE, TOLERATE, label_fix


def site(function, callee, handled=None, ordinal=0, line=1):
    return {"file": "a.ts", "function": function, "callee": callee, "ordinal": ordinal,
            "handledBy": handled, "line": line, "source": f"function {function}() {{}}"}


def test_a_call_made_tolerant_is_positive_and_its_untouched_neighbour_a_hard_negative():
    before = [site("load", "readImpact"), site("load", "readSteps"), site("other", "readX")]
    after = [site("load", "readImpact"), site("load", "readSteps", "try"), site("other", "readX")]
    labels = {(e.function, e.callee): e.label for e in label_fix(before, after)}
    assert labels == {("load", "readSteps"): TOLERATE, ("load", "readImpact"): PROPAGATE}


def test_functions_the_fix_made_nothing_tolerant_in_are_not_labelled():
    before = [site("load", "a"), site("load", "b")]
    after = [site("load", "a"), site("load", "b")]
    assert label_fix(before, after) == []


def test_calls_already_handled_or_removed_by_the_fix_are_not_labelled():
    before = [site("load", "a", "call"), site("load", "gone"), site("load", "b")]
    after = [site("load", "a", "call"), site("load", "b", "call")]
    assert [(e.callee, e.label) for e in label_fix(before, after)] == [("b", TOLERATE)]


def test_occurrences_of_the_same_callee_pair_separately():
    before = [site("load", "get", ordinal=0), site("load", "get", ordinal=1)]
    after = [site("load", "get", ordinal=0), site("load", "get", "call", ordinal=1)]
    assert [(e.ordinal, e.label) for e in label_fix(before, after)] == [(0, PROPAGATE), (1, TOLERATE)]
