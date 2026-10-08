from bench.points.split import is_quote, split_spans


def pieces(text: str) -> list[str]:
    return [text[s:e] for s, e in split_spans(text)]


def test_paragraphs_and_list_items():
    text = "First point here.\nStill first.\n\n- second\n- third\n  continued\n\n1. fourth\n"
    assert pieces(text) == ["First point here.\nStill first.", "- second", "- third\n  continued", "1. fourth"]


def test_fenced_code_stays_with_its_paragraph():
    text = "Use this instead:\n```go\nx := 1\n\ny := 2\n```\n\nAlso rename it."
    assert pieces(text) == ["Use this instead:\n```go\nx := 1\n\ny := 2\n```", "Also rename it."]


def test_spans_index_the_original_text():
    text = "\n\n  lead\r\n\r\ntail  "
    assert pieces(text) == ["  lead", "tail"]


def test_empty_text_has_no_points():
    assert split_spans("") == [] and split_spans("\n \n") == []


def test_quotes():
    assert is_quote("> you wrote this\n> and this")
    assert not is_quote("> you wrote this\nbut I disagree")
