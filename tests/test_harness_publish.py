from bench.harness.publish import MESSAGE_LIMIT, published_finding, short_message

TOOL_MESSAGE = ("[naming] Function name is not snake_case, rename it to match the module. "
                'Found: "def ParseConfig(path):" in config.py')


def test_quoted_code_never_survives():
    short = short_message(TOOL_MESSAGE)
    assert "ParseConfig" not in short and "Found" not in short and len(short) <= MESSAGE_LIMIT
    assert short.startswith("[naming] Function name is not snake_case")
    assert short_message("Use `os.path.join(a, b)` here, and 'x = 1' too") == "Use here, and too"
    assert short_message('"leaked"') == ""


def test_long_messages_are_cut_and_other_fields_kept():
    finding = {"path": "a.py", "line": 3, "end_line": None, "blocking": True, "rule": "r", "message": "word " * 50}
    published = published_finding(finding)
    assert len(published["message"]) <= MESSAGE_LIMIT
    assert {k: v for k, v in published.items() if k != "message"} == {k: v for k, v in finding.items()
                                                                      if k != "message"}
