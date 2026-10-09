import pytest

from bench.collect.text_rules import is_acknowledgement, is_command, is_review_point


@pytest.mark.parametrize("text", ["", None, "LGTM", "Thank you!", "lgtm, thanks 🚀", "+1", ":shipit:",
                                  "Looks good to me, thanks", "SGTM"])
def test_acknowledgements(text):
    assert is_acknowledgement(text)


@pytest.mark.parametrize("text", ["LGTM but please add a test", "why not reuse the helper?",
                                  "nit: typo", "thanks, but this leaks the file handle",
                                  "looks good to me thanks a lot for this work"])
def test_review_points_are_not_acknowledgements(text):
    assert not is_acknowledgement(text)


@pytest.mark.parametrize("text", ["/retest", "  \n/lgtm cancel", "@dependabot rebase", "@renovate-bot retry",
                                  "@mergebot[bot] merge"])
def test_commands(text):
    assert is_command(text) and not is_review_point(text)


@pytest.mark.parametrize("text", ["@alice why is this needed?", "path /usr/bin is wrong", "a/b should be b/a",
                                  "@botanist-team please look"])
def test_not_commands(text):
    assert not is_command(text)


def test_non_latin_script_counts_as_acknowledgement():
    """Documented limit of ACK-1: it assumes English-script reviews."""
    assert is_acknowledgement("この関数は空の入力で失敗します")
