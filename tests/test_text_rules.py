import pytest

from bench.collect.text_rules import is_acknowledgement


@pytest.mark.parametrize("text", ["", None, "LGTM", "Thank you!", "lgtm, thanks 🚀", "+1", ":shipit:",
                                  "Looks good to me, thanks", "SGTM"])
def test_acknowledgements(text):
    assert is_acknowledgement(text)


@pytest.mark.parametrize("text", ["LGTM but please add a test", "why not reuse the helper?",
                                  "nit: typo", "thanks, but this leaks the file handle",
                                  "looks good to me thanks a lot for this work"])
def test_review_points_are_not_acknowledgements(text):
    assert not is_acknowledgement(text)
