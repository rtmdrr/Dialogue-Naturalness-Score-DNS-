import pytest

from deep_persona_dns.turns import (
    Turn,
    jaccard,
    lcs_length,
    normalize_text,
    rouge_l,
    safe_divide,
    split_turns,
    split_verbal_and_actions,
    tokenize,
)


def test_normalize_folds_case_and_composes():
    assert normalize_text("CAFÉ") == normalize_text("café")


def test_tokenize_drops_digits_and_punctuation():
    assert tokenize("Hello, world! 42 times.") == ["hello", "world", "times"]


def test_tokenize_keeps_word_internal_marks():
    assert tokenize("don't") == ["don't"]
    assert tokenize('צה"ל') == ['צה"ל']


def test_split_separates_actions_and_collapses_whitespace():
    verbal, actions = split_verbal_and_actions("I'm fine  [avoids eye contact] really [smiles]")
    assert verbal == "I'm fine really"
    assert actions == ["avoids eye contact", "smiles"]


def test_split_handles_response_that_is_only_action():
    verbal, actions = split_verbal_and_actions("[shrugs]")
    assert verbal == ""
    assert actions == ["shrugs"]


def test_split_turns_preserves_order_and_both_channels():
    split = split_turns([Turn("hi", "hey [waves]"), Turn("how are you", "fine")])
    assert [t.verbal for t in split] == ["hey", "fine"]
    assert split[0].actions == ("waves",)
    assert split[1].actions == ()


def test_lcs_length_is_subsequence_not_substring():
    assert lcs_length(["a", "b", "c"], ["a", "x", "c"]) == 2


@pytest.mark.parametrize("measure", [rouge_l, jaccard])
def test_similarity_is_one_for_identical_text(measure):
    assert measure("the same words", "the same words") == pytest.approx(1.0)


@pytest.mark.parametrize("measure", [rouge_l, jaccard])
def test_similarity_is_zero_when_nothing_is_shared(measure):
    assert measure("alpha beta", "gamma delta") == pytest.approx(0.0)


@pytest.mark.parametrize("measure", [rouge_l, jaccard])
def test_similarity_is_zero_against_empty_text(measure):
    assert measure("", "something") == 0.0
    assert measure("something", "") == 0.0


def test_rouge_l_respects_order_where_jaccard_does_not():
    forward, backward = "a b c d", "d c b a"
    assert rouge_l(forward, backward) < jaccard(forward, backward)


def test_safe_divide_returns_zero_rather_than_raising():
    assert safe_divide(1.0, 0.0) == 0.0
    assert safe_divide(1.0, 4.0) == 0.25
