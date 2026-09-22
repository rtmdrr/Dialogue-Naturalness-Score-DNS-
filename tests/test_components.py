import pytest

from deep_persona_dns.components import (
    affective_congruence_score,
    emotional_expression_score,
    joint_attention_score,
    pragmatic_score,
)
from deep_persona_dns.turns import Turn, split_turns

from conftest import EMOTION_LEXICON, INTENSIFIERS


# --------------------------------------------------------------- pragmatics

def test_echoing_scores_far_below_varied(varied_dialogue, echoing_dialogue):
    varied = pragmatic_score(split_turns(varied_dialogue)).score
    echoing = pragmatic_score(split_turns(echoing_dialogue)).score
    assert varied > echoing
    assert echoing < 0.5


def test_verbatim_echo_is_detected_as_echolalia(echoing_dialogue):
    assert pragmatic_score(split_turns(echoing_dialogue)).echolalia_rate == 1.0


def test_self_repetition_rises_when_the_agent_repeats_itself():
    repeated = split_turns([Turn("a", "i really like cheese")] * 3)
    varied = split_turns([
        Turn("a", "i really like cheese"),
        Turn("b", "the weather turned cold"),
        Turn("c", "my train leaves at eight"),
    ])
    assert pragmatic_score(repeated).self_repetition > pragmatic_score(varied).self_repetition


def test_self_repetition_window_bounds_how_far_back_it_looks():
    turns = split_turns(
        [Turn("a", "the very same sentence")]
        + [Turn("b", f"a different sentence number {i}") for i in range(4)]
        + [Turn("c", "the very same sentence")]
    )
    near = pragmatic_score(turns, self_repetition_window=5).self_repetition
    far = pragmatic_score(turns, self_repetition_window=2).self_repetition
    assert near > far


def test_weights_must_sum_to_one(varied_dialogue):
    with pytest.raises(ValueError, match="sum to 1"):
        pragmatic_score(split_turns(varied_dialogue), alpha=0.5, beta=0.5, gamma=0.5)


def test_empty_dialogue_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        pragmatic_score([])


def test_clamp_keeps_the_score_non_negative(echoing_dialogue):
    clamped = pragmatic_score(split_turns(echoing_dialogue), gamma=1.0, alpha=0.0, beta=0.0)
    assert clamped.score >= 0.0


def test_unknown_overlap_measure_is_rejected(varied_dialogue):
    with pytest.raises(ValueError, match="overlap measure"):
        pragmatic_score(split_turns(varied_dialogue), overlap_measure="cosine")


# ----------------------------------------------------------- joint attention

def test_acknowledged_entity_scores_one(extractor):
    turns = split_turns([
        Turn("i have a border collie", "how old is she?"),
        Turn("three", "a border collie needs a lot of exercise"),
    ])
    assert joint_attention_score(turns, extractor).score == 1.0


def test_ignored_entity_scores_zero(extractor):
    turns = split_turns([
        Turn("i have a border collie", "anyway, about the weather"),
        Turn("sure", "it has been raining"),
    ])
    assert joint_attention_score(turns, extractor).score == 0.0


def test_window_next_ignores_acknowledgement_in_the_same_turn(extractor):
    turns = split_turns([
        Turn("i have a border collie", "a border collie is hard work"),
        Turn("she is", "mm"),
    ])
    assert joint_attention_score(turns, extractor).score == 0.0
    permissive = joint_attention_score(
        turns, extractor, acknowledgement_window="current_or_next"
    )
    assert permissive.score == 1.0


def test_an_entity_counts_only_the_first_time_it_appears(extractor):
    turns = split_turns([
        Turn("i have a border collie", "lovely"),
        Turn("the border collie is asleep", "let her rest"),
    ])
    assert joint_attention_score(turns, extractor).n_new_turns == 1


def test_no_new_entities_gives_zero_over_zero(extractor):
    turns = split_turns([Turn("mm", "quite"), Turn("indeed", "yes")])
    result = joint_attention_score(turns, extractor)
    assert result.n_new_turns == 0
    assert result.score == 0.0


def test_unknown_window_is_rejected(extractor, varied_dialogue):
    with pytest.raises(ValueError, match="acknowledgement_window"):
        joint_attention_score(
            split_turns(varied_dialogue), extractor, acknowledgement_window="eventually"
        )


# ------------------------------------------------------ emotional expression

def _score(turns, **kwargs):
    return emotional_expression_score(
        split_turns(turns), EMOTION_LEXICON, INTENSIFIERS, **kwargs
    )


def test_affective_language_scores_above_neutral_language():
    affective = _score([Turn("a", "i am very happy and deeply in love")])
    neutral = _score([Turn("a", "the meeting is at four")])
    assert affective.score > neutral.score


def test_per_turn_counting_rewards_repetition_where_distinct_does_not():
    turns = [Turn("a", "happy happy happy"), Turn("b", "happy")]
    assert _score(turns, counting="per_turn").emotion_diversity > 1.0
    assert _score(turns, counting="distinct").emotion_diversity == 0.5


def test_adjacent_scope_ignores_a_modifier_far_from_any_emotion_term():
    near = [Turn("a", "i am very happy")]
    far = [Turn("a", "very much the sort of thing one does on a happy")]
    assert _score(near, intensity_scope="adjacent").intensity_diversity == 1.0
    assert _score(far, intensity_scope="adjacent").intensity_diversity == 0.0
    assert _score(far, intensity_scope="turn").intensity_diversity == 1.0


def test_nonverbal_actions_do_not_contribute_emotion_terms():
    spoken = _score([Turn("a", "i am happy")])
    acted = _score([Turn("a", "mm [looks happy]")])
    assert spoken.emotion_diversity == 1.0
    assert acted.emotion_diversity == 0.0


def test_weights_must_sum_to_one_too():
    with pytest.raises(ValueError, match="sum to 1"):
        _score([Turn("a", "happy")], emotion_weight=0.9, intensity_weight=0.9)


# ------------------------------------------------------- affective congruence

def test_matching_speech_and_action_score_higher_than_mismatched(vectorizer):
    congruent = affective_congruence_score(
        split_turns([Turn("how are you", "i am sad [trembling]")]), vectorizer
    )
    incongruent = affective_congruence_score(
        split_turns([Turn("how are you", "i am happy [trembling]")]), vectorizer
    )
    assert congruent.score > incongruent.score


def test_score_is_none_when_there_are_no_actions(vectorizer, varied_dialogue):
    result = affective_congruence_score(split_turns(varied_dialogue), vectorizer)
    assert result.score is None
    assert result.n_pairs == 0


def test_uninformative_pairs_are_set_aside_and_counted(vectorizer):
    turns = split_turns([Turn("a", "the meeting is at four [checks the time]")])
    skipped = affective_congruence_score(turns, vectorizer)
    assert skipped.score is None
    assert skipped.n_uninformative == 1

    kept = affective_congruence_score(turns, vectorizer, skip_uninformative=False)
    assert kept.score is not None
    assert kept.n_scored == 1


def test_every_action_in_a_turn_becomes_its_own_pair(vectorizer):
    turns = split_turns([Turn("a", "i am sad [trembling] [flinches]")])
    assert affective_congruence_score(turns, vectorizer).n_pairs == 2


def test_unknown_similarity_is_rejected(vectorizer):
    with pytest.raises(ValueError, match="similarity"):
        affective_congruence_score(
            split_turns([Turn("a", "hi [waves]")]), vectorizer, similarity="euclidean"
        )
