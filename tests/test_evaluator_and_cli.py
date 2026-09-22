import json

import numpy as np
import pytest

from deep_persona_dns.cli import main
from deep_persona_dns.evaluator import (
    COMPONENTS,
    TEXT_ONLY_COMPONENTS,
    DNSEvaluator,
    EvaluatorConfig,
)
from deep_persona_dns.turns import Turn

from conftest import EMOTION_LEXICON


@pytest.fixture
def evaluator(extractor):
    return DNSEvaluator(EMOTION_LEXICON, entity_extractor=extractor)


@pytest.fixture
def full_evaluator(extractor, vectorizer):
    return DNSEvaluator(
        EMOTION_LEXICON, entity_extractor=extractor, emotion_vectorizer=vectorizer
    )


# --------------------------------------------------------------- evaluator

def test_profile_width_follows_the_available_backends(evaluator, full_evaluator):
    assert tuple(evaluator.components) == TEXT_ONLY_COMPONENTS
    assert tuple(full_evaluator.components) == COMPONENTS


def test_congruence_is_absent_without_a_vectorizer(evaluator, varied_dialogue):
    assert evaluator.evaluate(varied_dialogue).scores()["congruence"] is None


def test_congruence_is_present_when_actions_exist(full_evaluator):
    profile = full_evaluator.evaluate([Turn("how are you", "i am sad [trembling]")])
    assert profile.scores()["congruence"] is not None


def test_missing_component_becomes_nan_not_zero(full_evaluator, varied_dialogue):
    vector = full_evaluator.evaluate(varied_dialogue).as_vector(COMPONENTS)
    assert np.isnan(vector[COMPONENTS.index("congruence")])


def test_profile_matrix_has_one_row_per_dialogue(evaluator, varied_dialogue, echoing_dialogue):
    assert evaluator.profiles([varied_dialogue, echoing_dialogue]).shape == (2, 3)


def test_varied_dialogue_beats_echoing_on_pragmatics(
    evaluator, varied_dialogue, echoing_dialogue
):
    varied = evaluator.evaluate(varied_dialogue).scores()["pragmatics"]
    echoing = evaluator.evaluate(echoing_dialogue).scores()["pragmatics"]
    assert varied > echoing


def test_configuration_changes_reach_the_components(extractor):
    dialogue = [Turn("i have a border collie", "a border collie is hard work"), Turn("yes", "mm")]
    strict = DNSEvaluator(EMOTION_LEXICON, entity_extractor=extractor)
    permissive = DNSEvaluator(
        EMOTION_LEXICON,
        entity_extractor=extractor,
        config=EvaluatorConfig(acknowledgement_window="current_or_next"),
    )
    assert strict.evaluate(dialogue).scores()["joint_attention"] == 0.0
    assert permissive.evaluate(dialogue).scores()["joint_attention"] == 1.0


def test_an_empty_lexicon_is_refused(extractor):
    with pytest.raises(ValueError, match="lexicon is empty"):
        DNSEvaluator(set(), entity_extractor=extractor)


def test_an_empty_dialogue_is_refused(evaluator):
    with pytest.raises(ValueError, match="empty"):
        evaluator.evaluate([])


def test_unknown_component_name_is_refused(evaluator, varied_dialogue):
    with pytest.raises(ValueError, match="unknown components"):
        evaluator.evaluate(varied_dialogue).as_vector(["pragmatics", "charisma"])


def test_profile_serialises_to_json(evaluator, varied_dialogue):
    payload = evaluator.evaluate(varied_dialogue).as_dict()
    assert json.loads(json.dumps(payload))["scores"]["pragmatics"] > 0


# --------------------------------------------------------------------- cli

def _write(path, dialogues):
    with open(path, "w", encoding="utf-8") as handle:
        for index, turns in enumerate(dialogues):
            handle.write(json.dumps({
                "dialogue_id": f"d{index}",
                "turns": [{"user": t.user, "agent": t.agent} for t in turns],
            }) + "\n")
    return str(path)


@pytest.fixture
def corpus(tmp_path, human_corpus, varied_dialogue, echoing_dialogue):
    human = _write(tmp_path / "human.jsonl", human_corpus)
    agent = _write(tmp_path / "agent.jsonl", [varied_dialogue, echoing_dialogue])
    return human, agent, tmp_path


def _stub_evaluator(monkeypatch, extractor):
    import deep_persona_dns.cli as cli

    monkeypatch.setattr(
        cli, "build_evaluator",
        lambda args: DNSEvaluator(EMOTION_LEXICON, entity_extractor=extractor),
    )


def test_fit_then_score_round_trip(corpus, monkeypatch, extractor, capsys):
    human, agent, tmp_path = corpus
    _stub_evaluator(monkeypatch, extractor)
    baseline = str(tmp_path / "baseline.json")
    scores = str(tmp_path / "scores.jsonl")

    assert main(["fit-baseline", human, "-o", baseline]) == 0
    saved = json.loads(open(baseline, encoding="utf-8").read())
    assert saved["components"] == list(TEXT_ONLY_COMPONENTS)
    assert saved["n"] == 8

    assert main(["score", agent, "--baseline", baseline, "-o", scores]) == 0
    rows = [json.loads(l) for l in open(scores, encoding="utf-8") if l.strip()]
    assert [r["dialogue_id"] for r in rows] == ["d0", "d1"]
    # Compare distances rather than scores: exp(-lambda * d^2) underflows to
    # zero for a dialogue far enough out, which would make the ordering
    # numerically meaningless even though it is substantively clear.
    assert rows[0]["mahalanobis_sq"] < rows[1]["mahalanobis_sq"]
    assert rows[0]["p_value"] > rows[1]["p_value"]
    assert set(rows[0]["components"]) == set(TEXT_ONLY_COMPONENTS)


def test_verbose_adds_diagnostics(corpus, monkeypatch, extractor, tmp_path):
    human, agent, _ = corpus
    _stub_evaluator(monkeypatch, extractor)
    baseline = str(tmp_path / "b.json")
    scores = str(tmp_path / "s.jsonl")
    main(["fit-baseline", human, "-o", baseline])
    main(["score", agent, "--baseline", baseline, "-o", scores, "--verbose"])
    first = json.loads(open(scores, encoding="utf-8").readline())
    assert "diagnostics" in first


def test_missing_input_reports_rather_than_traces(monkeypatch, extractor):
    _stub_evaluator(monkeypatch, extractor)
    with pytest.raises(SystemExit, match="no such file"):
        main(["fit-baseline", "absent.jsonl", "-o", "out.json"])


def test_malformed_baseline_reports_rather_than_traces(corpus, monkeypatch, extractor, tmp_path):
    _, agent, _ = corpus
    _stub_evaluator(monkeypatch, extractor)
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="not a baseline"):
        main(["score", agent, "--baseline", str(bad)])


def test_a_turn_without_agent_text_is_reported(tmp_path, monkeypatch, extractor):
    _stub_evaluator(monkeypatch, extractor)
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"turns": [{"user": "hi"}]}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="'user' and 'agent'"):
        main(["fit-baseline", str(path), "-o", str(tmp_path / "o.json")])


def test_scoring_refuses_a_baseline_it_cannot_reproduce(corpus, monkeypatch, extractor, tmp_path):
    human, agent, _ = corpus
    _stub_evaluator(monkeypatch, extractor)
    baseline = tmp_path / "wide.json"
    main(["fit-baseline", human, "-o", str(baseline)])
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    saved["components"].append("congruence")
    baseline.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(SystemExit, match="congruence"):
        main(["score", agent, "--baseline", str(baseline)])
