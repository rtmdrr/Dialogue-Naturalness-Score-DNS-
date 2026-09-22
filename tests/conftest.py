"""Shared fixtures.

The tests use stub backends throughout. Entity extraction and emotion
classification are separate concerns with their own test suites upstream, and
depending on downloaded models here would make the suite slow, network-bound
and dependent on a particular model's behaviour.
"""

from __future__ import annotations

import pytest

from deep_persona_dns.emotion import CallableEmotionVectorizer
from deep_persona_dns.entities import CallableEntityExtractor
from deep_persona_dns.turns import Turn

KNOWN_ENTITIES = ("border collie", "hiking", "cello", "wedding")

EMOTION_LEXICON = {"happy", "sad", "love", "afraid", "delighted", "angry"}
INTENSIFIERS = {"very", "really", "extremely", "slightly"}


@pytest.fixture
def extractor():
    """Recognises a fixed vocabulary, so tests control what counts as new."""
    return CallableEntityExtractor(
        lambda text: [e for e in KNOWN_ENTITIES if e in text.lower()]
    )


@pytest.fixture
def vectorizer():
    """Two-dimensional emotion space: roughly negative versus positive."""

    def encode(texts):
        vectors = []
        for text in texts:
            lowered = str(text).lower()
            if any(w in lowered for w in ("sad", "afraid", "angry", "trembling", "flinches")):
                vectors.append([0.9, 0.1])
            elif any(w in lowered for w in ("happy", "delighted", "love", "smiles", "grins")):
                vectors.append([0.1, 0.9])
            else:
                vectors.append([0.5, 0.5])
        return vectors

    return CallableEmotionVectorizer(encode)


@pytest.fixture
def varied_dialogue():
    return [
        Turn("i have a border collie", "what is her name?"),
        Turn("bess, and she herds everything", "that is very border collie of her"),
        Turn("we go hiking together", "how far do you usually walk?"),
        Turn("about ten kilometres", "hiking that far takes real stamina"),
    ]


@pytest.fixture
def echoing_dialogue():
    return [
        Turn("i have a border collie", "i have a border collie"),
        Turn("we go hiking together", "we go hiking together"),
        Turn("about ten kilometres", "about ten kilometres"),
    ]


@pytest.fixture
def human_corpus():
    """Eight dialogues of the same shape as ``varied_dialogue``.

    A baseline has to resemble the dialogues it will be used to score. These
    share the structure -- four turns, an entity introduced and then taken up
    -- while differing in wording, so the fitted distribution has variance
    without being drawn from a different population than the test cases.
    """
    subjects = [
        ("border collie", "bess", "herds everything"),
        ("border collie", "juno", "never sits still"),
        ("cello", "a rented one", "smells of rosin"),
        ("cello", "my grandfather's", "needs new strings"),
        ("wedding", "in may", "is going to be small"),
        ("wedding", "next spring", "has been years in the planning"),
        ("hiking", "most weekends", "keeps me sane"),
        ("hiking", "when i can", "clears my head"),
    ]
    return [
        [
            Turn(f"i wanted to tell you about the {topic}", "go on, i am listening"),
            Turn(f"it is {detail}", f"the {topic} sounds like it matters to you"),
            Turn(f"it {quality}", "what made you take it up?"),
            Turn("a friend suggested it years ago", "friends are good for that sort of thing"),
        ]
        for topic, detail, quality in subjects
    ]
