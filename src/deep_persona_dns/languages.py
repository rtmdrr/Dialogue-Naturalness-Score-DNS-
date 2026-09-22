"""Per-language configuration.

A language contributes three things to the metric: the set of intensity
modifiers used by the emotional expression component, the pattern that marks
nonverbal actions, and sensible default model names for the entity and emotion
backends.

The emotion lexicon is deliberately not part of this configuration. It is
larger, it carries its own licence, and which one to use is a decision the
caller should make explicitly. See :mod:`deep_persona_dns.lexicons`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Optional

from .turns import BRACKET_RE

ENGLISH_INTENSIFIERS: FrozenSet[str] = frozenset({
    "very", "really", "extremely", "so", "too", "quite", "super",
    "incredibly", "deeply", "utterly",
    "slightly", "somewhat", "barely", "hardly", "almost",
})

HEBREW_INTENSIFIERS: FrozenSet[str] = frozenset({
    "מאוד", "ממש", "כלכך", "כל-כך", "במיוחד", "לגמרי", "באמת",
    "קצת", "טיפה", "די", "בערך", "בקושי", "כמעט",
})


@dataclass(frozen=True)
class LanguageConfig:
    """Language-specific settings for the evaluator.

    Attributes:
        code: a short language identifier, used in diagnostics.
        intensifiers: intensity modifiers counted by the emotional expression
            component.
        bracket_pattern: the pattern marking nonverbal actions in an agent
            response.
        default_emotion_model: model name for the affective congruence
            component, if one is available for this language.
        default_entity_model: model name for the entity backend, if the
            language uses a model rather than a rule-based extractor.
    """

    code: str
    intensifiers: FrozenSet[str]
    bracket_pattern: object = field(default=BRACKET_RE)
    default_emotion_model: Optional[str] = None
    default_entity_model: Optional[str] = None


ENGLISH = LanguageConfig(
    code="en",
    intensifiers=ENGLISH_INTENSIFIERS,
    default_emotion_model="bhadresh-savani/distilbert-base-uncased-emotion",
    default_entity_model="en_core_web_sm",
)

HEBREW = LanguageConfig(
    code="he",
    intensifiers=HEBREW_INTENSIFIERS,
    default_emotion_model="avichr/hebEMO",
    default_entity_model="avichr/heBERT_NER",
)

_BY_CODE = {
    "en": ENGLISH, "eng": ENGLISH, "english": ENGLISH,
    "he": HEBREW, "heb": HEBREW, "hebrew": HEBREW, "iw": HEBREW,
}


def get_language(code: str) -> LanguageConfig:
    """Look up a bundled language configuration by name or code."""
    key = str(code).strip().lower()
    if key not in _BY_CODE:
        raise ValueError(
            f"unknown language {code!r}; bundled configurations are "
            f"{sorted(set(c.code for c in _BY_CODE.values()))}. Build a "
            "LanguageConfig directly to add another."
        )
    return _BY_CODE[key]
