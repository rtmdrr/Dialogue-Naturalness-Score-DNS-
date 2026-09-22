"""The evaluator that turns a dialogue into a scoring profile."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from .components import (
    AffectiveCongruenceScore,
    EmotionalExpressionScore,
    JointAttentionScore,
    PragmaticScore,
    affective_congruence_score,
    emotional_expression_score,
    joint_attention_score,
    pragmatic_score,
)
from .emotion import EmotionVectorizer
from .entities import EntityExtractor
from .languages import ENGLISH, LanguageConfig
from .turns import Turn, split_turns

#: The four components, in the order the scoring profile uses.
COMPONENTS = ("pragmatics", "joint_attention", "congruence", "emotion")

#: The three components available for transcripts without nonverbal actions.
TEXT_ONLY_COMPONENTS = ("pragmatics", "joint_attention", "emotion")


@dataclass(frozen=True)
class EvaluatorConfig:
    """Settings for the component scores.

    The defaults reproduce the configuration reported in the paper. Each
    alternative is documented on the component that uses it.
    """

    # pragmatic alignment
    alpha: float = 1 / 3
    beta: float = 1 / 3
    gamma: float = 1 / 3
    echolalia_threshold: float = 0.65
    self_repetition_window: int = 5
    overlap_measure: str = "rouge_l"
    clamp_pragmatics: bool = True

    # joint attention
    acknowledgement_window: str = "next"

    # emotional expression
    emotion_weight: float = 0.7
    intensity_weight: float = 0.3
    emotion_counting: str = "per_turn"
    intensity_scope: str = "adjacent"
    intensity_window: int = 3

    # affective congruence
    congruence_similarity: str = "cosine"
    skip_uninformative_pairs: bool = True


@dataclass(frozen=True)
class DialogueProfile:
    """The component scores for one dialogue.

    ``as_vector`` produces the profile in the order given by ``components``,
    which is what the baseline and scoring functions consume.
    """

    pragmatics: PragmaticScore
    joint_attention: JointAttentionScore
    emotion: EmotionalExpressionScore
    congruence: AffectiveCongruenceScore
    n_turns: int

    def scores(self) -> Dict[str, Optional[float]]:
        """The four component values, keyed by name."""
        return {
            "pragmatics": self.pragmatics.score,
            "joint_attention": self.joint_attention.score,
            "congruence": self.congruence.score,
            "emotion": self.emotion.score,
        }

    def as_vector(self, components: Sequence[str] = TEXT_ONLY_COMPONENTS) -> np.ndarray:
        """The profile as an array, in the given component order.

        A component that could not be computed contributes NaN, so that
        incomplete dialogues can be filtered rather than silently scored.
        """
        values = self.scores()
        unknown = [c for c in components if c not in values]
        if unknown:
            raise ValueError(f"unknown components {unknown}; expected from {COMPONENTS}")
        return np.array(
            [np.nan if values[c] is None else float(values[c]) for c in components],
            dtype=float,
        )

    def as_dict(self) -> Dict[str, object]:
        """A JSON-serialisable view, including the component diagnostics."""
        return {
            "scores": self.scores(),
            "n_turns": self.n_turns,
            "pragmatics": asdict(self.pragmatics),
            "joint_attention": asdict(self.joint_attention),
            "emotion": asdict(self.emotion),
            "congruence": asdict(self.congruence),
        }


class DNSEvaluator:
    """Computes the scoring profile of a dialogue.

    Args:
        emotion_lexicon: emotion-bearing word forms, see
            :mod:`deep_persona_dns.lexicons`.
        language: language configuration, English by default.
        entity_extractor: backend for joint attention. Required; construct one
            with :func:`deep_persona_dns.entities.build_entity_extractor`.
        emotion_vectorizer: backend for affective congruence. Optional -- when
            it is absent, congruence is not computed and the profile has three
            components.
        config: component settings.

    Example::

        evaluator = DNSEvaluator(
            emotion_lexicon=load_nrc_english(),
            entity_extractor=SpacyEntityExtractor(),
        )
        profile = evaluator.evaluate([Turn(user="...", agent="...")])
    """

    def __init__(
        self,
        emotion_lexicon: Set[str],
        *,
        entity_extractor: EntityExtractor,
        language: LanguageConfig = ENGLISH,
        emotion_vectorizer: Optional[EmotionVectorizer] = None,
        config: Optional[EvaluatorConfig] = None,
    ):
        if not emotion_lexicon:
            raise ValueError(
                "emotion_lexicon is empty; the emotional expression component "
                "would score every dialogue at zero"
            )
        self.lexicon = set(emotion_lexicon)
        self.language = language
        self.entity_extractor = entity_extractor
        self.emotion_vectorizer = emotion_vectorizer
        self.config = config or EvaluatorConfig()

    @property
    def components(self) -> Sequence[str]:
        """The components this evaluator can produce, in profile order."""
        return COMPONENTS if self.emotion_vectorizer is not None else TEXT_ONLY_COMPONENTS

    def evaluate(self, turns: Sequence[Turn]) -> DialogueProfile:
        """Score one dialogue."""
        if not turns:
            raise ValueError("cannot score an empty dialogue")
        cfg = self.config
        split = split_turns(turns)

        pragmatics = pragmatic_score(
            split,
            alpha=cfg.alpha, beta=cfg.beta, gamma=cfg.gamma,
            echolalia_threshold=cfg.echolalia_threshold,
            self_repetition_window=cfg.self_repetition_window,
            overlap_measure=cfg.overlap_measure,
            clamp=cfg.clamp_pragmatics,
        )
        attention = joint_attention_score(
            split, self.entity_extractor,
            acknowledgement_window=cfg.acknowledgement_window,
        )
        emotion = emotional_expression_score(
            split, self.lexicon, set(self.language.intensifiers),
            emotion_weight=cfg.emotion_weight,
            intensity_weight=cfg.intensity_weight,
            counting=cfg.emotion_counting,
            intensity_scope=cfg.intensity_scope,
            intensity_window=cfg.intensity_window,
        )
        if self.emotion_vectorizer is not None:
            congruence = affective_congruence_score(
                split, self.emotion_vectorizer,
                similarity=cfg.congruence_similarity,
                skip_uninformative=cfg.skip_uninformative_pairs,
            )
        else:
            congruence = AffectiveCongruenceScore(None, 0, 0, 0)

        return DialogueProfile(
            pragmatics=pragmatics,
            joint_attention=attention,
            emotion=emotion,
            congruence=congruence,
            n_turns=len(split),
        )

    def evaluate_many(self, dialogues: Sequence[Sequence[Turn]]) -> List[DialogueProfile]:
        """Score several dialogues."""
        return [self.evaluate(d) for d in dialogues]

    def profiles(
        self,
        dialogues: Sequence[Sequence[Turn]],
        components: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Score several dialogues and stack them into a profile matrix.

        Returns an array of shape ``(n_dialogues, n_components)``, ready for
        :func:`deep_persona_dns.baseline.fit_baseline`.
        """
        chosen = tuple(components or self.components)
        return np.vstack([p.as_vector(chosen) for p in self.evaluate_many(dialogues)])
