"""Dialogue Naturalness Score (DNS).

A reference-free metric that scores a dialogue by how far its behavioural
profile sits from a human baseline.

Four components are computed from the agent's turns -- pragmatic alignment,
joint attention, affective congruence and emotional expression -- and compared
with the distribution of the same components over human dialogue, by squared
Mahalanobis distance. That distance gives both a score in ``(0, 1]`` and a
chi-squared test of whether the dialogue is distinguishable from the baseline.

Typical use::

    from deep_persona_dns import (
        DNSEvaluator, SpacyEntityExtractor, Turn, dns_score, fit_baseline,
        load_nrc_english,
    )

    evaluator = DNSEvaluator(
        load_nrc_english(), entity_extractor=SpacyEntityExtractor()
    )
    baseline = fit_baseline(
        evaluator.profiles(human_dialogues), evaluator.components
    )
    scores = dns_score(evaluator.profiles(agent_dialogues), baseline)
"""

from .baseline import ESTIMATORS, HumanBaseline, fit_baseline
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
from .emotion import (
    CallableEmotionVectorizer,
    EmotionVectorizer,
    EnsembleEmotionVectorizer,
    TransformerEmotionVectorizer,
    build_emotion_vectorizer,
)
from .entities import (
    CallableEntityExtractor,
    EntityExtractor,
    SpacyEntityExtractor,
    TransformerEntityExtractor,
    build_entity_extractor,
)
from .evaluator import (
    COMPONENTS,
    TEXT_ONLY_COMPONENTS,
    DialogueProfile,
    DNSEvaluator,
    EvaluatorConfig,
)
from .languages import ENGLISH, HEBREW, LanguageConfig, get_language
from .lexicons import load_from_file, load_hebrew_psychological, load_nrc_english
from .score import (
    dns_score,
    is_indistinguishable,
    lambda_for_df,
    mahalanobis_sq,
    naturalness_pvalue,
)
from .turns import Turn

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # dialogues
    "Turn",
    # evaluation
    "DNSEvaluator", "EvaluatorConfig", "DialogueProfile",
    "COMPONENTS", "TEXT_ONLY_COMPONENTS",
    # components
    "pragmatic_score", "joint_attention_score",
    "affective_congruence_score", "emotional_expression_score",
    "PragmaticScore", "JointAttentionScore",
    "AffectiveCongruenceScore", "EmotionalExpressionScore",
    # languages and lexicons
    "LanguageConfig", "ENGLISH", "HEBREW", "get_language",
    "load_nrc_english", "load_hebrew_psychological", "load_from_file",
    # backends
    "EntityExtractor", "SpacyEntityExtractor", "TransformerEntityExtractor",
    "CallableEntityExtractor", "build_entity_extractor",
    "EmotionVectorizer", "TransformerEmotionVectorizer",
    "EnsembleEmotionVectorizer", "CallableEmotionVectorizer",
    "build_emotion_vectorizer",
    # baseline and scoring
    "fit_baseline", "HumanBaseline", "ESTIMATORS",
    "dns_score", "mahalanobis_sq", "naturalness_pvalue",
    "is_indistinguishable", "lambda_for_df",
]
