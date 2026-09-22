"""The component scores that make up a dialogue's scoring profile."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence, Set

from .entities import EntityExtractor
from .turns import SplitTurn, jaccard, normalize_text, rouge_l, safe_divide

#: Similarity measures available for user-agent overlap.
OVERLAP_MEASURES = ("rouge_l", "jaccard")

#: Where an entity may be acknowledged for the turn to count.
#:
#: ``"next"`` follows the definition in the paper: the entity must appear in
#: the agent response after the one in which the user introduced it.
#: ``"current_or_next"`` also accepts acknowledgement in the immediate
#: response, on the view that answering at once is no worse than answering a
#: turn later. The second is the more permissive of the two and will not score
#: any dialogue lower than the first.
ACKNOWLEDGEMENT_WINDOWS = ("next", "current_or_next")


@dataclass(frozen=True)
class PragmaticScore:
    """Pragmatic fluidity and echolalia.

    ``score`` is the component value; the remaining fields are the three
    penalties it combines, exposed because they are informative on their own.
    """

    score: float
    overlap: float
    self_repetition: float
    echolalia_rate: float
    n_turns: int


def _similarity(measure: str, prediction: str, reference: str) -> float:
    if measure == "rouge_l":
        return rouge_l(prediction, reference)
    if measure == "jaccard":
        return jaccard(prediction, reference)
    raise ValueError(f"overlap measure must be one of {OVERLAP_MEASURES}, got {measure!r}")


def pragmatic_score(
    turns: Sequence[SplitTurn],
    *,
    alpha: float = 1 / 3,
    beta: float = 1 / 3,
    gamma: float = 1 / 3,
    echolalia_threshold: float = 0.65,
    self_repetition_window: int = 5,
    overlap_measure: str = "rouge_l",
    clamp: bool = True,
) -> PragmaticScore:
    """Score how far the agent avoids mirroring the user and repeating itself.

    Three penalties are averaged over the dialogue and subtracted from one:

    * **user-agent overlap** -- similarity between the user utterance and the
      agent response at the same turn. High values indicate mirroring rather
      than a substantive reply.
    * **self-repetition** -- the greatest similarity between the current agent
      response and any of the previous ``self_repetition_window`` responses.
      High values indicate lexical fixation.
    * **echolalia rate** -- the proportion of turns whose overlap exceeds
      ``echolalia_threshold``.

    Args:
        turns: the dialogue, already split into verbal and nonverbal channels.
        alpha, beta, gamma: weights of the three penalties. They must sum to 1.
        echolalia_threshold: overlap above which a turn counts as echolalia.
        self_repetition_window: how many previous agent responses to compare
            against.
        overlap_measure: ``"rouge_l"`` or ``"jaccard"``.
        clamp: whether to floor the score at 0. Without it the score can go
            negative when all three penalties are high.

    Returns:
        A :class:`PragmaticScore`, higher being more fluid.
    """
    if not turns:
        raise ValueError("cannot score an empty dialogue")
    if abs(alpha + beta + gamma - 1.0) > 1e-6:
        raise ValueError("alpha, beta and gamma must sum to 1.0")
    if self_repetition_window < 1:
        raise ValueError("self_repetition_window must be at least 1")

    overlaps: List[float] = []
    self_repetitions: List[float] = []
    echolalia_events = 0
    history: List[str] = []

    for turn in turns:
        verbal = turn.verbal or ""

        overlap = _similarity(overlap_measure, verbal, turn.user)
        overlaps.append(overlap)
        if overlap > echolalia_threshold:
            echolalia_events += 1

        if history and verbal:
            window = history[-self_repetition_window:]
            self_repetitions.append(
                max(_similarity(overlap_measure, verbal, prior) for prior in window)
            )
        else:
            self_repetitions.append(0.0)

        history.append(verbal)

    n = len(turns)
    mean_overlap = sum(overlaps) / n
    mean_self_repetition = sum(self_repetitions) / n
    echolalia_rate = safe_divide(echolalia_events, n)

    penalty = alpha * mean_overlap + beta * mean_self_repetition + gamma * echolalia_rate
    score = 1.0 - penalty
    if clamp:
        score = max(0.0, score)

    return PragmaticScore(
        score=float(score),
        overlap=float(mean_overlap),
        self_repetition=float(mean_self_repetition),
        echolalia_rate=float(echolalia_rate),
        n_turns=n,
    )


@dataclass(frozen=True)
class JointAttentionScore:
    """Joint attention capability.

    Attributes:
        score: the proportion of turns introducing a new entity in which the
            agent went on to reference one of them. Zero when the dialogue
            introduces nothing new, which is also reported by ``n_new_turns``.
        n_new_turns: how many turns introduced at least one new entity.
        n_acknowledged: how many of those were acknowledged.
    """

    score: float
    n_new_turns: int
    n_acknowledged: int


def _references(entities: Set[str], text: str) -> bool:
    """Whether any entity appears in the text, whole or by its head word."""
    haystack = normalize_text(text)
    if not haystack:
        return False
    for entity in entities:
        entity = normalize_text(entity).strip()
        if not entity:
            continue
        if entity in haystack:
            return True
        head = entity.split()[-1]
        if len(head) >= 2 and re.search(rf"(?<!\w){re.escape(head)}(?!\w)", haystack):
            return True
    return False


def joint_attention_score(
    turns: Sequence[SplitTurn],
    extractor: EntityExtractor,
    *,
    acknowledgement_window: str = "next",
) -> JointAttentionScore:
    """Score how reliably the agent takes up what the user introduces.

    For every turn in which the user mentions an entity not seen earlier in the
    dialogue, the agent is credited if it references one of those entities.

    Args:
        turns: the dialogue, already split.
        extractor: identifies the entities in a user utterance.
        acknowledgement_window: which agent responses may carry the
            acknowledgement, see :data:`ACKNOWLEDGEMENT_WINDOWS`.

    Returns:
        A :class:`JointAttentionScore`, higher being more attentive.
    """
    if acknowledgement_window not in ACKNOWLEDGEMENT_WINDOWS:
        raise ValueError(
            f"acknowledgement_window must be one of {ACKNOWLEDGEMENT_WINDOWS}, "
            f"got {acknowledgement_window!r}"
        )

    seen: Set[str] = set()
    new_turns = 0
    acknowledged = 0

    for index, turn in enumerate(turns):
        introduced = extractor.extract(turn.user) - seen
        if not introduced:
            continue

        new_turns += 1
        candidates = []
        if acknowledgement_window == "current_or_next":
            candidates.append(turn.verbal)
        if index + 1 < len(turns):
            candidates.append(turns[index + 1].verbal)

        if any(_references(introduced, text or "") for text in candidates):
            acknowledged += 1

        seen |= introduced

    return JointAttentionScore(
        score=float(safe_divide(acknowledged, new_turns)),
        n_new_turns=new_turns,
        n_acknowledged=acknowledged,
    )
