"""The component scores that make up a dialogue's scoring profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .turns import SplitTurn, jaccard, rouge_l, safe_divide

#: Similarity measures available for user-agent overlap.
OVERLAP_MEASURES = ("rouge_l", "jaccard")


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
