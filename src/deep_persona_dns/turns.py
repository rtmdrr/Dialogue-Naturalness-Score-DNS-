"""Dialogue turn representation and the text operations the components share.

A dialogue is a sequence of :class:`Turn` objects, each pairing one user
utterance with the agent response that follows it.

An agent response may carry nonverbal actions in square brackets, for example
``"I'm fine [avoids eye contact]"``. Every component except affective
congruence operates on the verbal content alone; affective congruence is the
component that compares the two.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

#: Matches a bracketed nonverbal action inside an agent response.
BRACKET_RE = re.compile(r"\[(.*?)\]")

#: Word characters, plus the marks that appear word-internally rather than
#: between words: the apostrophe, the Hebrew geresh and gershayim, and the
#: ASCII double quote, which is commonly typed in place of a gershayim.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['\"׳״][^\W\d_]+)*", flags=re.UNICODE)


@dataclass(frozen=True)
class Turn:
    """One exchange: a user utterance and the agent response to it."""

    user: str
    agent: str


@dataclass(frozen=True)
class SplitTurn:
    """A turn with the agent response separated into its two channels."""

    user: str
    agent: str
    verbal: str
    actions: Tuple[str, ...]


def normalize_text(text: str) -> str:
    """Apply NFKC normalisation and case folding."""
    return unicodedata.normalize("NFKC", str(text)).lower()


def tokenize(text: str) -> List[str]:
    """Split normalised text into word tokens, discarding digits and symbols."""
    return _TOKEN_RE.findall(normalize_text(text))


def split_verbal_and_actions(text: str) -> Tuple[str, List[str]]:
    """Separate an agent response into verbal content and bracketed actions.

    Returns the utterance with bracketed spans removed and whitespace collapsed,
    together with the list of action descriptions in order of appearance.
    """
    text = str(text)
    actions = [a.strip() for a in BRACKET_RE.findall(text) if a and a.strip()]
    verbal = " ".join(BRACKET_RE.sub("", text).split()).strip()
    return verbal, actions


def split_turns(turns: Sequence[Turn]) -> List[SplitTurn]:
    """Split every turn in a dialogue once, so later passes can reuse the result."""
    out = []
    for turn in turns:
        verbal, actions = split_verbal_and_actions(turn.agent)
        out.append(SplitTurn(turn.user, turn.agent, verbal, tuple(actions)))
    return out


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, in O(len(a) * len(b)) time."""
    if not a or not b:
        return 0
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            current = row[j]
            row[j] = prev + 1 if x == y else max(row[j], row[j - 1])
            prev = current
    return row[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between two strings, on normalised word tokens."""
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(ref, pred)
    recall = lcs / len(ref)
    precision = lcs / len(pred)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def jaccard(prediction: str, reference: str) -> float:
    """Jaccard similarity between the token sets of two strings."""
    pred: Set[str] = set(tokenize(prediction))
    ref: Set[str] = set(tokenize(reference))
    if not pred or not ref:
        return 0.0
    return float(len(pred & ref) / len(pred | ref))


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is zero."""
    return numerator / denominator if denominator else 0.0
