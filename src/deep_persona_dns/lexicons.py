"""Emotion lexicon loading.

The emotional expression component counts emotion-bearing words, which means
it needs a word list. Which list is a choice the caller makes: the metric is
defined over any lexicon, and different research questions call for different
ones.

A lexicon here is simply a set of lowercase word forms. Three ways to obtain
one are provided -- a plain text file, the NRC Emotion Lexicon for English, and
the Hebrew Psychological Lexicons for Hebrew -- and any other set of strings
works just as well.

Both bundled loaders read from separately distributed packages rather than
copying word lists into this repository, so that each resource keeps its own
licence and its own citation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Set

#: Emotion categories in the Hebrew Psychological Lexicons whose members are
#: emotion-bearing words. Each has a negated counterpart, prefixed "Not",
#: which is also emotion-bearing and is included alongside it.
_HEBREW_EMOTION_CATEGORIES = (
    "Amused", "Anger", "Anticipation", "Anxiety", "Ashamed", "Calm",
    "Confusion", "Contentment", "Disgust", "Enthusiastic", "Fatigue",
    "Guilty", "Hostile", "Interested", "Joy", "Nervous", "Proud", "Sad",
    "Surprise", "Trust", "Vigor",
)


def normalise_lexicon(words: Iterable[str]) -> Set[str]:
    """Lowercase and strip a collection of words, dropping empty entries."""
    return {w.strip().lower() for w in words if w and w.strip()}


def load_from_file(path: str | Path) -> Set[str]:
    """Load a lexicon from a UTF-8 text file with one word per line.

    Blank lines and lines beginning with ``#`` are ignored, so a file may carry
    its own provenance note at the top.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return normalise_lexicon(l for l in lines if not l.lstrip().startswith("#"))


def load_nrc_english() -> Set[str]:
    """Load the English NRC Emotion Lexicon (Mohammad and Turney, 2013).

    Every word carrying at least one emotion or sentiment tag is included.

    Requires the ``nrclex`` package, installed by the ``english`` extra.
    """
    try:
        import nrclex
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "load_nrc_english() requires the nrclex package, which the 'english' "
            "extra installs; see the README. Alternatively supply your own "
            "lexicon with load_from_file()."
        ) from exc

    data_path = Path(nrclex.__file__).resolve().parent / "data" / "nrc_en.json"
    with open(data_path, "r", encoding="utf-8") as handle:
        return normalise_lexicon(json.load(handle).keys())


def load_hebrew_psychological() -> Set[str]:
    """Load the Hebrew emotion word lists from the Hebrew Psychological Lexicons.

    The union of the emotion categories and their negated counterparts is
    returned. The lexicon is distributed under CC BY-SA 4.0 and should be cited
    as Shapira et al., CLPsych 2021.

    Requires the ``hepsylex`` package, installed by the ``hebrew`` extra.
    """
    try:
        from hepsylex import Lexicons
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "load_hebrew_psychological() requires the hepsylex package, which the "
            "'hebrew' extra installs; see the README. Alternatively supply your "
            "own lexicon with load_from_file()."
        ) from exc

    lexicons = Lexicons()
    words: Set[str] = set()
    missing = []
    for category in _HEBREW_EMOTION_CATEGORIES:
        for name in (f"EmotionalVariety_{category}", f"EmotionalVariety_Not{category}"):
            entry = getattr(lexicons, name, None)
            if entry is None:
                missing.append(name)
                continue
            words.update(normalise_lexicon(entry))
    if not words:
        raise RuntimeError(
            "no emotion categories were found in the installed hepsylex version; "
            f"expected names such as {missing[0] if missing else 'EmotionalVariety_Joy'}"
        )
    return words
