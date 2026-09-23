"""Backends that identify the entities a user utterance introduces.

The joint attention component needs to know which turns introduce something
new, so that it can ask whether the agent picked it up. That question can be
answered by an annotator of any kind, so it is expressed here as a protocol
with three implementations:

* :class:`SpacyEntityExtractor` -- noun chunks and named entities, for English.
* :class:`TransformerEntityExtractor` -- a token classification pipeline, for
  languages served by a named entity model rather than a parser.
* :class:`CallableEntityExtractor` -- an adapter for an annotator supplied by
  the caller, including a language model prompted to list new entities.

The first two are deterministic, run offline and cost nothing, which is why
one of them is the default. The third is provided as an interface rather than
an implementation, because a language model annotator needs credentials, costs
money per call and does not necessarily return the same answer twice -- all
reasonable in a study, none of them reasonable in a default.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Protocol, Set, runtime_checkable

from .turns import normalize_text

#: Shorter strings are discarded, as they are rarely referential.
MIN_ENTITY_CHARS = 3


@runtime_checkable
class EntityExtractor(Protocol):
    """Anything that can list the entities mentioned in an utterance."""

    def extract(self, text: str) -> Set[str]:
        """Return the normalised entity strings mentioned in ``text``."""
        ...


def _clean(candidates: Iterable[str], min_chars: int) -> Set[str]:
    out = set()
    for candidate in candidates:
        text = normalize_text(str(candidate).strip())
        if len(text) >= min_chars:
            out.add(text)
    return out


class SpacyEntityExtractor:
    """Noun chunks and named entities from a spaCy pipeline.

    Noun chunks are included alongside named entities because a new topic is
    frequently an ordinary noun phrase rather than a name.

    Requires spaCy, installed by the ``english`` extra, and a spaCy model::

        python -m spacy download en_core_web_sm
    """

    def __init__(self, model: str = "en_core_web_sm", *, min_chars: int = MIN_ENTITY_CHARS):
        try:
            import spacy
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "SpacyEntityExtractor requires spaCy, which the 'english' extra "
                "installs; see the README."
            ) from exc
        self._nlp = spacy.load(model)
        self._min_chars = min_chars

    def extract(self, text: str) -> Set[str]:
        doc = self._nlp(str(text))
        candidates = [chunk.text for chunk in doc.noun_chunks]
        candidates += [ent.text for ent in doc.ents]
        return _clean(candidates, self._min_chars)


class TransformerEntityExtractor:
    """Named entities from a token classification pipeline.

    Requires ``transformers``, installed by the ``hebrew`` extra.
    """

    def __init__(
        self,
        model: str,
        *,
        device: Optional[int] = None,
        min_chars: int = MIN_ENTITY_CHARS,
    ):
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "TransformerEntityExtractor requires transformers, which the "
                "'hebrew' extra installs; see the README."
            ) from exc
        if device is None:
            device = _default_device()
        self._ner = pipeline(
            task="token-classification",
            model=model,
            tokenizer=model,
            aggregation_strategy="simple",
            device=device,
        )
        self._min_chars = min_chars

    def extract(self, text: str) -> Set[str]:
        text = str(text).strip()
        if not text:
            return set()
        return _clean((span.get("word", "") for span in self._ner(text)), self._min_chars)


class CallableEntityExtractor:
    """Adapter for an entity annotator supplied by the caller.

    The callable receives one utterance and returns the entities or topics it
    introduces. This is the seam for a language model annotator: prompt a model
    to list the new entities in an utterance and wrap the call in a function.

    Example::

        def annotate(utterance: str) -> list[str]:
            reply = my_model.complete(PROMPT.format(utterance=utterance))
            return [line.strip("- ") for line in reply.splitlines() if line.strip()]

        extractor = CallableEntityExtractor(annotate)

    Results are cached per utterance, so a repeated utterance costs one call.
    """

    def __init__(
        self,
        annotator: Callable[[str], Iterable[str]],
        *,
        min_chars: int = MIN_ENTITY_CHARS,
        cache: bool = True,
    ):
        self._annotator = annotator
        self._min_chars = min_chars
        self._cache: Optional[dict] = {} if cache else None

    def extract(self, text: str) -> Set[str]:
        text = str(text)
        if self._cache is not None and text in self._cache:
            return self._cache[text]
        result = _clean(self._annotator(text) or (), self._min_chars)
        if self._cache is not None:
            self._cache[text] = result
        return result


def _default_device() -> int:
    try:
        import torch
    except ImportError:  # pragma: no cover - depends on the environment
        return -1
    return 0 if torch.cuda.is_available() else -1


def build_entity_extractor(language, **kwargs) -> EntityExtractor:
    """Construct the default extractor for a language configuration."""
    model = language.default_entity_model
    if model is None:
        raise ValueError(
            f"language {language.code!r} has no default entity model; pass an "
            "EntityExtractor explicitly"
        )
    if language.code == "en":
        return SpacyEntityExtractor(model, **kwargs)
    return TransformerEntityExtractor(model, **kwargs)
