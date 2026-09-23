"""Backends that turn a piece of text into an emotion probability vector.

Affective congruence compares what the agent says with what it is described as
doing, by embedding both in the same emotion space and measuring the angle
between them. Any model that maps text to a fixed-length probability vector
will serve, so the requirement is expressed as a protocol.

Two implementations are provided: a single multi-class classifier, and an
ensemble of per-emotion binary classifiers whose positive-class probabilities
are stacked into one vector.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

#: Plutchik emotions, the order used when stacking per-emotion classifiers.
PLUTCHIK_EMOTIONS = (
    "anger", "disgust", "anticipation", "fear",
    "joy", "sadness", "surprise", "trust",
)


@runtime_checkable
class EmotionVectorizer(Protocol):
    """Anything that maps texts to emotion probability vectors."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an array of shape ``(len(texts), n_emotions)``."""
        ...


def is_near_uniform(vector: np.ndarray, tolerance: float = 1e-3) -> bool:
    """Whether a probability vector is flat enough to carry no information.

    A classifier that spreads its mass evenly has not identified an emotion,
    and the angle between two such vectors says nothing about how well the
    texts agree.
    """
    values = np.asarray(vector, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return True
    values = values / total
    return bool(values.max() - values.min() < tolerance)


def _resolve_device(device: Optional[str]):
    import torch

    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerEmotionVectorizer:
    """A single multi-class or multi-label emotion classifier.

    Logits are converted with softmax for a multi-class model and with the
    logistic function for a multi-label one, following the model's own
    ``problem_type``.

    Requires ``transformers``, installed by the ``congruence`` extra.
    """

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "TransformerEmotionVectorizer requires transformers and torch, which "
                "the 'congruence' extra installs; see the README."
            ) from exc

        self._torch = torch
        self._device = _resolve_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForSequenceClassification.from_pretrained(model)
        self._model.to(self._device)
        self._model.eval()
        self._batch_size = batch_size
        self._max_length = max_length

        config = self._model.config
        id2label = getattr(config, "id2label", None) or {}
        self.labels: List[str] = (
            [id2label[i] for i in sorted(id2label)]
            if id2label
            else [f"label_{i}" for i in range(int(config.num_labels))]
        )
        self._multilabel = getattr(config, "problem_type", None) == "multi_label_classification"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=float)
        out = []
        for start in range(0, len(texts), self._batch_size):
            batch = [str(t) for t in texts[start : start + self._batch_size]]
            encoded = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=self._max_length, return_tensors="pt",
            ).to(self._device)
            with self._torch.no_grad():
                logits = self._model(**encoded).logits
            probs = (
                self._torch.sigmoid(logits)
                if self._multilabel
                else self._torch.softmax(logits, dim=-1)
            )
            out.append(probs.cpu().numpy())
        return np.vstack(out)


class EnsembleEmotionVectorizer:
    """One binary classifier per emotion, stacked into a single vector.

    Each model contributes its positive-class probability, in the order the
    emotions are given. This is the shape of the Hebrew emotion models, which
    are published as a family of per-emotion classifiers.

    Requires ``transformers``, installed by the ``hebrew`` extra.
    """

    def __init__(
        self,
        model_template: str,
        emotions: Sequence[str] = PLUTCHIK_EMOTIONS,
        *,
        batch_size: int = 16,
        device: Optional[str] = None,
        max_length: int = 512,
        positive_index: int = 1,
    ):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "EnsembleEmotionVectorizer requires transformers and torch, which the "
                "'hebrew' extra installs; see the README."
            ) from exc

        self._torch = torch
        self._device = _resolve_device(device)
        self.labels = list(emotions)
        names = [model_template.format(emotion=e) for e in self.labels]
        self._tokenizer = AutoTokenizer.from_pretrained(names[0])
        self._models = []
        for name in names:
            model = AutoModelForSequenceClassification.from_pretrained(name)
            model.to(self._device)
            model.eval()
            self._models.append(model)
        self._batch_size = batch_size
        self._max_length = max_length
        self._positive_index = positive_index

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, len(self.labels)), dtype=float)
        columns = []
        for model in self._models:
            scores = []
            for start in range(0, len(texts), self._batch_size):
                batch = [str(t) for t in texts[start : start + self._batch_size]]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self._max_length, return_tensors="pt",
                ).to(self._device)
                with self._torch.no_grad():
                    probs = self._torch.softmax(model(**encoded).logits, dim=-1)
                scores.append(probs[:, self._positive_index].cpu().numpy())
            columns.append(np.concatenate(scores))
        return np.stack(columns, axis=1)


class CallableEmotionVectorizer:
    """Adapter for an emotion model supplied by the caller."""

    def __init__(self, encoder: Callable[[Sequence[str]], Iterable[Iterable[float]]]):
        self._encoder = encoder

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(list(self._encoder(texts)), dtype=float)


def build_emotion_vectorizer(language, **kwargs) -> EmotionVectorizer:
    """Construct the default emotion vectorizer for a language configuration."""
    model = language.default_emotion_model
    if model is None:
        raise ValueError(
            f"language {language.code!r} has no default emotion model; pass an "
            "EmotionVectorizer explicitly"
        )
    if language.code == "he":
        return EnsembleEmotionVectorizer(model + "_{emotion}", **kwargs)
    return TransformerEmotionVectorizer(model, **kwargs)
