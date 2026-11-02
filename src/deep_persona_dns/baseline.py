"""Estimating the human baseline distribution.

DNS is a distance from a reference distribution, so it needs one: the mean and
covariance of the scoring profile over a corpus of high-quality human dialogue.
Everything downstream is relative to that fit, which makes the choice of
baseline corpus as consequential as any parameter in the metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

#: Available covariance estimators.
#:
#: ``"empirical"`` (default) is the sample covariance of the human profiles,
#: as in the paper's definition of the reference distribution. It is singular
#: only when a component is constant across the whole baseline, in which case
#: the pseudo-inverse is used.
#:
#: ``"scale_invariant"`` regularises for small baselines: it applies
#: Ledoit-Wolf shrinkage to the correlation matrix and then restores each
#: component's own variance, so the components keep their relative scales.
#:
#: ``"shrinkage"`` applies Ledoit-Wolf shrinkage to the covariance itself. Its
#: target gives every component the same variance, so when the components are
#: on very different scales -- as they are here -- it inflates the variance of
#: the smaller ones and sharply reduces their weight in the distance. Prefer
#: ``"scale_invariant"`` when regularisation is needed.
ESTIMATORS = ("empirical", "scale_invariant", "shrinkage")


@dataclass(frozen=True)
class HumanBaseline:
    """The fitted reference distribution.

    Attributes:
        mean: the centroid of the human profiles.
        precision: the inverse covariance used by the Mahalanobis distance.
        components: component names, in the order of ``mean``.
        n: how many dialogues the fit used.
        estimator: which covariance estimator produced it.
        scale: per-component standard deviations, when the estimator records
            them separately.
    """

    mean: np.ndarray
    precision: np.ndarray
    components: Sequence[str]
    n: int
    estimator: str
    scale: Optional[np.ndarray] = None

    @property
    def n_components(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict:
        """A JSON-serialisable view."""
        out = {
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "components": list(self.components),
            "n": int(self.n),
            "estimator": self.estimator,
        }
        if self.scale is not None:
            out["scale"] = self.scale.tolist()
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "HumanBaseline":
        scale = data.get("scale")
        return cls(
            mean=np.asarray(data["mean"], dtype=float),
            precision=np.asarray(data["precision"], dtype=float),
            components=list(data["components"]),
            n=int(data["n"]),
            estimator=str(data.get("estimator", "shrinkage")),
            scale=np.asarray(scale, dtype=float) if scale is not None else None,
        )


def _drop_incomplete(profiles: np.ndarray) -> np.ndarray:
    return profiles[~np.isnan(profiles).any(axis=1)]


def fit_baseline(
    profiles: np.ndarray,
    components: Sequence[str],
    *,
    estimator: str = "empirical",
) -> HumanBaseline:
    """Fit the reference distribution to a matrix of human scoring profiles.

    Args:
        profiles: array of shape ``(n_dialogues, n_components)``, one row per
            human dialogue. Rows containing NaN are dropped.
        components: component names, matching the column order.
        estimator: covariance estimator, see :data:`ESTIMATORS`.

    Returns:
        A :class:`HumanBaseline`.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")

    profiles = _drop_incomplete(np.asarray(profiles, dtype=float))
    if profiles.ndim != 2:
        raise ValueError("profiles must be a two-dimensional array")
    if profiles.shape[1] != len(components):
        raise ValueError(
            f"profiles has {profiles.shape[1]} columns but {len(components)} "
            "component names were given"
        )
    if profiles.shape[0] < 2:
        raise ValueError(
            f"need at least 2 complete baseline dialogues, got {profiles.shape[0]}"
        )

    spread = profiles.std(axis=0, ddof=1)
    if np.all(spread < 1e-12):
        # Every dialogue has the same profile, so the reference distribution is
        # a point. Every distance would be zero and every dialogue would score
        # a perfect 1, which is worse than an error because it looks like an
        # answer. A single component with no variance is fine and is handled by
        # the estimators below.
        raise ValueError(
            "the baseline profiles show no variation, so every dialogue would "
            "score identically; fit on a corpus of distinct human dialogues"
        )

    mean = profiles.mean(axis=0)

    if estimator == "empirical":
        covariance = np.cov(profiles, rowvar=False)
        covariance = np.atleast_2d(covariance)
        return HumanBaseline(
            mean=mean, precision=np.linalg.pinv(covariance),
            components=list(components), n=profiles.shape[0], estimator=estimator,
        )

    from sklearn.covariance import LedoitWolf

    if estimator == "shrinkage":
        precision = LedoitWolf(assume_centered=False).fit(profiles).precision_
        return HumanBaseline(
            mean=mean, precision=precision, components=list(components),
            n=profiles.shape[0], estimator=estimator,
        )

    # scale_invariant
    scale = profiles.std(axis=0, ddof=1)
    # A component with no variance at all carries no information; leave its
    # scale at one rather than dividing by zero.
    scale = np.where(scale < 1e-12, 1.0, scale)
    standardised = (profiles - mean) / scale
    correlation = LedoitWolf(assume_centered=True).fit(standardised).covariance_
    covariance = np.outer(scale, scale) * correlation
    return HumanBaseline(
        mean=mean, precision=np.linalg.pinv(covariance), components=list(components),
        n=profiles.shape[0], estimator=estimator, scale=scale,
    )
