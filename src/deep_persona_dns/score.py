"""Mahalanobis distance, the DNS score, and the naturalness test.

Given a fitted human baseline, a dialogue's scoring profile is compared with
it by squared Mahalanobis distance. Two readings follow:

* a **score**, ``DNS = exp(-lambda * d^2)``, which is 1 at the centroid and
  falls towards 0 as the profile departs from human behaviour;
* a **test**, since under the assumption that human profiles are multivariate
  normal, ``d^2`` follows a chi-squared distribution with as many degrees of
  freedom as there are components. A non-significant result means the profile
  is not distinguishable from the human baseline.

``lambda`` sets how quickly the score decays. The default places the half-way
point of the score at the 95th percentile of the reference distribution, so
that a dialogue on the significance boundary scores 0.5 whatever the number of
components.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
from scipy.stats import chi2

from .baseline import HumanBaseline

#: Quantile of the chi-squared distribution at which DNS equals 0.5.
DEFAULT_HALF_LIFE_QUANTILE = 0.95


def lambda_for_df(df: int, quantile: float = DEFAULT_HALF_LIFE_QUANTILE) -> float:
    """The decay constant that puts DNS at 0.5 on a given chi-squared quantile.

    With three components and the default quantile this is 0.089.
    """
    if df < 1:
        raise ValueError("df must be at least 1")
    return float(np.log(2) / chi2.ppf(quantile, df))


def _as_matrix(profiles: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    array = np.atleast_2d(np.asarray(profiles, dtype=float))
    return array


def mahalanobis_sq(
    profiles: Union[np.ndarray, Sequence[float]],
    baseline: HumanBaseline,
) -> np.ndarray:
    """Squared Mahalanobis distance from each profile to the baseline centroid.

    Accepts a single profile or a matrix of them, and always returns a
    one-dimensional array.
    """
    matrix = _as_matrix(profiles)
    if matrix.shape[1] != baseline.n_components:
        raise ValueError(
            f"profiles have {matrix.shape[1]} components but the baseline was "
            f"fitted on {baseline.n_components} ({list(baseline.components)})"
        )
    delta = matrix - baseline.mean
    return np.einsum("ij,jk,ik->i", delta, baseline.precision, delta)


def dns_score(
    profiles: Union[np.ndarray, Sequence[float]],
    baseline: HumanBaseline,
    *,
    lam: Optional[float] = None,
    quantile: float = DEFAULT_HALF_LIFE_QUANTILE,
) -> np.ndarray:
    """The Dialogue Naturalness Score of each profile.

    Args:
        profiles: one profile or a matrix of them.
        baseline: the fitted human reference.
        lam: decay constant. Derived from the number of components when
            omitted, which is the recommended use; pass a value to reproduce
            results computed with a particular constant.
        quantile: used when deriving ``lam``.

    Returns:
        Scores in ``(0, 1]``, higher meaning closer to human behaviour.
    """
    if lam is None:
        lam = lambda_for_df(baseline.n_components, quantile)
    if lam <= 0:
        raise ValueError("lam must be positive")
    return np.exp(-lam * mahalanobis_sq(profiles, baseline))


def naturalness_pvalue(
    profiles: Union[np.ndarray, Sequence[float]],
    baseline: HumanBaseline,
    *,
    df: Optional[int] = None,
) -> np.ndarray:
    """p-values of the chi-squared naturalness test.

    The null hypothesis is that the profile is drawn from the human baseline
    distribution. A large p-value means the dialogue is not distinguishable
    from human behaviour on these components; a small one means it departs
    from the baseline.

    Note that this is a test of profile typicality, not of quality, and that
    failing to reject a null is not evidence for it -- a small baseline or
    noisy components make rejection unlikely regardless of the dialogue.
    """
    if df is None:
        df = baseline.n_components
    return chi2.sf(mahalanobis_sq(profiles, baseline), df)


def is_indistinguishable(
    profiles: Union[np.ndarray, Sequence[float]],
    baseline: HumanBaseline,
    *,
    alpha: float = 0.05,
    df: Optional[int] = None,
) -> np.ndarray:
    """Whether each profile passes the naturalness test at level ``alpha``."""
    return naturalness_pvalue(profiles, baseline, df=df) > alpha
