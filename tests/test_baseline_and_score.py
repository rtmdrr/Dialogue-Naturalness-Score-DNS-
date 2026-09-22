import json

import numpy as np
import pytest
from scipy.stats import chi2

from deep_persona_dns.baseline import ESTIMATORS, HumanBaseline, fit_baseline
from deep_persona_dns.score import (
    dns_score,
    is_indistinguishable,
    lambda_for_df,
    mahalanobis_sq,
    naturalness_pvalue,
)

COMPONENTS = ["pragmatics", "joint_attention", "emotion"]


@pytest.fixture
def profiles():
    rng = np.random.default_rng(0)
    return np.column_stack([
        rng.normal(0.90, 0.05, 300),
        rng.normal(0.60, 0.20, 300),
        rng.normal(1.20, 0.90, 300),
    ])


@pytest.fixture
def baseline(profiles):
    return fit_baseline(profiles, COMPONENTS)


# ------------------------------------------------------------------ fitting

@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_every_estimator_produces_a_usable_fit(profiles, estimator):
    fitted = fit_baseline(profiles, COMPONENTS, estimator=estimator)
    assert fitted.n == len(profiles)
    assert fitted.mean.shape == (3,)
    assert np.isfinite(fitted.precision).all()
    assert fitted.estimator == estimator


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_a_constant_component_does_not_break_the_fit(profiles, estimator):
    constant = profiles.copy()
    constant[:, 1] = 0.5
    fitted = fit_baseline(constant, COMPONENTS, estimator=estimator)
    assert np.isfinite(fitted.precision).all()


def test_rows_with_a_missing_component_are_dropped(profiles):
    with_gap = profiles.copy()
    with_gap[0, 0] = np.nan
    assert fit_baseline(with_gap, COMPONENTS).n == len(profiles) - 1


def test_a_baseline_with_no_variation_is_refused():
    identical = np.tile([0.9, 0.6, 1.2], (20, 1))
    with pytest.raises(ValueError, match="no variation"):
        fit_baseline(identical, COMPONENTS)


def test_too_few_dialogues_is_refused():
    with pytest.raises(ValueError, match="at least 2"):
        fit_baseline(np.array([[1.0, 2.0, 3.0]]), COMPONENTS)


def test_column_count_must_match_the_component_names(profiles):
    with pytest.raises(ValueError, match="component names"):
        fit_baseline(profiles, ["only", "two"])


def test_unknown_estimator_is_refused(profiles):
    with pytest.raises(ValueError, match="estimator"):
        fit_baseline(profiles, COMPONENTS, estimator="oracle")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_baseline_survives_a_json_round_trip(profiles, estimator):
    fitted = fit_baseline(profiles, COMPONENTS, estimator=estimator)
    restored = HumanBaseline.from_dict(json.loads(json.dumps(fitted.to_dict())))
    assert restored.components == fitted.components
    assert restored.estimator == fitted.estimator
    assert np.allclose(restored.mean, fitted.mean)
    assert np.allclose(restored.precision, fitted.precision)


# ------------------------------------------------------------------ scoring

def test_distance_at_the_centroid_is_zero(baseline):
    assert mahalanobis_sq(baseline.mean, baseline)[0] == pytest.approx(0.0, abs=1e-9)


def test_distance_grows_with_departure_from_the_centroid(baseline, profiles):
    near = baseline.mean + 0.5 * profiles.std(axis=0)
    far = baseline.mean + 3.0 * profiles.std(axis=0)
    assert mahalanobis_sq(far, baseline)[0] > mahalanobis_sq(near, baseline)[0]


def test_a_single_profile_and_a_matrix_agree(baseline, profiles):
    single = mahalanobis_sq(profiles[0], baseline)
    many = mahalanobis_sq(profiles, baseline)
    assert single.shape == (1,)
    assert single[0] == pytest.approx(many[0])


def test_profile_width_must_match_the_baseline(baseline):
    with pytest.raises(ValueError, match="components"):
        mahalanobis_sq(np.zeros((2, 4)), baseline)


def test_score_is_one_at_the_centroid_and_falls_away(baseline, profiles):
    assert dns_score(baseline.mean, baseline)[0] == pytest.approx(1.0)
    far = baseline.mean + 10 * profiles.std(axis=0)
    assert dns_score(far, baseline)[0] < 0.01


def test_score_stays_within_its_range(baseline, profiles):
    scores = dns_score(profiles, baseline)
    assert ((scores > 0) & (scores <= 1)).all()


def test_lambda_matches_the_published_value_for_three_components():
    assert lambda_for_df(3) == pytest.approx(0.089, abs=5e-4)


def test_lambda_puts_the_half_way_point_on_the_chosen_quantile():
    for df in (3, 4):
        assert np.exp(-lambda_for_df(df) * chi2.ppf(0.95, df)) == pytest.approx(0.5)


def test_lambda_is_derived_from_the_baseline_width(profiles):
    three = fit_baseline(profiles, COMPONENTS)
    four = fit_baseline(
        np.column_stack([profiles, profiles[:, :1]]), COMPONENTS + ["congruence"]
    )
    point = np.full(3, 2.0)
    assert dns_score(point, three)[0] != pytest.approx(
        dns_score(np.append(point, 2.0), four)[0]
    )


def test_an_explicit_lambda_overrides_the_derived_one(baseline, profiles):
    point = baseline.mean + profiles.std(axis=0)
    assert dns_score(point, baseline, lam=1.0)[0] < dns_score(point, baseline)[0]


def test_non_positive_lambda_is_refused(baseline):
    with pytest.raises(ValueError, match="lam"):
        dns_score(baseline.mean, baseline, lam=0.0)


# --------------------------------------------------------------------- test

def test_the_baseline_itself_mostly_passes_its_own_test(profiles, baseline):
    assert is_indistinguishable(profiles, baseline).mean() > 0.9


def test_a_distant_profile_is_rejected(baseline, profiles):
    far = baseline.mean + 10 * profiles.std(axis=0)
    assert naturalness_pvalue(far, baseline)[0] < 0.01
    assert not is_indistinguishable(far, baseline)[0]


def test_p_values_are_probabilities(profiles, baseline):
    pvalues = naturalness_pvalue(profiles, baseline)
    assert ((pvalues >= 0) & (pvalues <= 1)).all()
