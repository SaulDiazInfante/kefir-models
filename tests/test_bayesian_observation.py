import numpy as np
import pytest

from kefir_models.bayesian_observation import (
    ObservationProblem,
    kalman_log_likelihood,
    log_median,
    reference_prior,
)


def problem(kind="ou"):
    z = reference_prior(kind).mean.copy()
    time = np.arange(8.0)
    rng = np.random.default_rng(15)
    values = np.exp(log_median(time, z, kind)[:, None] + rng.normal(0, 0.04, (8, 3)))
    return ObservationProblem(time, values, kind), z


@pytest.mark.parametrize("kind", ["ou", "iid"])
def test_covariance_and_independent_kalman_likelihood_agree(kind):
    data, z = problem(kind)
    assert data.log_likelihood(z) == pytest.approx(kalman_log_likelihood(data, z), abs=1e-10)


def test_iid_likelihood_has_response_scale_jacobian():
    data, z = problem("iid")
    residual = np.log(data.values) - log_median(data.time, z, "iid")[:, None]
    variance = 0.03**2
    expected = (
        -0.5 * np.sum(np.log(2 * np.pi * variance) + residual**2 / variance)
        - np.log(data.values).sum()
    )
    assert data.log_likelihood(z) == pytest.approx(expected)


def test_conditional_ou_variance_matches_filter_then_exact_propagation():
    z = reference_prior("ou").mean
    times = np.array([0.0, 1.0])
    median = np.exp(log_median(times, z, "ou"))
    data = ObservationProblem(times, median[:, None], "ou")
    mean, covariance, noise = data.conditional_log_distribution(z, np.array([2.0]))
    phi = np.exp(-0.5)
    variance = 0.03**2
    for _ in times:
        variance = variance * 0.03**2 / (variance + 0.03**2)
        variance = phi**2 * variance + 0.03**2 * (1 - phi**2)
    assert covariance[0, 0] == pytest.approx(variance)
    assert mean[0, 0] == pytest.approx(log_median(np.array([2.0]), z, "ou")[0])
    assert noise == pytest.approx(0.03**2)


def test_future_observations_are_not_part_of_training_problem():
    data, z = problem()
    shortened = ObservationProblem(data.time[:5], data.values[:5], "ou")
    changed = data.values.copy()
    changed[5:] *= 100
    other = ObservationProblem(data.time[:5], changed[:5], "ou")
    assert shortened.log_posterior(z) == other.log_posterior(z)


def test_noise_prior_sensitivity_changes_prior_not_likelihood():
    data, z = problem()
    other = ObservationProblem(data.time, data.values, "ou", 2.0)
    assert data.log_likelihood(z) == other.log_likelihood(z)
    assert data.log_posterior(z) != other.log_posterior(z)


def test_positive_response_required():
    with pytest.raises(ValueError):
        ObservationProblem(np.array([0.0, 1.0]), np.array([[1.0], [0.0]]))
