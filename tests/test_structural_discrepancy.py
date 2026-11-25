import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import multivariate_normal

from kefir_models.bayesian_observation import ObservationProblem, log_median, reference_prior
from kefir_models.structural_discrepancy import (
    StructuralProblem,
    correlation,
    state_space_parameters,
)


@pytest.mark.parametrize("kernel", ["ou", "matern32"])
def test_kernels_match_variance_and_e_folding_time(kernel):
    np.testing.assert_allclose(correlation(np.array([0.0, 2.0]), 0.5, kernel), [1.0, np.exp(-1.0)])


@pytest.mark.parametrize("kernel", ["ou", "matern32"])
def test_state_space_stationarity_and_covariance_match_kernel(kernel):
    z = reference_prior("ou").mean
    drift, noise, stationary = state_space_parameters(z, kernel)
    np.testing.assert_allclose(drift @ stationary + stationary @ drift.T + noise, 0.0, atol=1e-16)
    for lag in [0.0, 0.4, 2.0, 7.0]:
        value = (expm(drift * lag) @ stationary)[0, 0]
        np.testing.assert_allclose(
            value, 0.03**2 * correlation(np.array(lag), 0.5, kernel), rtol=1e-11
        )


@pytest.mark.parametrize("kernel", ["ou", "matern32"])
@pytest.mark.parametrize("sharing", ["shared", "independent"])
def test_reduced_likelihood_matches_full_joint_gaussian(kernel, sharing):
    z = reference_prior("ou").mean
    time = np.arange(5.0)
    residual = np.random.default_rng(12).normal(0, 0.02, (5, 3))
    values = np.exp(log_median(time, z, "ou")[:, None] + residual)
    problem = StructuralProblem(time, values, kernel, sharing)
    covariance = problem.latent_covariance(z, time, time)
    groups = np.ones((3, 3)) if sharing == "shared" else np.eye(3)
    full = np.kron(covariance, groups) + 0.03**2 * np.eye(15)
    expected = multivariate_normal.logpdf(residual.ravel(), cov=full) - np.log(values).sum()
    assert problem.log_likelihood(z) == pytest.approx(expected, abs=1e-10)


def test_independent_ou_is_exact_existing_reference():
    z = reference_prior("ou").mean
    time = np.arange(5.0)
    values = np.exp(
        log_median(time, z, "ou")[:, None] + np.random.default_rng(5).normal(0, 0.03, (5, 3))
    )
    base = ObservationProblem(time, values)
    alternative = StructuralProblem(time, values, "ou", "independent")
    assert base.log_posterior(z) == alternative.log_posterior(z)
    for first, second in zip(
        base.conditional_log_distribution(z, np.array([5.0, 6.0])),
        alternative.conditional_log_distribution(z, np.array([5.0, 6.0])),
    ):
        np.testing.assert_allclose(first, second, atol=1e-13)


@pytest.mark.parametrize("kernel", ["ou", "matern32"])
def test_shared_prediction_matches_full_gaussian_conditioning(kernel):
    z = reference_prior("ou").mean
    time = np.arange(4.0)
    future = np.array([4.0, 5.0])
    j = 3
    values = np.exp(
        log_median(time, z, "ou")[:, None] + np.random.default_rng(5).normal(0, 0.02, (4, j))
    )
    problem = StructuralProblem(time, values, kernel, "shared")
    training = np.kron(
        problem.latent_covariance(z, time, time), np.ones((j, j))
    ) + 0.03**2 * np.eye(len(time) * j)
    cross = np.kron(problem.latent_covariance(z, future, time), np.ones((j, j)))
    prior = np.kron(problem.latent_covariance(z, future, future), np.ones((j, j)))
    residual = (np.log(values) - log_median(time, z, "ou")[:, None]).ravel()
    expected_mean = np.repeat(log_median(future, z, "ou"), j) + cross @ np.linalg.solve(
        training, residual
    )
    expected_cov = prior - cross @ np.linalg.solve(training, cross.T)
    mean, cov, noise = problem.conditional_log_distribution(z, future)
    np.testing.assert_allclose(mean.ravel(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(np.kron(cov, np.ones((j, j))), expected_cov, atol=1e-14)
    observed = np.exp(mean + 0.01)
    expected = (
        multivariate_normal.logpdf(
            np.log(observed).ravel(),
            mean=expected_mean,
            cov=expected_cov + noise * np.eye(len(future) * j),
        )
        - np.log(observed).sum()
    )
    assert problem.joint_log_prediction(z, future, observed) == pytest.approx(expected, abs=1e-10)
