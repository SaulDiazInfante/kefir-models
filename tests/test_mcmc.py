import numpy as np
import pytest

from kefir_models.mcmc import diagnostics, metropolis_chain


def test_rank_diagnostics_flag_location_and_scale_disagreement():
    rng = np.random.default_rng(1)
    draws = rng.normal(size=(4, 2000))
    good = diagnostics(draws)
    assert good["rank_split_rhat"] < 1.01
    assert good["bulk_ess"] > 4000
    shifted = draws.copy()
    shifted[0] += 3
    assert diagnostics(shifted)["rank_split_rhat"] > 1.1
    scaled = draws.copy()
    scaled[0] *= 5
    assert diagnostics(scaled)["rank_split_rhat"] > 1.1


def test_metropolis_recovers_known_correlated_gaussian():
    covariance = np.array([[1.0, 0.8], [0.8, 2.0]])
    precision = np.linalg.inv(covariance)
    mean = np.array([0.5, -1.0])

    def target(x):
        delta = x - mean
        return -0.5 * delta @ precision @ delta

    results = [
        metropolis_chain(target, np.array([3.0, -3.0]), covariance, 1000, 5000, seed)
        for seed in [1, 2, 3, 4]
    ]
    draws = np.stack([result.draws for result in results])
    np.testing.assert_allclose(draws.mean(axis=(0, 1)), mean, atol=0.1)
    np.testing.assert_allclose(np.cov(draws.reshape(-1, 2).T), covariance, atol=0.15)
    assert all(0.1 < result.acceptance < 0.6 for result in results)
    assert diagnostics(draws[:, :, 0])["rank_split_rhat"] < 1.02


def test_autocorrelated_samples_have_smaller_ess():
    rng = np.random.default_rng(7)
    values = np.zeros((4, 5000))
    for index in range(1, values.shape[1]):
        values[:, index] = 0.8 * values[:, index - 1] + rng.normal(size=4)
    effective = diagnostics(values)["bulk_ess"]
    assert 1200 < effective < 3200


def test_sampler_rejects_invalid_start():
    with pytest.raises(ValueError):
        metropolis_chain(lambda x: -np.inf, np.zeros(2), np.eye(2), 1000, 1000, 1)
