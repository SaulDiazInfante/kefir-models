import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import lognorm, norm

from kefir_models.bayesian_observation import log_median, reference_prior
from kefir_models.structural_discrepancy import StructuralProblem
from kefir_models.tempered_noise import TemperedNoiseProblem, median_matched_prior


@pytest.mark.parametrize("eta", [0.0, 2.0, 8.0])
@pytest.mark.parametrize("median", [0.015, 0.03, 0.06])
def test_prior_normalization_and_median_matching(eta, median):
    p = median_matched_prior(median, eta)
    np.testing.assert_allclose(p.cdf(np.log(median)), 0.5, atol=2e-11)
    np.testing.assert_allclose(np.exp(p.quantile(0.5)), median, rtol=1e-9)
    # Independent quadrature in SD rather than the implementation's mode-
    # centered log coordinate. The change-of-variable Jacobian is explicit.
    normalizer = quad(
        lambda b: lognorm.pdf(b, s=p.scale, scale=np.exp(p.location)) * np.exp(-eta * b * b),
        0.0,
        1.0,
        epsabs=1e-12,
    )[0]
    normalizer += quad(
        lambda b: lognorm.pdf(b, s=p.scale, scale=np.exp(p.location)) * np.exp(-eta * b * b),
        1.0,
        np.inf,
        epsabs=1e-12,
    )[0]
    assert np.exp(p.log_normalizer) == pytest.approx(normalizer, abs=2e-11)


@pytest.mark.parametrize("eta", [2.0, 8.0])
def test_exponential_moment_includes_boundary_and_rejects_beyond(eta):
    p = median_matched_prior(0.03, eta)
    assert p.log_square_mgf(eta) == pytest.approx(-p.log_normalizer)
    assert p.log_square_mgf(eta + 0.01) == np.inf
    t = 0.5
    direct = quad(
        lambda b: lognorm.pdf(b, s=p.scale, scale=np.exp(p.location))
        * np.exp(-(eta - t) * b * b - p.log_normalizer),
        0.0,
        2.0,
        epsabs=1e-12,
    )[0]
    direct += quad(
        lambda b: lognorm.pdf(b, s=p.scale, scale=np.exp(p.location))
        * np.exp(-(eta - t) * b * b - p.log_normalizer),
        2.0,
        np.inf,
        epsabs=1e-12,
    )[0]
    assert np.exp(p.log_square_mgf(t)) == pytest.approx(direct, abs=2e-11)


@pytest.mark.parametrize("sharing", ["independent", "shared"])
@pytest.mark.parametrize("kernel", ["ou", "matern32"])
def test_normalized_target_and_unmodified_likelihood(sharing, kernel):
    original_z = reference_prior("ou").mean
    time = np.arange(4.0)
    values = np.repeat(np.exp(log_median(time, original_z, "ou"))[:, None], 3, axis=1)
    p = TemperedNoiseProblem(time, values, kernel, sharing, eta=2.0)
    original = StructuralProblem(time, values, kernel, sharing)
    z = original_z + np.array([0.05, -0.1, 0.1, -0.2, 0.3, -0.4])
    assert p.log_likelihood(z) == original.log_likelihood(z)
    expected = (
        p.log_likelihood(z)
        + norm.logpdf(z[:4], loc=p.prior.mean[:4], scale=p.prior.sd[:4]).sum()
        + sum(p.noise_prior.logpdf(v) for v in z[4:])
    )
    assert p.log_posterior(z) == pytest.approx(expected, abs=1e-11)
    untempered = TemperedNoiseProblem(time, values, kernel, sharing, eta=0.0)
    assert untempered.log_posterior(z) == original.log_posterior(z)
    assert p.finite_response_moment(1.0)
    assert p.finite_response_moment(2.0)
    assert not p.finite_response_moment(2.01)


def test_tail_control_is_on_both_noise_components():
    time = np.arange(3.0)
    z = reference_prior("ou").mean
    values = np.exp(log_median(time, z, "ou"))[:, None]
    problem = TemperedNoiseProblem(time, values, eta=8.0)
    for coordinate in [4, 5]:
        point = z.copy()
        point[coordinate] = 4.0
        gaussian_prior = norm.logpdf(point, loc=problem.prior.mean, scale=problem.prior.sd).sum()
        correction = problem.log_posterior(point) - problem.log_likelihood(point) - gaussian_prior
        expected = -8 * np.exp(2 * point[4:]).sum() - 2 * problem.noise_prior.log_normalizer
        assert correction == pytest.approx(expected)
    assert problem.finite_response_moment(4.0)
    assert not problem.finite_response_moment(4.01)
