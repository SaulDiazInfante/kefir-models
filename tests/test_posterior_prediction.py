import numpy as np
import pytest
from scipy.special import ndtr, ndtri

from kefir_models.posterior_prediction import gaussian_mixture_quantile


def test_single_gaussian_quantiles():
    for probability in [0.05, 0.5, 0.95]:
        assert gaussian_mixture_quantile(
            np.array([2.0]), np.array([9.0]), probability
        ) == pytest.approx(2 + 3 * ndtri(probability))


def test_mixture_quantile_solves_actual_mixture_cdf():
    means = np.array([-1.0, 1.0])
    variance = np.array([1.0, 1.0])
    assert gaussian_mixture_quantile(means, variance, 0.5) == pytest.approx(0.0, abs=1e-10)
    value = gaussian_mixture_quantile(means, variance, 0.95)
    assert ndtr(value - means).mean() == pytest.approx(0.95, abs=1e-10)
