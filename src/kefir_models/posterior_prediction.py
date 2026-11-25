"""Rao-Blackwellized pointwise quantiles of a Gaussian posterior mixture."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri


def gaussian_mixture_quantile(mean: np.ndarray, variance: np.ndarray, probability: float) -> float:
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if mean.shape != variance.shape or mean.ndim != 1 or not len(mean):
        raise ValueError("Expected matching nonempty vectors of component moments.")
    if not 0 < probability < 1 or np.any(variance <= 0):
        raise ValueError("Need positive variances and a probability strictly between zero and one.")
    deviation = np.sqrt(variance)
    component_quantiles = mean + ndtri(probability) * deviation
    left = float(component_quantiles.min()) - 1e-10
    right = float(component_quantiles.max()) + 1e-10
    return float(
        brentq(lambda x: ndtr((x - mean) / deviation).mean() - probability, left, right, xtol=1e-11)
    )
