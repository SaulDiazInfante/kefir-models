"""Smoothly tempered lognormal noise priors with controlled response moments.

The prior for z=log(SD) is proportional to
Normal(z; location, scale**2) * exp(-eta*exp(2*z)). Both noise SDs
are tempered. Their support remains (0,infinity). Kernel locations can be
calibrated to retain the reference prior medians; they are not prior means.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr

from kefir_models.bayesian_observation import Prior
from kefir_models.structural_discrepancy import StructuralProblem


@dataclass(frozen=True)
class TemperedLogSD:
    location: float
    scale: float = 1.0
    eta: float = 2.0

    def __post_init__(self):
        if (
            not np.isfinite([self.location, self.scale, self.eta]).all()
            or self.scale <= 0
            or self.eta < 0
        ):
            raise ValueError("Need finite location, positive scale, and nonnegative eta.")

    def kernel(self, z):
        if z > 350 and self.eta > 0:
            return -np.inf
        return -0.5 * ((z - self.location) / self.scale) ** 2 - (
            self.eta * np.exp(2 * z) if self.eta else 0.0
        )

    @property
    def quadrature(self):
        return _quadrature(self.location, self.scale, self.eta)

    @property
    def log_normalizer(self):
        if self.eta == 0:
            return 0.0
        _, width, peak, mass = self.quadrature
        return float(peak + np.log(width * mass) - np.log(self.scale * np.sqrt(2 * np.pi)))

    def logpdf(self, z):
        return self.kernel(z) - np.log(self.scale * np.sqrt(2 * np.pi)) - self.log_normalizer

    def cdf(self, z):
        if self.eta == 0:
            return float(ndtr((z - self.location) / self.scale))
        mode, width, peak, mass = self.quadrature
        upper = (z - mode) / width
        integral = quad(
            lambda u: np.exp(self.kernel(mode + width * u) - peak),
            -np.inf,
            upper,
            epsabs=1e-12,
            epsrel=1e-11,
        )[0]
        return float(np.clip(integral / mass, 0.0, 1.0))

    def quantile(self, probability):
        if not 0 < probability < 1:
            raise ValueError("Probability must be strictly between zero and one.")
        mode, _, _, _ = self.quadrature
        left, right = mode - 4 * self.scale, mode + 4 * self.scale
        step = 4 * self.scale
        while self.cdf(left) > probability:
            left -= step
            step *= 2
        step = 4 * self.scale
        while self.cdf(right) < probability:
            right += step
            step *= 2
        return brentq(lambda z: self.cdf(z) - probability, left, right, xtol=1e-11)

    def log_square_mgf(self, coefficient):
        """log E[exp(t*SD**2)]; finite exactly when t <= eta."""
        if not np.isfinite(coefficient):
            raise ValueError("Need a finite coefficient.")
        if coefficient > self.eta:
            return np.inf
        numerator = TemperedLogSD(self.location, self.scale, self.eta - coefficient)
        return numerator.log_normalizer - self.log_normalizer


@lru_cache(maxsize=512)
def _quadrature(location, scale, eta):
    if eta == 0:
        return location, scale, 0.0, np.sqrt(2 * np.pi)
    prior = TemperedLogSD(location, scale, eta)

    def derivative(z):
        return (z - location) / scale**2 + (2 * eta * np.exp(2 * z) if z < 350 else np.inf)

    left = location - scale
    step = scale
    while derivative(left) > 0:
        step *= 2
        left = location - step
    mode = brentq(derivative, left, location, xtol=1e-12)
    width = 1 / np.sqrt(1 / scale**2 + 4 * eta * np.exp(2 * mode))
    peak = prior.kernel(mode)
    mass = quad(
        lambda u: np.exp(prior.kernel(mode + width * u) - peak),
        -np.inf,
        np.inf,
        epsabs=1e-12,
        epsrel=1e-11,
    )[0]
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Prior quadrature failed.")
    return mode, width, peak, mass


@lru_cache(maxsize=64)
def median_matched_prior(median=0.03, eta=2.0, scale=1.0):
    if not np.isfinite(median) or median <= 0:
        raise ValueError("Need a finite positive SD median.")
    target = float(np.log(median))
    if eta == 0:
        return TemperedLogSD(target, scale, eta)

    def objective(location):
        return TemperedLogSD(location, scale, eta).cdf(target) - 0.5

    left, right, step = target - scale, target + scale, scale
    while objective(right) > 0:
        step *= 2
        right = target + step
    location = brentq(objective, left, right, xtol=1e-10)
    return TemperedLogSD(location, scale, eta)


class TemperedNoiseProblem(StructuralProblem):
    """The same likelihood and dynamical priors, with normalized noise tempering."""

    def __init__(self, time, values, kernel="ou", sharing="shared", eta=2.0, noise_multiplier=1.0):
        super().__init__(time, values, kernel, sharing, noise_multiplier)
        self.noise_prior = median_matched_prior(0.03 * noise_multiplier, eta)
        self.eta = eta
        location = self.prior.mean.copy()
        location[4:] = self.noise_prior.location
        # The inherited object supplies Gaussian kernel locations/scales and
        # initial proposals. These are not moments of the tempered prior.
        self.prior = Prior(location, self.prior.sd.copy())

    def log_posterior(self, z):
        base = super().log_posterior(z)
        if not np.isfinite(base) or self.eta == 0:
            return base
        with np.errstate(over="ignore"):
            penalty = self.eta * np.exp(2 * np.asarray(z)[4:]).sum()
        return float(base - penalty - 2 * self.noise_prior.log_normalizer)

    def finite_response_moment(self, order):
        """Global guarantee for nonnegative raw response moments, including boundary."""
        if not np.isfinite(order) or order < 0:
            raise ValueError("Need a finite nonnegative moment order.")
        return bool(order**2 / 2 <= self.eta)
