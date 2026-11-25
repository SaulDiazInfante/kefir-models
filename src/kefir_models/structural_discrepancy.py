"""Controlled temporal-kernel and cross-trial discrepancy alternatives.

All models retain the existing six transformed parameters and their priors.
The OU and Matérn-3/2 kernels share stationary variance and e-fold time.
Shared discrepancies describe a common structural departure, rather than
independent trial process noise. Exact Gaussian likelihoods integrate states.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import brentq

from kefir_models.bayesian_observation import ObservationProblem, log_median, physical_parameters

MATERN_EFOLD = brentq(lambda x: np.log1p(x) - x + 1, 1.0, 4.0, xtol=1e-14)


def correlation(distance: np.ndarray, decay: float, kernel: str) -> np.ndarray:
    if kernel not in {"ou", "matern32"}:
        raise ValueError("kernel must be ou or matern32.")
    with np.errstate(over="ignore", invalid="ignore"):
        x = np.asarray(distance) * decay * (MATERN_EFOLD if kernel == "matern32" else 1.0)
        result = np.exp(-x)
        if kernel == "matern32":
            result = (1 + x) * result
        return np.where(np.isposinf(x), 0.0, result)


class StructuralProblem(ObservationProblem):
    """Logistic population median plus shared or independent Gaussian deviations."""

    def __init__(self, time, values, kernel="ou", sharing="shared", noise_multiplier=1.0):
        if kernel not in {"ou", "matern32"} or sharing not in {"shared", "independent"}:
            raise ValueError("Invalid kernel or sharing specification.")
        self.kernel = kernel
        self.sharing = sharing
        super().__init__(time, values, "ou", noise_multiplier)

    def latent_covariance(self, z, first, second):
        p = physical_parameters(z, "ou")
        distance = np.abs(np.asarray(first)[:, None] - np.asarray(second)[None, :])
        return p["process_sd"] ** 2 * correlation(distance, p["decay_per_interval"], self.kernel)

    def covariance(self, z):
        p = physical_parameters(z, "ou")
        trials = self.values.shape[1] if self.sharing == "shared" else 1
        return self.latent_covariance(z, self.time, self.time) + p[
            "observation_sd"
        ] ** 2 / trials * np.eye(len(self.time))

    def log_likelihood(self, z):
        if self.sharing == "independent":
            return super().log_likelihood(z)
        p = physical_parameters(z, "ou")
        if not np.isfinite(list(p.values())).all() or p["observation_sd"] <= 0:
            return -np.inf
        residual = self.log_values - log_median(self.time, z, "ou")[:, None]
        return (
            self.shared_log_density(
                residual, self.latent_covariance(z, self.time, self.time), p["observation_sd"] ** 2
            )
            - self.log_values.sum()
        )

    @staticmethod
    def shared_log_density(residual, latent_covariance, noise):
        """Exact orthogonal decomposition into replicate mean and contrasts."""
        n, trials = residual.shape
        mean = residual.mean(axis=1)
        within = np.sum((residual - mean[:, None]) ** 2)
        try:
            # sqrt(J)*mean has unit-Jacobian coordinates in the orthogonal
            # replicate basis; contrasts have variance noise independently.
            factor = np.linalg.cholesky(trials * latent_covariance + noise * np.eye(n))
            white = solve_triangular(factor, np.sqrt(trials) * mean, lower=True, check_finite=False)
            return float(
                -0.5
                * (
                    n * trials * np.log(2 * np.pi)
                    + 2 * np.log(factor.diagonal()).sum()
                    + n * (trials - 1) * np.log(noise)
                    + white @ white
                    + within / noise
                )
            )
        except np.linalg.LinAlgError:
            return -np.inf

    def conditional_log_distribution(self, z, prediction_time):
        times = np.asarray(prediction_time)
        p = physical_parameters(z, "ou")
        cross = self.latent_covariance(z, times, self.time)
        prior = self.latent_covariance(z, times, times)
        factor = np.linalg.cholesky(self.covariance(z))
        residual = self.log_values - log_median(self.time, z, "ou")[:, None]
        if self.sharing == "shared":
            residual = residual.mean(axis=1, keepdims=True)
        mean = log_median(times, z, "ou")[:, None] + cross @ cho_solve(
            (factor, True), residual, check_finite=False
        )
        if self.sharing == "shared":
            mean = np.repeat(mean, self.values.shape[1], axis=1)
        covariance = prior - cross @ cho_solve((factor, True), cross.T, check_finite=False)
        return mean, (covariance + covariance.T) / 2, p["observation_sd"] ** 2

    def joint_log_prediction(self, z, times, observed):
        mean, latent, noise = self.conditional_log_distribution(z, times)
        residual = np.log(observed) - mean
        if self.sharing == "shared":
            return self.shared_log_density(residual, latent, noise) - np.log(observed).sum()
        factor = np.linalg.cholesky(latent + noise * np.eye(len(times)))
        white = solve_triangular(factor, residual, lower=True, check_finite=False)
        return float(
            -0.5
            * (
                observed.size * np.log(2 * np.pi)
                + 2 * observed.shape[1] * np.log(factor.diagonal()).sum()
                + (white**2).sum()
            )
            - np.log(observed).sum()
        )


def state_space_parameters(z, kernel):
    """Stationary exact linear SDE representation, used for independent validation."""
    p = physical_parameters(z, "ou")
    a = p["process_sd"]
    rate = p["decay_per_interval"]
    if kernel == "ou":
        return np.array([[-rate]]), np.array([[2 * rate * a * a]]), np.array([[a * a]])
    if kernel != "matern32":
        raise ValueError("Unknown kernel.")
    rate *= MATERN_EFOLD
    return (
        np.array([[0.0, 1.0], [-(rate**2), -2 * rate]]),
        np.array([[0.0, 0.0], [0.0, 4 * a * a * rate**3]]),
        np.diag([a * a, a * a * rate**2]),
    )
