"""Logistic population median with an exact Gaussian latent OU observation model.

Time is expressed in fixed 12-original-unit intervals, independent of any
training cutoff. Responses retain their original numerical units. The OU
process is a deviation in log response, not the earlier logistic diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular


@dataclass(frozen=True)
class Prior:
    """Independent normal priors on explicitly transformed coordinates."""

    mean: np.ndarray
    sd: np.ndarray


def parameter_indices(kind: str) -> list[int]:
    if kind not in {"ou", "iid"}:
        raise ValueError("kind must be 'ou' or 'iid'.")
    return list(range(6)) if kind == "ou" else [0, 1, 2, 5]


def reference_prior(kind: str, noise_multiplier: float = 1.0) -> Prior:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive.")
    mean = np.log([10.0, 3.0, 0.5, 0.5, 0.03 * noise_multiplier, 0.03 * noise_multiplier])
    sd = np.array([1.5, 1.0, 1.25, 1.25, 1.0, 1.0])
    indices = parameter_indices(kind)
    return Prior(mean[indices], sd[indices])


def physical_parameters(z: np.ndarray, kind: str) -> dict[str, float]:
    full = np.zeros(6)
    full[parameter_indices(kind)] = z
    with np.errstate(over="ignore", invalid="ignore"):
        initial, margin, rate, decay, process_sd, observation_sd = np.exp(full)
    if kind == "iid":
        process_sd = 0.0
    return {
        "initial": initial,
        "capacity": initial * (1 + margin),
        "margin": margin,
        "rate_per_interval": rate,
        "decay_per_interval": decay,
        "process_sd": process_sd,
        "observation_sd": observation_sd,
    }


def log_median(time: np.ndarray, z: np.ndarray, kind: str) -> np.ndarray:
    # Stable even when the logistic ratio or normalized elapsed time is large.
    full = np.zeros(6)
    full[parameter_indices(kind)] = z
    return (
        full[0]
        + np.logaddexp(0.0, full[1])
        - np.logaddexp(0.0, full[1] - np.exp(full[2]) * np.asarray(time))
    )


@dataclass
class ObservationProblem:
    time: np.ndarray
    values: np.ndarray
    kind: str = "ou"
    noise_multiplier: float = 1.0

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        parameter_indices(self.kind)
        if (
            self.time.ndim != 1
            or self.values.ndim != 2
            or len(self.time) != len(self.values)
            or len(self.time) < 2
        ):
            raise ValueError("Expected at least two times and a time-by-trial response matrix.")
        if not np.isfinite(self.values).all() or np.any(self.values <= 0):
            raise ValueError("Responses must be finite and strictly positive.")
        if not np.isfinite(self.time).all() or np.any(np.diff(self.time) <= 0):
            raise ValueError("Times must be finite and increasing.")
        self.log_values = np.log(self.values)
        self.distance = np.abs(self.time[:, None] - self.time[None, :])
        self.prior = reference_prior(self.kind, self.noise_multiplier)

    def covariance(self, z: np.ndarray) -> np.ndarray:
        p = physical_parameters(z, self.kind)
        covariance = p["process_sd"] ** 2 * np.exp(-p["decay_per_interval"] * self.distance)
        return covariance + p["observation_sd"] ** 2 * np.eye(len(self.time))

    def log_likelihood(self, z: np.ndarray) -> float:
        p = physical_parameters(z, self.kind)
        if not np.isfinite(list(p.values())).all() or p["observation_sd"] <= 0:
            return -np.inf
        try:
            factor = np.linalg.cholesky(self.covariance(z))
            residual = self.log_values - log_median(self.time, z, self.kind)[:, None]
            white = solve_triangular(factor, residual, lower=True, check_finite=False)
            count, trials = self.values.shape
            # The response-scale Jacobian is constant in z but is retained for
            # a correctly normalized likelihood and predictive-score checks.
            return float(
                -0.5
                * (
                    count * trials * np.log(2 * np.pi)
                    + 2 * trials * np.log(factor.diagonal()).sum()
                    + (white**2).sum()
                )
                - self.log_values.sum()
            )
        except np.linalg.LinAlgError:
            return -np.inf

    def log_posterior(self, z: np.ndarray) -> float:
        z = np.asarray(z)
        if z.shape != self.prior.mean.shape or not np.isfinite(z).all():
            return -np.inf
        standardized = (z - self.prior.mean) / self.prior.sd
        prior = -0.5 * np.sum(standardized**2 + np.log(2 * np.pi * self.prior.sd**2))
        return float(prior + self.log_likelihood(z))

    def conditional_log_distribution(
        self, z: np.ndarray, prediction_time: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Conditional latent log response mean/covariance and measurement variance.

        At training times this predicts latent states and new measurement
        replicates, rather than conditioning a value on itself with zero error.
        """

        times = np.asarray(prediction_time)
        p = physical_parameters(z, self.kind)
        cross = p["process_sd"] ** 2 * np.exp(
            -p["decay_per_interval"] * np.abs(times[:, None] - self.time[None, :])
        )
        prior = p["process_sd"] ** 2 * np.exp(
            -p["decay_per_interval"] * np.abs(times[:, None] - times[None, :])
        )
        factor = np.linalg.cholesky(self.covariance(z))
        residual = self.log_values - log_median(self.time, z, self.kind)[:, None]
        mean = log_median(times, z, self.kind)[:, None] + cross @ cho_solve(
            (factor, True), residual, check_finite=False
        )
        covariance = prior - cross @ cho_solve((factor, True), cross.T, check_finite=False)
        covariance = (covariance + covariance.T) / 2
        return mean, covariance, p["observation_sd"] ** 2


def kalman_log_likelihood(problem: ObservationProblem, z: np.ndarray) -> float:
    """Independent exact scalar Kalman implementation for likelihood validation."""

    p = physical_parameters(z, problem.kind)
    residual = problem.log_values - log_median(problem.time, z, problem.kind)[:, None]
    mean = np.zeros(problem.values.shape[1])
    variance = p["process_sd"] ** 2
    result = 0.0
    for index, time in enumerate(problem.time):
        if index:
            phi = np.exp(-p["decay_per_interval"] * (time - problem.time[index - 1]))
            mean *= phi
            variance = phi**2 * variance + p["process_sd"] ** 2 * (
                -np.expm1(-2 * p["decay_per_interval"] * (time - problem.time[index - 1]))
            )
        innovation_variance = variance + p["observation_sd"] ** 2
        innovation = residual[index] - mean
        result += -0.5 * (
            len(mean) * np.log(2 * np.pi * innovation_variance)
            + (innovation**2).sum() / innovation_variance
        )
        gain = variance / innovation_variance
        mean += gain * innovation
        variance *= 1 - gain
    return float(result - problem.log_values.sum())
