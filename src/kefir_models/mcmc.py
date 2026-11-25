"""Finite-dimensional adaptive-warmup Metropolis and multichain diagnostics.

Adaptation stops before retained draws. No convergence diagnostic is a proof
of convergence; rank/folded split R-hat and bulk/tail ESS flag finite-run issues.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, rankdata


@dataclass(frozen=True)
class ChainResult:
    draws: np.ndarray
    acceptance: float


def metropolis_chain(
    target: Callable[[np.ndarray], float],
    start: np.ndarray,
    initial_covariance: np.ndarray,
    warmup: int,
    draws: int,
    seed: int,
) -> ChainResult:
    if warmup < 200 or draws < 20:
        raise ValueError("Need at least 200 warmup iterations and 20 retained draws.")
    rng = np.random.default_rng(seed)
    current = np.asarray(start).copy()
    value = target(current)
    if not np.isfinite(value):
        raise ValueError("Initial log density must be finite.")
    dimension = len(current)
    base = np.linalg.cholesky(initial_covariance)
    factor = base * (2.38 / np.sqrt(dimension))
    running_mean = current.copy()
    scatter = np.zeros((dimension, dimension))
    history = np.empty((draws, dimension))
    accepted = block_accepted = 0
    log_scale = 0.0
    for index in range(warmup + draws):
        proposal_factor = 0.1 * base if rng.random() < 0.05 else factor
        proposal = current + proposal_factor @ rng.normal(size=dimension)
        candidate = target(proposal)
        take = np.log(rng.random()) < candidate - value
        if take:
            current, value = proposal, candidate
        if index < warmup:
            block_accepted += take
            delta = current - running_mean
            running_mean += delta / (index + 2)
            scatter += np.outer(delta, current - running_mean)
            if (index + 1) % 100 == 0:
                log_scale += (block_accepted / 100 - 0.234) / np.sqrt((index + 1) / 100)
                block_accepted = 0
                covariance = (
                    initial_covariance
                    if index < 499
                    else (scatter / (index + 1) + 1e-8 * np.eye(dimension))
                )
                factor = (
                    np.linalg.cholesky(covariance) * (2.38 / np.sqrt(dimension)) * np.exp(log_scale)
                )
        else:
            history[index - warmup] = current
            accepted += take
    return ChainResult(history, accepted / draws)


def split_chains(chains: np.ndarray) -> np.ndarray:
    chains = np.asarray(chains)
    if chains.ndim != 2 or len(chains) < 2 or chains.shape[1] < 20:
        raise ValueError("Diagnostics require at least two chains of twenty draws.")
    half = chains.shape[1] // 2
    return np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)


def rank_normalize(chains: np.ndarray) -> np.ndarray:
    ranks = rankdata(chains.ravel(), method="average")
    return norm.ppf((ranks - 0.375) / (len(ranks) + 0.25)).reshape(chains.shape)


def ordinary_rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    within = np.var(chains, axis=1, ddof=1).mean()
    if within <= 0:
        return np.inf
    variance = (n - 1) / n * within + np.var(chains.mean(axis=1), ddof=1)
    return float(np.sqrt(variance / within))


def effective_sample_size(chains: np.ndarray) -> float:
    """Multichain initial-positive, initial-monotone paired autocorrelation ESS."""

    m, n = chains.shape
    centered = chains - chains.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    transform = np.fft.rfft(centered, n=size, axis=1)
    covariance = np.fft.irfft(transform * transform.conjugate(), n=size, axis=1)[:, :n] / n
    within = np.var(chains, axis=1, ddof=1).mean()
    variance = (n - 1) / n * within + np.var(chains.mean(axis=1), ddof=1)
    if variance <= 0:
        return 0.0
    rho = 1 - (within - covariance.mean(axis=0)) / variance
    rho[0] = 1.0
    pairs = []
    for index in range(0, n - 1, 2):
        pair = rho[index] + rho[index + 1]
        if pair < 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    tau = max(-1 + 2 * sum(pairs), 1.0)
    return float(min(m * n / tau, m * n))


def diagnostics(chains: np.ndarray) -> dict[str, float]:
    split = split_chains(chains)
    ranked = rank_normalize(split)
    folded = rank_normalize(np.abs(split - np.median(split)))
    low, high = np.quantile(split, [0.05, 0.95])
    return {
        "rank_split_rhat": max(ordinary_rhat(ranked), ordinary_rhat(folded)),
        "bulk_ess": effective_sample_size(ranked),
        "tail_ess": min(
            effective_sample_size((split <= low).astype(float)),
            effective_sample_size((split >= high).astype(float)),
        ),
    }
