"""Experimental Gaussian moment transitions for scalar Itô diffusions.

This is a Gaussian assumed-density approximation, not an exact nonlinear SDE
transition density. Each internal Euler step propagates Gaussian quadrature
moments by the laws of total expectation and total variance.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from kefir_models.sde_compare import NeuralSDEFunction, TransitionBatch


def gaussian_transition_moments(
    model: NeuralSDEFunction, batch: TransitionBatch, substeps: int
) -> tuple[Tensor, Tensor]:
    """Return conditional mean/variance, adding the legacy variance floor once.

    Three-point Gauss-Hermite quadrature has nodes {-sqrt(3), 0, sqrt(3)}
    and weights {1/6, 2/3, 1/6} for a standard normal. Gaussian projection
    after every Euler step introduces closure error for nonlinear models.
    More substeps address time discretization, not that closure error.
    """

    if substeps < 1:
        raise ValueError("substeps must be positive.")
    if bool((batch.delta_time <= 0).any()):
        raise ValueError("Transition durations must be positive.")
    if substeps == 1:
        mean = batch.current + model.drift(batch.time, batch.current) * batch.delta_time
        variance = model.diffusion(batch.time, batch.current).square() * batch.delta_time
        return mean, variance + 1e-6

    mean = batch.current
    variance = torch.zeros_like(mean)
    nodes = mean.new_tensor([-math.sqrt(3), 0.0, math.sqrt(3)]).reshape(1, 3, 1)
    weights = mean.new_tensor([1 / 6, 2 / 3, 1 / 6]).reshape(1, 3, 1)
    step = batch.delta_time / substeps
    for index in range(substeps):
        # The tiny floor only regularizes sqrt's derivative at the initial
        # point mass; it is not an estimated process-noise contribution.
        states = mean[:, None] + variance.clamp_min(1e-12).sqrt()[:, None] * nodes
        time = (batch.time + index * step)[:, None].expand_as(states)
        shifted = states + model.drift(time, states) * step[:, None]
        next_mean = (weights * shifted).sum(dim=1)
        variance = (weights * (shifted - next_mean[:, None]).square()).sum(dim=1)
        variance = variance + (weights * model.diffusion(time, states).square()).sum(dim=1) * step
        mean = next_mean
    return mean, variance + 1e-6


def gaussian_moment_nll(model: NeuralSDEFunction, batch: TransitionBatch, substeps: int) -> Tensor:
    mean, variance = gaussian_transition_moments(model, batch, substeps)
    return (
        0.5 * (torch.log(2 * math.pi * variance) + (batch.following - mean).square() / variance)
    ).mean()
